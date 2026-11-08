# SPDX-License-Identifier: Apache-2.0
"""Exercise real persistence and streaming orchestration, not model quality."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, redirect_stderr, redirect_stdout
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import importlib.util
import io
import json
import os
from pathlib import Path
import select
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from personal_threats import (
    AnalysisReader,
    AnalysisSettings,
    ContractError,
    HistoryStore,
    Message,
    Prediction,
    Scores,
    StoreError,
    ThreatAnalyzer,
    WriterBusy,
)
from personal_threats.cli import main
from personal_threats.core import canonical_json
from personal_threats.model import ModelError
from personal_threats.review import QualityProfile
from personal_threats.store import APPLICATION_ID, SCHEMA


ROOT = Path(__file__).resolve().parents[1]


def message(number=1, text="hello мир", *, dialog="d", **metadata):
    stamp = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(
        seconds=number
    )
    return Message(
        dialog,
        str(number),
        "sender",
        text,
        {"timestamp": stamp.isoformat(), **metadata},
    )


class FixtureScorer:
    """Explicit numeric fixture for transaction and scheduling tests."""

    def __init__(self, digest="a" * 64, hook=None):
        self.bundle = SimpleNamespace(
            manifest_sha256=digest,
            manifest={
                "model_version": "numeric-fixture-1",
                "purpose": "test_fixture",
                "task": "window_features_v3",
            },
        )
        self.hook = hook
        self.calls = []

    def runtime_configuration(self, *, batch_size):
        return {
            "bundle_sha256": self.bundle.manifest_sha256,
            "backend": "explicit_numeric_fixture",
            "batch_size": batch_size,
        }

    def score(self, current, *, context_text, batch_size):
        self.calls.append((current, context_text))
        if self.hook is not None:
            self.hook(current)
        return Prediction(
            current.dialog_id,
            current.message_id,
            current.sender_id,
            self.bundle.manifest["model_version"],
            self.bundle.manifest_sha256,
            Scores(0.8, (0.7, 0.1, 0.1, 0.1, 0.1, 0.1)),
            model_purpose="test_fixture",
            window_count=1,
        )


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "history.sqlite3"

    def analyzer(self, scorer=None, **kwargs):
        result = ThreatAnalyzer(
            self.path,
            scorer or FixtureScorer(),
            allow_test_fixture=True,
            **kwargs,
        )
        self.addCleanup(result.close)
        return result

    def test_fixture_requires_explicit_opt_in_before_opening_the_database(
        self,
    ):
        with self.assertRaisesRegex(ModelError, "opt-in"):
            ThreatAnalyzer(self.path, FixtureScorer())
        self.assertFalse(self.path.exists())

    def test_streaming_input_scores_every_message_and_restores_after_restart(
        self,
    ):
        scorer = FixtureScorer()
        analyzer = self.analyzer(scorer)
        results = list(
            analyzer.process_messages(
                (message(i) for i in range(1, 8)), batch_size=3
            )
        )
        self.assertEqual(len(results), 7)
        self.assertTrue(all(r.status == "assessed" for r in results))
        self.assertTrue(all(r.decision is None for r in results))
        self.assertEqual(analyzer.statistics()["positive_messages"], None)
        self.assertEqual(analyzer.statistics()["uncertain_messages"], 7)
        self.assertEqual(analyzer.statistics()["outstanding_jobs"], 0)
        signature = analyzer.configuration_sha256
        analyzer.close()
        replacement = self.analyzer()
        self.assertEqual(replacement.configuration_sha256, signature)
        self.assertEqual(list(replacement.process_pending()), [])
        self.assertEqual(len(list(replacement.results("d"))), 7)

    def test_edit_recalculates_suffix_and_preserves_original_assessments(self):
        scorer = FixtureScorer()
        analyzer = self.analyzer(scorer)
        list(analyzer.process_messages(message(i) for i in (1, 2, 3)))
        old_ids = {r.assessment_id for r in analyzer.results("d")}
        updated = list(analyzer.process_message(message(2, "new text")))
        self.assertEqual([r.message_id for r in updated], ["2", "3"])
        self.assertTrue(all(r.basis == "recomputed" for r in updated))
        self.assertIn("new text", scorer.calls[-1][1])
        all_rows = list(analyzer.store.assessments("d", current=False))
        self.assertEqual(len(all_rows), 5)
        self.assertTrue(old_ids <= {r["assessment_id"] for r in all_rows})
        self.assertEqual(analyzer.statistics()["assessed_messages"], 3)

    def test_late_message_never_includes_future_context(self):
        scorer = FixtureScorer()
        analyzer = self.analyzer(scorer)
        list(analyzer.process_message(message(3, "future")))
        list(analyzer.process_message(message(1, "earlier")))
        contexts = [c for m, c in scorer.calls if m.message_id == "1"]
        self.assertEqual(contexts, [""])
        self.assertEqual(analyzer.statistics()["assessed_messages"], 2)

    def test_deleted_history_policy_is_persisted_and_changed_atomically(self):
        scorer = FixtureScorer()
        analyzer = self.analyzer(scorer)
        list(analyzer.process_messages([message(1), message(2)]))
        analyzer.store.mark_deleted("d", "1")
        list(analyzer.process_pending("d"))
        self.assertTrue(
            list(analyzer.results("d"))[0].context["current_message_deleted"]
        )
        self.assertIn("[deleted]", scorer.calls[-1][1])
        self.assertTrue(
            list(analyzer.results("d"))[-1].context[
                "reliance_on_deleted_context"
            ]
        )
        analyzer.close()
        visible = FixtureScorer()
        replacement = self.analyzer(
            visible,
            settings=AnalysisSettings(context_policy="visible_history"),
        )
        changed = list(replacement.process_pending("d"))
        self.assertEqual(len(changed), 2)
        self.assertEqual(visible.calls[-1][1], "")
        self.assertTrue(all(r.basis == "recomputed" for r in changed))

    def test_edit_during_inference_drops_stale_output_and_scores_latest(self):
        analyzer = self.analyzer()
        changed = False

        def revise(current):
            nonlocal changed
            if not changed:
                changed = True
                analyzer.ingest(message(1, "edited during inference"))

        analyzer.scorer.hook = revise
        results = list(analyzer.process_message(message()))
        self.assertEqual(
            [r.status for r in results], ["superseded", "assessed"]
        )
        self.assertIsNone(results[0].prediction)
        self.assertEqual(results[1].revision_seq, 2)
        self.assertEqual(analyzer.statistics()["archived_assessments"], 1)

    def test_another_thread_can_ingest_an_edit_while_scoring_is_paused(self):
        paused = threading.Event()
        resume = threading.Event()
        first = True

        def hook(current):
            nonlocal first
            if first:
                first = False
                paused.set()
                if not resume.wait(10):
                    raise RuntimeError("Concurrent edit test did not resume")

        analyzer = self.analyzer(FixtureScorer(hook=hook))
        analyzer.ingest(message())
        with ThreadPoolExecutor(2) as executor:
            worker = executor.submit(lambda: list(analyzer.process_pending()))
            try:
                self.assertTrue(paused.wait(5))
                edit = executor.submit(analyzer.ingest, message(text="new"))
                self.assertEqual(edit.result(3).status, "revised")
            finally:
                resume.set()
            results = worker.result(10)
        self.assertEqual(
            [r.status for r in results], ["superseded", "assessed"]
        )
        self.assertEqual(analyzer.statistics()["current_assessments"], 1)

    def test_backend_failure_blocks_only_its_dialogue_and_can_be_retried(self):
        failing = True

        def hook(current):
            if failing and current.dialog_id == "a":
                raise ValueError("No private message text should reach logs")

        analyzer = self.analyzer(FixtureScorer(hook=hook))
        results = list(
            analyzer.process_messages(
                [
                    message(1, dialog="a"),
                    message(2, dialog="a"),
                    message(1, dialog="b"),
                    message(2, dialog="b"),
                ]
            )
        )
        self.assertEqual([r.status for r in results].count("failed"), 1)
        self.assertEqual(analyzer.statistics("b")["assessed_messages"], 2)
        self.assertEqual(analyzer.statistics("a")["assessed_messages"], 0)
        self.assertEqual(analyzer.statistics("a")["failed_jobs"], 1)
        failing = False
        analyzer.store.retry_failed("a")
        self.assertEqual(len(list(analyzer.process_pending("a"))), 2)

    def test_keyboard_interrupt_returns_owned_work_to_pending(self):
        def hook(current):
            raise KeyboardInterrupt

        analyzer = self.analyzer(FixtureScorer(hook=hook))
        with self.assertRaises(KeyboardInterrupt):
            list(analyzer.process_message(message()))
        self.assertEqual(analyzer.statistics()["failed_jobs"], 0)
        analyzer.scorer.hook = None
        self.assertEqual(len(list(analyzer.process_pending())), 1)

    def test_different_model_cannot_complete_a_new_binding_with_old_weights(
        self,
    ):
        store = HistoryStore(self.path)
        self.addCleanup(store.close)
        first = ThreatAnalyzer(store, FixtureScorer(), allow_test_fixture=True)
        second = ThreatAnalyzer(
            store, FixtureScorer("b" * 64), allow_test_fixture=True
        )
        first.scorer.hook = lambda current: store.bind_analysis(
            second.configuration
        )
        result = list(first.process_message(message()))
        self.assertEqual([r.status for r in result], ["superseded"])
        self.assertEqual(store.statistics()["current_assessments"], 0)
        self.assertEqual(len(list(second.process_pending())), 1)
        self.assertEqual(list(first.results()), [])
        self.assertEqual(
            list(second.results())[0].prediction.model_sha256, "b" * 64
        )

    def test_threshold_change_reuses_scores_and_keeps_raw_predictions(self):
        store = HistoryStore(self.path)
        self.addCleanup(store.close)
        scorer = FixtureScorer()
        first = ThreatAnalyzer(store, scorer, allow_test_fixture=True)
        list(first.process_message(message()))
        profile = QualityProfile("balanced", "a" * 64, (0.5,) * 7)
        second = ThreatAnalyzer(
            store, scorer, profile=profile, allow_test_fixture=True
        )
        self.assertEqual(list(second.process_pending()), [])
        self.assertEqual(len(scorer.calls), 1)
        result = list(second.results())[0]
        self.assertTrue(result.decision["threat_present"])
        self.assertFalse(result.decision["production_ready_detector"])
        self.assertIsNone(result.prediction.to_dict()["threat_present"])

    def test_reader_needs_no_writer_lock_and_paging_is_bounded(self):
        analyzer = self.analyzer()
        list(analyzer.process_messages(message(i) for i in range(1, 5)))
        with AnalysisReader(self.path) as reader:
            self.assertEqual(len(list(reader.results("d"))), 4)
            self.assertFalse(reader.statistics()["model_available"])
            with self.assertRaises(WriterBusy):
                HistoryStore(self.path)
            pages = reader.store.assessments(page_size=1)
            first = next(pages)
            list(analyzer.process_message(message(5)))
            remainder = list(pages)
            self.assertEqual(len(remainder), 3)
            later = list(reader.results("d", after_id=first["assessment_id"]))
            self.assertEqual(len(later), 4)

    def test_input_error_preserves_committed_prefix_and_pending_jobs(self):
        analyzer = self.analyzer()

        def incoming():
            yield message()
            raise ContractError("bad input")

        with self.assertRaises(ContractError):
            list(analyzer.process_messages(incoming(), batch_size=64))
        self.assertEqual(analyzer.statistics()["messages"], 1)
        self.assertEqual(analyzer.statistics()["outstanding_jobs"], 1)
        self.assertEqual(analyzer.statistics()["assessed_messages"], 0)
        self.assertEqual(len(list(analyzer.process_pending())), 1)

    def test_work_limit_still_ingests_and_queues_the_entire_input(self):
        analyzer = self.analyzer()
        results = list(
            analyzer.process_messages(
                (message(i) for i in range(1, 11)), batch_size=3, max_jobs=2
            )
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(analyzer.statistics()["messages"], 10)
        self.assertEqual(analyzer.statistics()["outstanding_jobs"], 8)
        self.assertEqual(len(list(analyzer.process_pending())), 8)

    def test_closing_a_result_stream_leaves_remaining_jobs_recoverable(self):
        analyzer = self.analyzer()
        analyzer.ingest(message(1))
        analyzer.ingest(message(2))
        stream = analyzer.process_pending()
        next(stream)
        stream.close()
        self.assertEqual(analyzer.statistics()["outstanding_jobs"], 1)
        self.assertEqual(len(list(analyzer.process_pending())), 1)

    def test_invalid_settings_and_work_limits_do_not_accept_input(self):
        for options in (
            {"window_batch_size": True},
            {"max_context_messages": 257},
            {"max_context_payload_bytes": -1},
            {"context_policy": "unknown"},
        ):
            with (
                self.subTest(options=options),
                self.assertRaises(ContractError),
            ):
                AnalysisSettings(**options)
        analyzer = self.analyzer()
        with self.assertRaises(ContractError):
            list(analyzer.process_messages([message()], max_jobs=True))
        self.assertEqual(analyzer.statistics()["messages"], 0)

    def test_reading_saved_scores_imports_no_neural_runtime_in_a_new_process(
        self,
    ):
        analyzer = self.analyzer()
        list(analyzer.process_message(message()))
        source = """
import json,sys
from personal_threats import AnalysisReader
with AnalysisReader(sys.argv[1]) as reader:
    count=sum(1 for _ in reader.results('d'))
print(json.dumps({'count':count,'torch':'torch' in sys.modules,
                  'onnxruntime':'onnxruntime' in sys.modules}))
"""
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(ROOT / "src")
        result = subprocess.run(
            [sys.executable, "-c", source, str(self.path)],
            capture_output=True,
            text=True,
            check=True,
            env=environment,
            timeout=10,
        )
        self.assertEqual(
            json.loads(result.stdout),
            {"count": 1, "torch": False, "onnxruntime": False},
        )

    def test_missing_reply_and_metadata_budget_are_explicit(self):
        analyzer = self.analyzer(
            settings=AnalysisSettings(max_context_payload_bytes=500)
        )
        analyzer.ingest(message(1, source_metadata={"blob": "x" * 2000}))
        results = list(
            analyzer.process_message(message(2, reply_to={"message_id": "1"}))
        )
        result = results[-1]
        self.assertTrue(result.context["context_budget_exhausted"])
        self.assertTrue(result.context["missing_reply"])
        self.assertTrue(result.prediction.context_insufficient)

    def test_empty_media_is_unavailable_rather_than_negative(self):
        scorer = FixtureScorer()
        analyzer = self.analyzer(scorer)
        result = list(
            analyzer.process_message(
                message(
                    text="", has_unprocessed_media=True, content_kind="voice"
                )
            )
        )[0]
        self.assertIsNone(result.prediction.scores)
        self.assertIsNone(result.decision)
        self.assertEqual(scorer.calls, [])
        self.assertEqual(analyzer.statistics()["unavailable_messages"], 1)

    def test_an_incorrect_author_from_the_backend_cannot_be_saved(self):
        scorer = FixtureScorer()
        original = scorer.score
        scorer.score = lambda *a, **k: replace(
            original(*a, **k), author_id="other"
        )
        analyzer = self.analyzer(scorer)
        result = list(analyzer.process_message(message()))[0]
        self.assertEqual(result.status, "failed")
        self.assertEqual(analyzer.statistics()["current_assessments"], 0)

    def test_recipient_filter_does_not_assign_an_unknown_target_to_classifier(
        self,
    ):
        analyzer = self.analyzer()
        list(analyzer.process_message(message()))
        with self.assertRaisesRegex(ModelError, "event extractor"):
            list(analyzer.results("d", recipient_id="sender"))

    def test_two_processing_threads_never_enter_the_scorer_together(self):
        active = maximum = 0
        lock = threading.Lock()

        def hook(current):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.001)
            with lock:
                active -= 1

        analyzer = self.analyzer(FixtureScorer(hook=hook))
        for i in range(1, 31):
            dialog = i % 3
            analyzer.ingest(message(i, dialog=f"d-{dialog}"))
        with ThreadPoolExecutor(2) as executor:
            futures = [
                executor.submit(lambda: list(analyzer.process_pending()))
                for _ in range(2)
            ]
            results = [r for future in futures for r in future.result(20)]
        self.assertEqual(maximum, 1)
        self.assertEqual(len({r.assessment_id for r in results}), 30)
        self.assertEqual(analyzer.statistics()["outstanding_jobs"], 0)

    def test_concurrent_readers_observe_only_committed_valid_predictions(self):
        analyzer = self.analyzer()
        done = threading.Event()

        def write():
            try:
                for i in range(1, 81):
                    list(analyzer.process_message(message(i)))
            finally:
                done.set()

        def read():
            checked = 0
            with AnalysisReader(self.path) as reader:
                while not done.is_set():
                    for result in reader.results("d"):
                        self.assertEqual(result.status, "assessed")
                        self.assertEqual(result.prediction.author_id, "sender")
                        self.assertIsNotNone(result.assessment_id)
                        checked += 1
                self.assertEqual(reader.statistics()["assessed_messages"], 80)
            return checked

        with ThreadPoolExecutor(4) as executor:
            readers = [executor.submit(read) for _ in range(3)]
            writer = executor.submit(write)
            writer.result(30)
            self.assertGreater(sum(f.result(30) for f in readers), 0)

    def test_schema_one_database_migrates_with_its_existing_queue_intact(self):
        current = message()
        payload = canonical_json(current.to_dict())
        with closing(sqlite3.connect(self.path)) as db, db:
            db.executescript(
                SCHEMA + f"PRAGMA application_id={APPLICATION_ID};"
                "PRAGMA user_version=1;"
            )
            db.execute("INSERT INTO dialogs VALUES('d','created')")
            db.execute(
                "INSERT INTO revisions VALUES(1,'d','1',1,?,?,?,?)",
                (
                    current.metadata["timestamp"],
                    "received",
                    current.semantic_hash(),
                    payload,
                ),
            )
            db.execute(
                "INSERT INTO messages VALUES('d','1',1,1,?)",
                (current.metadata["timestamp"],),
            )
            db.execute(
                "INSERT INTO jobs VALUES('d','1',1,1,'online',"
                "'pending',NULL,0,NULL)"
            )
        with HistoryStore(self.path, read_only=True) as reader:
            self.assertEqual(reader.statistics()["messages"], 1)
        analyzer = self.analyzer()
        self.assertEqual(len(list(analyzer.process_pending())), 1)
        with closing(sqlite3.connect(self.path)) as db, db:
            self.assertEqual(
                db.execute("PRAGMA user_version").fetchone()[0], 2
            )


class CrashTests(unittest.TestCase):
    def test_kill_during_scoring_transaction_and_after_commit_are_recoverable(
        self,
    ):
        for phase in ("scoring", "transaction", "committed"):
            with (
                self.subTest(phase=phase),
                tempfile.TemporaryDirectory() as folder,
            ):
                path = Path(folder) / "history.sqlite3"
                with HistoryStore(path) as store:
                    store.ingest(message())
                source = """
import sys
from personal_threats import ThreatAnalyzer
from tests.test_service import FixtureScorer
path,phase=sys.argv[1:]
def pause(*args):
    print('ready',flush=True)
    sys.stdin.read(1)
scorer=FixtureScorer(hook=pause if phase=='scoring' else None)
with ThreatAnalyzer(path,scorer,allow_test_fixture=True) as analyzer:
    if phase=='transaction':
        def trace(sql):
            if sql.startswith('DELETE FROM jobs WHERE'):
                pause()
        analyzer.store._connection.set_trace_callback(trace)
    next(analyzer.process_pending(max_jobs=1))
    if phase=='committed':
        pause()
"""
                environment = dict(os.environ)
                environment["PYTHONPATH"] = (
                    str(ROOT / "src") + os.pathsep + str(ROOT)
                )
                process = subprocess.Popen(
                    [sys.executable, "-c", source, str(path), phase],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=environment,
                )
                try:
                    ready, _, _ = select.select([process.stdout], [], [], 10)
                    self.assertTrue(
                        ready, "Child did not reach the crash point"
                    )
                    marker = process.stdout.readline().strip()
                    if marker != "ready":
                        self.fail("Child exited: " + process.stderr.read())
                    process.kill()
                    process.wait(timeout=10)
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.wait(timeout=10)
                    for pipe in (
                        process.stdin,
                        process.stdout,
                        process.stderr,
                    ):
                        pipe.close()
                scorer = FixtureScorer()
                with ThreatAnalyzer(
                    path, scorer, allow_test_fixture=True
                ) as a:
                    result = list(a.process_pending())
                    self.assertEqual(
                        len(result), 0 if phase == "committed" else 1
                    )
                    self.assertEqual(a.statistics()["current_assessments"], 1)
                    self.assertEqual(a.statistics()["outstanding_jobs"], 0)
                    self.assertEqual(a.statistics()["failed_jobs"], 0)
                with closing(sqlite3.connect(path)) as db, db:
                    self.assertEqual(
                        db.execute("PRAGMA integrity_check").fetchone()[0],
                        "ok",
                    )


@unittest.skipUnless(
    importlib.util.find_spec("onnxruntime"), "Requires ONNX runtime"
)
class OnnxServiceTests(unittest.TestCase):
    def test_real_windowed_inference_and_cli_readers_without_a_second_session(
        self,
    ):
        from tests.model_fixtures import build_bundle

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            bundle = build_bundle(root / "model", current_pooling=True)
            path = root / "history.sqlite3"
            with ThreatAnalyzer.from_bundle(
                path, bundle, allow_test_fixture=True, device="cpu", threads=1
            ) as analyzer:
                text = ("😀hello мир\n" * 6000)[:65536]
                result = list(analyzer.process_message(message(text=text)))[0]
                self.assertGreater(result.prediction.window_count, 1000)
                self.assertEqual(
                    analyzer.store.get_message("d", "1").text, text
                )
                with patch(
                    "personal_threats.service.OnnxWindowScorer"
                ) as load:
                    with AnalysisReader.from_bundle(
                        path, bundle, allow_test_fixture=True
                    ) as reader:
                        self.assertEqual(len(list(reader.results("d"))), 1)
                    load.assert_not_called()
                out, err = io.StringIO(), io.StringIO()
                with redirect_stdout(out), redirect_stderr(err):
                    code = main(
                        [
                            "--db",
                            str(path),
                            "results",
                            "d",
                            "--model",
                            str(bundle),
                            "--allow-test-fixture",
                            "--details",
                        ]
                    )
                self.assertEqual(code, 0)
                rows = [
                    json.loads(line) for line in out.getvalue().splitlines()
                ]
                self.assertEqual(rows[0]["event"], "assessment")
                self.assertIsNone(rows[0]["decision"])
                self.assertEqual(rows[-1]["scored_messages"], 1)

    def test_cli_input_default_summary_and_durable_pending_resume(self):
        from tests.model_fixtures import build_bundle

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            bundle = build_bundle(root / "model")
            path = root / "history.sqlite3"
            source = root / "input.jsonl"
            source.write_text(
                "".join(
                    canonical_json(message(i).to_dict()) + "\n"
                    for i in range(1, 4)
                )
            )
            base = [
                "--db",
                str(path),
                "analyze",
                "--model",
                str(bundle),
                "--allow-test-fixture",
                "--device",
                "cpu",
                "--threads",
                "1",
            ]
            out, err = io.StringIO(), io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                self.assertEqual(
                    main([*base, str(source), "--max-jobs", "1"]), 0
                )
            summary = json.loads(out.getvalue())
            self.assertEqual(summary["event"], "analysis_summary")
            self.assertEqual(summary["messages"], 3)
            self.assertEqual(summary["outstanding_jobs"], 2)
            self.assertIsNone(summary["positive_messages"])
            for line in err.getvalue().splitlines():
                self.assertIn("event", json.loads(line))
            out = io.StringIO()
            with redirect_stdout(out), redirect_stderr(io.StringIO()):
                self.assertEqual(main([*base, "--details"]), 0)
            rows = [json.loads(line) for line in out.getvalue().splitlines()]
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[-1]["assessed_messages"], 3)
            self.assertEqual(rows[-1]["outstanding_jobs"], 0)


if __name__ == "__main__":
    unittest.main()
