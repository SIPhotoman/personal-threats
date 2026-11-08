# SPDX-License-Identifier: Apache-2.0
"""Exercise persistence failures, causal history and lossless import."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, redirect_stderr, redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest


from personal_threats import (
    ContractError,
    HistoryStore,
    Message,
    StaleJob,
    StoreError,
    WriterBusy,
    text_windows,
    utf16_span,
)
from personal_threats.cli import main
from personal_threats.core import strict_json
from personal_threats.importers import (
    read_jsonl,
    read_telegram,
    telegram_message,
)
from personal_threats.store import ContextMessage


def message(key="1", text="Example", *, dialog="dialog-one", **metadata):
    """Make a known-author message with an explicit source timestamp."""
    metadata.setdefault("timestamp", f"2026-01-01T00:00:{int(key):02d}Z")
    return Message(dialog, key, "user-one", text, metadata)


def finish(store, job):
    """Persist an explicit test fixture; this is not a detector prediction."""
    return store.complete_job(
        job,
        {"test_fixture": True},
        model_version="test-fixture-v1",
        context=store.context(job),
    )


class CoreTests(unittest.TestCase):
    def test_unicode_windows_cover_every_original_character(self):
        text = ("😀е\u0301я\n" * 14000)[:65536]
        windows = list(text_windows(text, size=257, overlap=39))
        rebuilt = windows[0].text
        end = windows[0].end
        for window in windows[1:]:
            self.assertLess(window.start, end)
            start, next_end = window.start, window.end
            self.assertEqual(window.text, text[start:next_end])
            rebuilt += text[end:next_end]
            end = next_end
        self.assertEqual(rebuilt, text)
        self.assertEqual(end, 65536)
        with self.assertRaises(ContractError):
            Message("d", "m", "p", text + "x")
        with self.assertRaises(ContractError):
            list(text_windows("abc", size=2, overlap=2))

    def test_utf16_conversion_rejects_half_an_emoji(self):
        self.assertEqual(utf16_span("a😀Б", 1, 2), (1, 2))
        self.assertEqual(utf16_span("a😀Б", 3, 1), (2, 3))
        for offset, length in ((2, 1), (0, 2), (5, 0), (-1, 1)):
            with self.subTest(offset=offset, length=length):
                with self.assertRaises(ContractError):
                    utf16_span("a😀Б", offset, length)

    def test_strict_contract_rejects_ambiguous_and_invalid_json(self):
        for value in ('{"a":1,"a":2}', '{"a":NaN}', '"\\ud800"'):
            with self.assertRaises(ContractError):
                strict_json(value)
        with self.assertRaises(ContractError):
            Message.from_dict(
                {"dialog_id": "d", "message_id": "m", "text": "hello"}
            )
        with self.assertRaises(ContractError):
            Message("d", "m", "p", "", {"is_deleted": 1})
        with self.assertRaises(ContractError):
            Message("d", "m", "p", "", {"timestamp": "2026-01-01T12:00"})


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "history.sqlite3"
        self.store = HistoryStore(self.path)
        self.addCleanup(self.store.close)

    def drain(self):
        while (job := self.store.claim_next()) is not None:
            finish(self.store, job)

    def test_idempotency_metadata_and_meaningful_edits(self):
        self.store.ingest(message(source_metadata={"views": 1}))
        self.drain()
        result = self.store.ingest(message(source_metadata={"views": 2}))
        self.assertEqual(result.status, "metadata_updated")
        self.assertEqual(result.revision, 1)
        self.assertEqual(result.queued_messages, 0)
        self.assertEqual(self.store.statistics()["observations"], 2)
        self.assertEqual(
            self.store.ingest(message(source_metadata={"views": 2})).status,
            "duplicate",
        )
        changed = self.store.ingest(message(reply_to={"message_id": "3"}))
        self.assertEqual(changed.revision, 2)
        self.assertEqual(changed.queued_messages, 1)
        self.assertEqual(self.store.statistics()["current_assessments"], 0)
        self.assertEqual(self.store.statistics()["archived_assessments"], 1)

    def test_edits_during_processing_reject_stale_completion(self):
        self.store.ingest(message())
        job = self.store.claim_next()
        context = self.store.context(job)
        self.store.ingest(message(text="An updated message"))
        with self.assertRaises(StaleJob):
            self.store.complete_job(
                job, {}, model_version="fixture", context=context
            )
        self.assertEqual(self.store.statistics()["archived_assessments"], 0)
        replacement = self.store.claim_next()
        self.assertNotEqual(job.token, replacement.token)
        self.assertEqual(replacement.revision_seq, 2)
        finish(self.store, replacement)
        self.assertEqual(len(list(self.store.history("dialog-one", "1"))), 2)

    def test_future_context_cannot_be_persisted_even_if_received_earlier(self):
        future = message("3", text="Later source timestamp")
        inserted = self.store.ingest(future)
        self.store.ingest(message("1"))
        job = self.store.claim_next()
        self.assertEqual(job.message_id, "1")
        context = replace(
            self.store.context(job),
            messages=(ContextMessage(future, inserted.revision_seq),),
        )
        with self.assertRaises(ContractError):
            self.store.complete_job(
                job, {}, model_version="fixture", context=context
            )
        self.assertEqual(self.store.statistics()["archived_assessments"], 0)

    def test_late_insert_invalidates_only_its_dialogue_suffix(self):
        for key in ("1", "3"):
            self.store.ingest(message(key))
        self.store.ingest(message("4", dialog="other"))
        self.drain()
        self.assertEqual(self.store.statistics()["current_assessments"], 3)
        changed = self.store.ingest(message("2"))
        self.assertEqual(changed.queued_messages, 2)
        self.assertEqual(self.store.statistics()["current_assessments"], 2)
        first = self.store.claim_next()
        self.assertEqual(first.message_id, "2")
        finish(self.store, first)
        second = self.store.claim_next()
        self.assertEqual(second.message_id, "3")
        self.assertEqual(second.basis, "recomputed")
        self.assertEqual(
            [
                item.message.message_id
                for item in self.store.context(second).messages
            ],
            ["1", "2"],
        )
        finish(self.store, second)
        self.assertEqual(self.store.statistics()["archived_assessments"], 5)

    def test_old_snapshot_excludes_future_messages_and_edits(self):
        self.store.ingest(message("1", text="Original"))
        finish(self.store, self.store.claim_next())
        self.store.ingest(message("2"))
        job = self.store.claim_next()
        self.store.ingest(message("3", text="Future message"))
        self.store.ingest(message("1", text="Later revision"))
        context = self.store.context(job)
        self.assertEqual(len(context.messages), 1)
        self.assertEqual(context.messages[0].message.text, "Original")
        with self.assertRaises(StaleJob):
            finish(self.store, job)

    def test_explicit_deletion_survives_reimport_and_supports_both_policies(
        self,
    ):
        original = message("1", text="Previously received text")
        self.store.ingest(original)
        self.store.ingest(message("2"))
        self.drain()
        self.store.mark_deleted("dialog-one", "1")
        self.store.ingest(original)
        deleted = self.store.get_message("dialog-one", "1")
        self.assertTrue(deleted.metadata["is_deleted"])
        self.assertEqual(deleted.text, original.text)
        finish(self.store, self.store.claim_next())
        job = self.store.claim_next()
        marked = self.store.context(job)
        visible = self.store.context(job, policy="visible_history")
        self.assertTrue(marked.reliance_on_deleted_context)
        self.assertEqual(len(marked.messages), 1)
        self.assertEqual(visible.messages, ())
        with self.assertRaises(ContractError):
            self.store.complete_job(
                job, {}, model_version="fixture", context=visible
            )

    def test_context_prioritizes_reply_with_bounded_budget(self):
        for key in ("1", "2", "3"):
            self.store.ingest(message(key))
        self.drain()
        self.store.ingest(message("4", reply_to={"message_id": "1"}))
        job = self.store.claim_next()
        context = self.store.context(job, max_messages=1, max_codepoints=100)
        self.assertEqual(context.messages[0].message.message_id, "1")
        self.assertTrue(context.budget_exhausted)
        self.assertFalse(context.missing_reply)
        empty = self.store.context(job, max_messages=0)
        self.assertEqual(empty.messages, ())
        self.assertTrue(empty.missing_reply)

    def test_known_older_edit_is_archived_without_reverting_current_text(self):
        self.store.ingest(
            message(text="Latest", edited_at="2026-01-03T00:00Z")
        )
        self.drain()
        result = self.store.ingest(
            message(text="Older", edited_at="2026-01-02T00:00Z")
        )
        self.assertEqual(result.status, "stale_observation")
        self.assertEqual(result.queued_messages, 0)
        self.assertEqual(
            self.store.get_message("dialog-one", "1").text, "Latest"
        )
        self.assertEqual(self.store.statistics()["observations"], 2)

    def test_writer_lock_readers_and_reopen_recover_claim(self):
        self.store.ingest(message())
        job = self.store.claim_next()
        with self.assertRaises(WriterBusy):
            HistoryStore(self.path)
        with HistoryStore(self.path, read_only=True) as reader:
            self.assertEqual(reader.statistics()["messages"], 1)
            with self.assertRaises(StoreError):
                reader.ingest(message("2"))
        self.store.close()
        with HistoryStore(self.path) as resumed:
            recovered = resumed.claim_next()
            self.assertEqual(recovered.revision_seq, job.revision_seq)
            self.assertNotEqual(recovered.token, job.token)
            finish(resumed, recovered)

    def test_failed_dialogue_does_not_block_other_dialogues(self):
        self.store.ingest(message("1"))
        self.store.ingest(message("2"))
        failed = self.store.claim_next()
        self.store.fail_job(failed, "Injected backend failure")
        self.store.ingest(message("3", dialog="other"))
        other = self.store.claim_next()
        self.assertEqual(other.dialog_id, "other")
        finish(self.store, other)
        self.assertIsNone(self.store.claim_next())
        self.assertEqual(self.store.retry_failed("dialog-one"), 1)
        self.drain()
        self.assertEqual(self.store.statistics()["outstanding_jobs"], 0)

    def test_parallel_calls_are_serialized_without_losing_messages(self):
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(
                executor.map(
                    self.store.ingest, (message(str(i)) for i in range(1, 25))
                )
            )
        self.assertEqual(len(results), 24)
        self.assertEqual(self.store.statistics()["messages"], 24)
        self.assertEqual(self.store.statistics()["revisions"], 24)
        with closing(sqlite3.connect(self.path)) as connection, connection:
            self.assertEqual(
                connection.execute("PRAGMA integrity_check").fetchone()[0],
                "ok",
            )

    def test_other_application_database_is_not_modified(self):
        path = self.path.with_name("unrelated.sqlite3")
        with closing(sqlite3.connect(path)) as connection, connection:
            connection.execute("CREATE TABLE original (value TEXT)")
            connection.execute("INSERT INTO original VALUES ('preserved')")
        original = path.read_bytes()
        with self.assertRaises(StoreError):
            HistoryStore(path)
        self.assertEqual(path.read_bytes(), original)

    def test_process_kill_preserves_committed_input_and_recovers_work(self):
        path = self.path.with_name("crash.sqlite3")
        code = """
import sys
from personal_threats import HistoryStore, Message
store = HistoryStore(sys.argv[1])
store.ingest(Message('crash-dialog', 'one', 'sender', 'Committed text'))
store.claim_next()
print('ready', flush=True)
sys.stdin.read()
"""
        process = subprocess.Popen(
            [sys.executable, "-c", code, str(path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            self.assertEqual(process.stdout.readline().strip(), "ready")
            process.kill()
            process.wait(timeout=10)
        finally:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=10)
        with HistoryStore(path) as recovered:
            self.assertEqual(
                recovered.get_message("crash-dialog", "one").text,
                "Committed text",
            )
            finish(recovered, recovered.claim_next())
            self.assertEqual(recovered.statistics()["outstanding_jobs"], 0)


class ImportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.chat = {"id": 123, "type": "private_supergroup", "name": "Test"}

    def export(self, payload):
        path = self.root / "result.json"
        path.write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
        return path

    def test_regular_text_entities_unicode_and_unknown_metadata_survive(self):
        record = {
            "id": 1,
            "type": "message",
            "from_id": "user42",
            "from": "A",
            "date_unixtime": "1767225600",
            "date": "2026-01-01T03:00:00",
            "text": [
                "😀 ",
                {"type": "mention_name", "text": "Иван", "user_id": 24},
            ],
            "reply_to_message_id": 5,
            "unknown_future_field": {"x": 3},
        }
        result = telegram_message(record, self.chat)
        self.assertEqual(result.text, "😀 Иван")
        self.assertEqual(result.sender_id, "telegram:user42")
        self.assertEqual(result.metadata["entities"][0]["start"], 2)
        self.assertEqual(result.metadata["entities"][0]["end"], 6)
        self.assertEqual(
            result.metadata["timestamp"], "2026-01-01T00:00:00.000000+00:00"
        )
        source = result.metadata["source_metadata"]
        self.assertEqual(source["telegram"], record)
        for span in source["text_map"]:
            start, end = span["start"], span["end"]
            self.assertEqual(
                result.text[start:end],
                "😀 " if span["start"] == 0 else "Иван",
            )
        self.assertEqual(
            result.metadata["reply_to"]["dialog_id"], result.dialog_id
        )

    def test_full_export_streams_separate_chats_and_preserves_missing_author(
        self,
    ):
        first = {**self.chat, "messages": [{"id": 1, "text": "one"}]}
        second = {
            "id": 456,
            "type": "personal_chat",
            "messages": [{"id": 1, "text": "two", "from_id": "user2"}],
        }
        path = self.export(
            {"chats": {"list": [first]}, "left_chats": {"list": [second]}}
        )
        messages = list(read_telegram(path.parent))
        self.assertEqual(len(messages), 2)
        self.assertNotEqual(messages[0].dialog_id, messages[1].dialog_id)
        self.assertNotEqual(messages[0].message_id, messages[1].message_id)
        self.assertIsNone(messages[0].sender_id)
        with self.assertRaises(ContractError):
            list(read_telegram(path, dialog_id="forced"))

    def test_rich_text_and_inserted_separators_have_source_maps(self):
        record = {
            "id": 2,
            "from_id": "user7",
            "rich_message": {
                "blocks": [
                    {
                        "kind": "paragraph",
                        "text": {"type": "plain", "text": "Hello"},
                    },
                    {
                        "kind": "paragraph",
                        "text": {
                            "type": "bold",
                            "text": {"type": "plain", "text": "😀 мир"},
                        },
                    },
                ]
            },
        }
        result = telegram_message(record, self.chat)
        self.assertEqual(result.text, "Hello\n😀 мир")
        self.assertEqual(result.metadata["entities"][0]["start"], 6)
        self.assertTrue(
            any(
                span["source_pointer"] is None
                for span in result.metadata["source_metadata"]["text_map"]
            )
        )

    def test_media_and_service_messages_are_explicit_not_negative_labels(self):
        media = telegram_message(
            {"id": 1, "photo": "photo.jpg", "text": "Hi"}, self.chat
        )
        self.assertTrue(media.metadata["has_unprocessed_media"])
        service = telegram_message(
            {
                "id": 2,
                "type": "service",
                "action": "join_group",
                "actor_id": "user5",
            },
            self.chat,
        )
        self.assertEqual(
            service.metadata["service_event"]["type"], "join_group"
        )
        self.assertEqual(service.sender_id, "telegram:user5")
        local_date = telegram_message(
            {"id": 3, "date": "2026-01-01T01:00:00"}, self.chat
        )
        self.assertNotIn("timestamp", local_date.metadata)
        self.assertIn(
            "date_timezone_unavailable",
            local_date.metadata["source_metadata"]["import_warnings"],
        )

    def test_rich_media_keeps_caption_and_marks_unprocessed_content(self):
        result = telegram_message(
            {
                "id": 1,
                "rich_message": {
                    "blocks": [
                        {
                            "type": "photo",
                            "photo_id": "123",
                            "caption": {
                                "text": {"type": "plain", "text": "Caption"},
                                "credit": {"type": "plain", "text": "Credit"},
                            },
                        }
                    ]
                },
            },
            self.chat,
        )
        self.assertEqual(result.text, "Caption\nCredit")
        self.assertTrue(result.metadata["has_unprocessed_media"])
        self.assertIn(
            "unprocessed_rich_media",
            result.metadata["source_metadata"]["import_warnings"],
        )

    def test_bad_export_and_bad_jsonl_are_not_silently_skipped(self):
        path = self.root / "bad.jsonl"
        valid = message().to_dict()
        path.write_text(json.dumps(valid) + '\n{"text":"missing IDs"}\n')
        iterator = read_jsonl(path)
        self.assertEqual(next(iterator).message_id, "1")
        with self.assertRaises(ContractError):
            next(iterator)
        malformed = self.export(
            {
                **self.chat,
                "messages": [{"id": 1, "text": "a", "text_entities": []}],
            }
        )
        with self.assertRaises(ContractError):
            list(read_telegram(malformed))
        malformed.write_text(
            '{"id":123,"type":"private_supergroup",'
            '"messages":[{"id":1,"id":2}]}'
        )
        with self.assertRaises(ContractError):
            list(read_telegram(malformed))

    def test_empty_export_and_duplicate_chat_headers(self):
        self.assertEqual(
            list(read_telegram(self.export({"chats": {"list": []}}))), []
        )
        path = self.root / "duplicate.json"
        path.write_text(
            '{"id":1,"id":2,"type":"personal_chat",' '"messages":[]}'
        )
        with self.assertRaises(ContractError):
            list(read_telegram(path))

    def test_utf16_entities_use_original_codepoint_coordinates(self):
        record = {
            "id": 1,
            "text": "😀 stop",
            "entities": [{"type": "bold", "offset": 3, "length": 4}],
        }
        result = telegram_message(record, self.chat)
        entity = result.metadata["entities"][0]
        self.assertEqual((entity["start"], entity["end"]), (2, 6))
        record["entities"][0]["offset"] = 1
        with self.assertRaises(ContractError):
            telegram_message(record, self.chat)

    def test_cli_conversion_failure_preserves_previous_output(self):
        source = self.export(
            {
                **self.chat,
                "messages": [{"id": 1, "text": "a"}, {"id": 2, "text": 7}],
            }
        )
        output = self.root / "out.jsonl"
        output.write_text("previous\n")
        stderr = io.StringIO()
        with redirect_stderr(stderr), redirect_stdout(io.StringIO()):
            status = main(
                ["convert-telegram", str(source), "--output", str(output)]
            )
        self.assertEqual(status, 1)
        self.assertEqual(output.read_text(), "previous\n")
        self.assertEqual(
            json.loads(stderr.getvalue())["event"], "command_failed"
        )
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_cli_import_repeats_safely_and_reports_pending_work(self):
        path = self.export(
            {**self.chat, "messages": [{"id": 1, "text": "test"}]}
        )
        db = self.root / "runtime.sqlite3"
        base = ["--db", str(db)]
        for expected in ("inserted", "duplicate"):
            stdout = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(io.StringIO()):
                self.assertEqual(
                    main(base + ["import-telegram", str(path)]), 0
                )
            self.assertEqual(
                json.loads(stdout.getvalue())["outcome_counts"], {expected: 1}
            )
        stdout = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(io.StringIO()):
            self.assertEqual(main(base + ["stats"]), 0)
        result = json.loads(stdout.getvalue())
        self.assertEqual(result["messages"], 1)
        self.assertEqual(result["outstanding_jobs"], 1)
        self.assertFalse(result["detector_available"])


if __name__ == "__main__":
    unittest.main()
