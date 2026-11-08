# SPDX-License-Identifier: Apache-2.0
"""JSON command-line interface for the persistent input layer."""

from __future__ import annotations

import argparse
import logging
import os
import resource
import sqlite3
import sys
import tempfile
import threading
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from . import __version__
from .core import ContractError, canonical_json, utc_now
from .importers import read_jsonl, read_telegram, write_jsonl
from .store import HistoryStore, StoreError

LOGGER = logging.getLogger("personal_threats")


class JsonFormatter(logging.Formatter):
    """Keep each log record on one line, including load and error records."""

    def format(self, record: logging.LogRecord) -> str:
        return canonical_json(
            {
                "timestamp": utc_now(),
                "level": record.levelname,
                "event": record.getMessage(),
                **getattr(record, "fields", {}),
            }
        )


def configure_logging(level: str) -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    LOGGER.handlers[:] = [handler]
    LOGGER.setLevel(level)
    LOGGER.propagate = False


def process_load() -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    try:
        resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
        resident_bytes = resident_pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        resident_bytes = None
    return {
        "rss_bytes": resident_bytes,
        "peak_rss_bytes": usage.ru_maxrss * 1024,
        "cpu_seconds": usage.ru_utime + usage.ru_stime,
    }


@contextmanager
def load_logging(interval: float) -> Iterator[None]:
    stop = threading.Event()

    def report() -> None:
        while not stop.wait(interval):
            LOGGER.info("process_load", extra={"fields": process_load()})

    worker = threading.Thread(target=report, name="load-logger", daemon=True)
    worker.start()
    try:
        yield
    finally:
        stop.set()
        worker.join()


@contextmanager
def atomic_output(path: Path) -> Iterator[Any]:
    """Publish a complete new file; failures preserve any previous output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix="." + path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _positive_number(value: str) -> float:
    import math

    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("Value must be finite and positive")
    return number


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="personal-threats",
        description="Persistent dialogue input and annotation infrastructure.",
    )
    result.add_argument("--version", action="version", version=__version__)
    result.add_argument("--db", type=Path, default=Path("var/history.sqlite3"))
    result.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    result.add_argument("--log-interval", type=_positive_number, default=30.0)
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("init", help="Create or verify the history database")
    for name in ("import-jsonl", "import-telegram", "convert-telegram"):
        command = commands.add_parser(name)
        command.add_argument("input", type=Path)
        if name != "import-jsonl":
            command.add_argument("--dialog-id")
        if name == "convert-telegram":
            command.add_argument("--output", type=Path, required=True)
    stats = commands.add_parser(
        "stats", help="Storage and pending work counts"
    )
    stats.add_argument("--dialog-id")
    history = commands.add_parser(
        "history", help="Stream all message revisions"
    )
    history.add_argument("dialog_id")
    history.add_argument("message_id")
    delete = commands.add_parser("delete", help="Record an explicit deletion")
    delete.add_argument("dialog_id")
    delete.add_argument("message_id")
    delete.add_argument("--deleted-at")
    retry = commands.add_parser(
        "retry-jobs", help="Retry failed dialogue work"
    )
    retry.add_argument("--dialog-id")
    commands.add_parser("doctor", help="Report actual runtime and readiness")
    model = commands.add_parser(
        "check-model", help="Verify a local model bundle and run an ONNX probe"
    )
    model.add_argument("directory", type=Path)
    model.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    model.add_argument("--threads", type=int, default=0)
    review = commands.add_parser(
        "review-pilot",
        help="Verify returned pilot artifacts and profile metrics",
    )
    review.add_argument("directory", type=Path)
    review.add_argument("--dataset-manifest", type=Path)
    review.add_argument("--probe", action="store_true")
    review.add_argument("--output", type=Path)
    for name in ("analyze", "results"):
        command = commands.add_parser(
            name,
            help=(
                "Score durable dialogue work"
                if name == "analyze"
                else "Read saved results without a model session"
            ),
        )
        command.add_argument("--model", type=Path, required=name == "analyze")
        command.add_argument("--profile-report", type=Path)
        command.add_argument(
            "--profile",
            choices=("strict", "balanced", "high_recall"),
            default="balanced",
        )
        command.add_argument("--details", action="store_true")
        command.add_argument("--allow-test-fixture", action="store_true")
        if name == "results":
            command.add_argument("dialog_id")
            command.add_argument("--after-id", type=int, default=0)
            command.add_argument("--recipient-id")
            command.add_argument("--include-unknown", action="store_true")
            continue
        command.add_argument("input", type=Path, nargs="?")
        command.add_argument(
            "--input-format", choices=("jsonl", "telegram"), default="jsonl"
        )
        command.add_argument("--dialog-id")
        command.add_argument("--batch-size", type=int, default=64)
        command.add_argument("--max-jobs", type=int)
        command.add_argument(
            "--device", choices=("auto", "cpu", "cuda"), default="auto"
        )
        command.add_argument("--threads", type=int, default=0)
        command.add_argument("--window-batch-size", type=int, default=8)
        command.add_argument(
            "--context-policy",
            choices=("marked_history", "visible_history"),
            default="marked_history",
        )
        command.add_argument("--max-context-messages", type=int, default=32)
        command.add_argument(
            "--max-context-codepoints", type=int, default=32768
        )
        command.add_argument(
            "--max-context-payload-bytes", type=int, default=8 * 1024**2
        )
    return result


def _emit(value: Any) -> None:
    sys.stdout.write(canonical_json(value) + "\n")


def _import(args: argparse.Namespace) -> int:
    messages = (
        read_jsonl(args.input)
        if args.command == "import-jsonl"
        else read_telegram(args.input, dialog_id=args.dialog_id)
    )
    counters = Counter()
    started = time.monotonic()
    # Committed records survive errors. Re-running identical input resumes by
    # idempotency rather than retaining an unbounded file-wide transaction.
    with HistoryStore(args.db) as store:
        try:
            for message in messages:
                result = store.ingest(message)
                counters[result.status] += 1
        except Exception:
            LOGGER.error(
                "import_interrupted",
                extra={
                    "fields": {
                        "committed_records": sum(counters.values()),
                        "outcome_counts": dict(counters),
                        "database": str(args.db),
                    }
                },
            )
            raise
    _emit(
        {
            "status": "imported",
            "records": sum(counters.values()),
            "outcome_counts": dict(counters),
            "elapsed_seconds": time.monotonic() - started,
            "detector_available": False,
        }
    )
    return 0


def run(args: argparse.Namespace) -> int:
    if args.command in ("analyze", "results"):
        return _analysis(args)
    if args.command == "review-pilot":
        from .model import file_sha256
        from .review import PilotReview

        review = PilotReview(
            args.directory,
            expected_dataset_sha256=(
                file_sha256(args.dataset_manifest)
                if args.dataset_manifest is not None
                else None
            ),
            probe=args.probe,
        )
        summary = review.summary()
        if args.output is not None:
            source = args.directory.resolve()
            if args.output.resolve().is_relative_to(source):
                raise ContractError(
                    "Review output must be outside the source run"
                )
            if (
                args.dataset_manifest is not None
                and args.output.resolve() == args.dataset_manifest.resolve()
            ):
                raise ContractError(
                    "Review output cannot replace the dataset manifest"
                )
            with atomic_output(args.output) as stream:
                stream.write(canonical_json(summary) + "\n")
        _emit(summary)
        return 0 if review.complete else 3
    if args.command == "check-model":
        from .model import OnnxWindowScorer

        scorer = OnnxWindowScorer(
            args.directory, device=args.device, threads=args.threads
        )
        _emit(scorer.probe())
        return 0
    if args.command == "doctor":
        import ijson

        _emit(
            {
                "package_version": __version__,
                "python": sys.version,
                "sqlite": sqlite3.sqlite_version,
                "ijson": ijson.__version__,
                "platform": sys.platform,
                "detector_available": False,
                "reason": "Trained, validated model artifact is not installed",
                **process_load(),
            }
        )
        return 0
    if args.command.startswith("import-"):
        return _import(args)
    if args.command == "convert-telegram":
        source = (
            args.input / "result.json" if args.input.is_dir() else args.input
        )
        if source.resolve() == args.output.resolve():
            raise ContractError("Output cannot replace the Telegram source")
        with atomic_output(args.output) as stream:
            count = write_jsonl(
                read_telegram(args.input, dialog_id=args.dialog_id), stream
            )
        _emit(
            {
                "status": "converted",
                "records": count,
                "output": str(args.output),
            }
        )
        return 0
    read_only = args.command in ("stats", "history")
    with HistoryStore(args.db, read_only=read_only) as store:
        if args.command == "init":
            _emit({"status": "ready", "database": str(store.path)})
        elif args.command == "stats":
            _emit(store.statistics(args.dialog_id))
        elif args.command == "history":
            found = False
            for row in store.history(args.dialog_id, args.message_id):
                found = True
                _emit(row)
            if not found:
                raise KeyError((args.dialog_id, args.message_id))
        elif args.command == "delete":
            result = store.mark_deleted(
                args.dialog_id, args.message_id, deleted_at=args.deleted_at
            )
            _emit(
                {
                    "status": result.status,
                    "revision": result.revision,
                    "queued_messages": result.queued_messages,
                }
            )
        elif args.command == "retry-jobs":
            _emit({"retried_jobs": store.retry_failed(args.dialog_id)})
    return 0


def _analysis(args):
    from .service import AnalysisReader, AnalysisSettings, ThreatAnalyzer

    if args.profile_report is not None and args.model is None:
        raise ContractError("A profile report requires --model")
    if args.command == "results":
        reader = (
            AnalysisReader.from_bundle(
                args.db,
                args.model,
                report=args.profile_report,
                profile=args.profile,
                allow_test_fixture=args.allow_test_fixture,
            )
            if args.model is not None
            else AnalysisReader(args.db)
        )
        with reader:
            outcomes = reader.results(
                args.dialog_id,
                after_id=args.after_id,
                recipient_id=args.recipient_id,
                include_unknown=args.include_unknown,
            )
            counts = _analysis_output(outcomes, details=args.details)
            _emit(
                {
                    "event": "analysis_summary",
                    **reader.statistics(args.dialog_id),
                    "returned_outcomes": dict(counts),
                    "recipient_id": args.recipient_id,
                }
            )
        return 0
    settings = AnalysisSettings(
        context_policy=args.context_policy,
        max_context_messages=args.max_context_messages,
        max_context_codepoints=args.max_context_codepoints,
        max_context_payload_bytes=args.max_context_payload_bytes,
        window_batch_size=args.window_batch_size,
    )
    ThreatAnalyzer._limit(args.max_jobs)
    if not 1 <= args.batch_size <= 1024:
        raise ContractError("batch_size must be in [1, 1024]")
    with ThreatAnalyzer.from_bundle(
        args.db,
        args.model,
        report=args.profile_report,
        profile=args.profile,
        settings=settings,
        device=args.device,
        threads=args.threads,
        allow_test_fixture=args.allow_test_fixture,
    ) as analyzer:
        if args.input is None:
            outcomes = analyzer.process_pending(
                args.dialog_id, max_jobs=args.max_jobs
            )
        else:
            messages = (
                read_jsonl(args.input)
                if args.input_format == "jsonl"
                else read_telegram(args.input, dialog_id=args.dialog_id)
            )
            outcomes = analyzer.process_messages(
                messages,
                batch_size=args.batch_size,
                max_jobs=args.max_jobs,
                dialog_id=args.dialog_id,
            )
        counts = _analysis_output(outcomes, details=args.details)
        summary = analyzer.statistics(args.dialog_id)
        _emit(
            {
                "event": "analysis_summary",
                **summary,
                "processed_outcomes": dict(counts),
            }
        )
        return 3 if summary["failed_jobs"] else 0


def _analysis_output(outcomes, *, details):
    counts = Counter()
    for result in outcomes:
        counts[result.status] += 1
        if details:
            _emit({"event": "assessment", **result.to_dict()})
        elif result.decision and result.decision["threat_present"]:
            _emit(
                {
                    "event": "threat",
                    "dialog_id": result.dialog_id,
                    "message_id": result.message_id,
                    "assessment_id": result.assessment_id,
                    "revision_seq": result.revision_seq,
                    "author_id": result.prediction.author_id,
                    "is_deleted": result.context.get(
                        "current_message_deleted", False
                    ),
                    "types": result.decision["types"],
                    "type_uncertain": result.decision["type_uncertain"],
                    "profile": result.decision["profile"],
                    "model_sha256": result.prediction.model_sha256,
                    "model_purpose": result.prediction.model_purpose,
                    "production_ready_detector": False,
                    "uncertain": bool(result.uncertainty),
                    "extraction_status": result.prediction.extraction_status,
                    "events": [e.to_dict() for e in result.prediction.events],
                }
            )
    return counts


def main(argv: list[str] | None = None) -> int:
    from .model import ModelError

    args = parser().parse_args(argv)
    configure_logging(args.log_level)
    try:
        with load_logging(args.log_interval):
            return run(args)
    except BrokenPipeError:
        return 0
    except KeyboardInterrupt:
        LOGGER.warning("interrupted")
        return 130
    except (
        ContractError,
        StoreError,
        ModelError,
        OSError,
        sqlite3.Error,
        KeyError,
    ) as error:
        LOGGER.error(
            "command_failed",
            extra={
                "fields": {
                    "error_type": type(error).__name__,
                    "detail": str(error),
                }
            },
        )
        return 1
