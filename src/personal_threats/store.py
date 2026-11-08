# SPDX-License-Identifier: Apache-2.0
"""Durable single-writer SQLite history, work queue, and causal snapshots."""

from __future__ import annotations

import fcntl
import hashlib
import os
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from .core import (
    CONTEXT_POLICIES,
    ContractError,
    Message,
    canonical_json,
    identifier,
    strict_json,
    utc_now,
)
from .results import Prediction

APPLICATION_ID = 0x50544852
SCHEMA_VERSION = 2
ANALYSIS_SCHEMA = """
CREATE TABLE IF NOT EXISTS analysis_bindings (
    dialog_id TEXT PRIMARY KEY REFERENCES dialogs(dialog_id),
    signature TEXT NOT NULL,
    configuration TEXT NOT NULL
) STRICT;
CREATE INDEX IF NOT EXISTS jobs_dialog_queue
    ON jobs(dialog_id, state, watermark);
"""
SCHEMA = """
CREATE TABLE dialogs (
    dialog_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL
) STRICT;
CREATE TABLE revisions (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    dialog_id TEXT NOT NULL REFERENCES dialogs(dialog_id),
    message_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    order_key TEXT NOT NULL,
    received_at TEXT NOT NULL,
    semantic_hash TEXT NOT NULL,
    payload TEXT NOT NULL,
    UNIQUE(dialog_id, message_id, revision)
) STRICT;
CREATE INDEX revisions_history
    ON revisions(dialog_id, message_id, seq DESC);
CREATE INDEX revisions_order ON revisions(dialog_id, order_key DESC, seq DESC);
CREATE TABLE messages (
    dialog_id TEXT NOT NULL REFERENCES dialogs(dialog_id),
    message_id TEXT NOT NULL,
    first_seq INTEGER NOT NULL REFERENCES revisions(seq),
    current_seq INTEGER NOT NULL REFERENCES revisions(seq),
    order_key TEXT NOT NULL,
    PRIMARY KEY(dialog_id, message_id)
) STRICT;
CREATE INDEX messages_order ON messages(dialog_id, order_key, first_seq);
CREATE TABLE observations (
    observation_id INTEGER PRIMARY KEY,
    dialog_id TEXT NOT NULL,
    message_id TEXT NOT NULL,
    revision_seq INTEGER NOT NULL REFERENCES revisions(seq),
    received_at TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    payload TEXT NOT NULL,
    UNIQUE(dialog_id, message_id, fingerprint)
) STRICT;
CREATE TABLE jobs (
    dialog_id TEXT NOT NULL,
    message_id TEXT NOT NULL,
    revision_seq INTEGER NOT NULL REFERENCES revisions(seq),
    watermark INTEGER NOT NULL,
    basis TEXT NOT NULL CHECK(basis IN ('online', 'recomputed')),
    state TEXT NOT NULL CHECK(state IN ('pending', 'running', 'failed')),
    token TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    PRIMARY KEY(dialog_id, message_id),
    FOREIGN KEY(dialog_id, message_id)
        REFERENCES messages(dialog_id, message_id)
) STRICT;
CREATE INDEX jobs_state ON jobs(state, watermark);
CREATE TABLE assessments (
    assessment_id INTEGER PRIMARY KEY,
    dialog_id TEXT NOT NULL,
    message_id TEXT NOT NULL,
    revision_seq INTEGER NOT NULL REFERENCES revisions(seq),
    watermark INTEGER NOT NULL,
    basis TEXT NOT NULL,
    model_version TEXT NOT NULL,
    context_policy TEXT NOT NULL,
    configuration TEXT NOT NULL,
    created_at TEXT NOT NULL,
    is_current INTEGER NOT NULL CHECK(is_current IN (0, 1)),
    payload TEXT NOT NULL,
    FOREIGN KEY(dialog_id, message_id)
        REFERENCES messages(dialog_id, message_id)
) STRICT;
CREATE INDEX assessments_current
    ON assessments(dialog_id, message_id, is_current);
CREATE TABLE dependencies (
    assessment_id INTEGER NOT NULL REFERENCES assessments(assessment_id),
    revision_seq INTEGER NOT NULL REFERENCES revisions(seq),
    PRIMARY KEY(assessment_id, revision_seq)
) STRICT;
"""


class StoreError(RuntimeError):
    """An operation cannot be safely completed against this store."""


class WriterBusy(StoreError):
    """Another writer owns the database's process lock."""


class StaleJob(StoreError):
    """A newer input invalidated an in-flight result."""


@dataclass(frozen=True, slots=True)
class IngestResult:
    dialog_id: str
    message_id: str
    revision: int
    revision_seq: int
    status: str
    queued_messages: int


@dataclass(frozen=True, slots=True)
class Job:
    dialog_id: str
    message_id: str
    revision_seq: int
    watermark: int
    basis: str
    token: str


@dataclass(frozen=True, slots=True)
class ContextMessage:
    message: Message
    revision_seq: int


@dataclass(frozen=True, slots=True)
class Context:
    messages: tuple[ContextMessage, ...]
    budget_exhausted: bool
    missing_reply: bool
    reliance_on_deleted_context: bool
    dialog_id: str
    target_seq: int
    watermark: int
    policy: str


class HistoryStore:
    """One serialized writer and independently opened read-only readers.

    All commits use FULL synchronous WAL. A process lock prevents a second
    writer, and readers cannot write or checkpoint. No connection is shared
    after fork. In-memory buffers do not grow with dialogue history.
    """

    def __init__(self, path: str | Path, *, read_only: bool = False) -> None:
        self.path = Path(path).resolve()
        self.read_only = read_only
        self._pid = os.getpid()
        self._lock = threading.RLock()
        self._writer_lock = None
        self._connection = None
        self._last_dialog = None
        try:
            if read_only:
                self._connection = sqlite3.connect(
                    self.path.as_uri() + "?mode=ro",
                    uri=True,
                    isolation_level=None,
                    check_same_thread=False,
                )
            else:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._writer_lock = open(
                    str(self.path) + ".writer.lock", "a+b"
                )
                try:
                    fcntl.flock(
                        self._writer_lock.fileno(),
                        fcntl.LOCK_EX | fcntl.LOCK_NB,
                    )
                except BlockingIOError as error:
                    raise WriterBusy(
                        "A writer already owns this database"
                    ) from error
                self._connection = sqlite3.connect(
                    self.path, isolation_level=None, check_same_thread=False
                )
            connection = self._connection
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA busy_timeout=5000")
            connection.execute("PRAGMA cache_size=-16384")
            connection.execute("PRAGMA mmap_size=0")
            connection.execute("PRAGMA temp_store=FILE")
            application = connection.execute(
                "PRAGMA application_id"
            ).fetchone()[0]
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            tables = connection.execute(
                "SELECT count(*) FROM sqlite_master WHERE name NOT LIKE "
                "'sqlite_%'"
            ).fetchone()[0]
            if (
                not read_only
                and not tables
                and not application
                and not version
            ):
                connection.executescript(
                    "BEGIN IMMEDIATE;"
                    + SCHEMA
                    + ANALYSIS_SCHEMA
                    + f"PRAGMA application_id={APPLICATION_ID};"
                    + f"PRAGMA user_version={SCHEMA_VERSION};COMMIT;"
                )
            elif application != APPLICATION_ID or version not in (
                1,
                SCHEMA_VERSION,
            ):
                raise StoreError("Unrecognized database or schema version")
            elif version == 1 and not read_only:
                connection.executescript(
                    "BEGIN IMMEDIATE;"
                    + ANALYSIS_SCHEMA
                    + f"PRAGMA user_version={SCHEMA_VERSION};COMMIT;"
                )
            if read_only:
                connection.execute("PRAGMA query_only=ON")
            else:
                mode = connection.execute(
                    "PRAGMA journal_mode=WAL"
                ).fetchone()[0]
                if mode != "wal":
                    raise StoreError("SQLite could not activate WAL")
                connection.execute("PRAGMA synchronous=FULL")
                with self._transaction():
                    connection.execute(
                        "UPDATE jobs SET state='pending', token=NULL "
                        "WHERE state='running'"
                    )
        except BaseException:
            self.close()
            raise

    def _check(self, *, write: bool = False) -> None:
        if self._pid != os.getpid():
            raise StoreError("Open a new store after fork")
        if self._connection is None:
            raise StoreError("Store is closed")
        if write and self.read_only:
            raise StoreError("Store is read-only")

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._check(write=True)
            connection = self._connection
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
                connection.execute("COMMIT")
            except BaseException:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                raise

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None
            if self._writer_lock is not None:
                self._writer_lock.close()
                self._writer_lock = None

    def __enter__(self) -> HistoryStore:
        self._check()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def ingest(self, message: Message) -> IngestResult:
        """Atomically preserve input and schedule affected work.

        Edits, deletions and late inserts invalidate the causal suffix of the
        dialogue. This conservative policy also handles previously unselected
        context; recorded dependencies alone cannot detect newly relevant text.
        """
        message = Message.from_dict(message.to_dict())
        payload = canonical_json(message.to_dict())
        fingerprint_payload = message.to_dict()
        fingerprint_payload.pop("observed_at", None)
        fingerprint = hashlib.sha256(
            canonical_json(fingerprint_payload).encode()
        ).hexdigest()
        now = utc_now()
        scope = (message.dialog_id, message.message_id)
        with self._transaction() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO dialogs VALUES (?, ?)",
                (message.dialog_id, now),
            )
            previous = connection.execute(
                "SELECT m.order_key, r.revision, r.seq, r.semantic_hash, "
                "r.payload "
                "FROM messages m JOIN revisions r ON r.seq=m.current_seq "
                "WHERE m.dialog_id=? AND m.message_id=?",
                scope,
            ).fetchone()
            stale = False
            if previous:
                previous_payload = strict_json(previous["payload"])
                earlier_edit = message.metadata.get("edited_at")
                latest_edit = previous_payload.get("edited_at")
                stale = bool(
                    earlier_edit and latest_edit and earlier_edit < latest_edit
                )
                if (
                    previous_payload.get("is_deleted")
                    and "is_deleted" not in message.metadata
                ):
                    merged = message.to_dict()
                    merged["is_deleted"] = True
                    merged["deleted_at"] = previous_payload.get("deleted_at")
                    message = Message.from_dict(merged)
            semantic_hash = message.semantic_hash()
            same = previous and previous["semantic_hash"] == semantic_hash
            queued = 0
            if same or stale:
                revision = previous["revision"]
                seq = previous["seq"]
                status = "stale_observation" if stale else "duplicate"
            else:
                order_key = message.metadata.get("timestamp") or (
                    previous["order_key"]
                    if previous
                    else message.metadata.get("observed_at") or now
                )
                revision = previous["revision"] + 1 if previous else 1
                cursor = connection.execute(
                    "INSERT INTO revisions (dialog_id, message_id, revision, "
                    "order_key, received_at, semantic_hash, payload) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        *scope,
                        revision,
                        order_key,
                        now,
                        semantic_hash,
                        canonical_json(message.to_dict()),
                    ),
                )
                seq = cursor.lastrowid
                connection.execute(
                    "INSERT INTO messages VALUES (?, ?, ?, ?, ?) "
                    "ON CONFLICT(dialog_id, message_id) DO UPDATE SET "
                    "current_seq=excluded.current_seq, "
                    "order_key=excluded.order_key",
                    (*scope, seq, seq, order_key),
                )
                earliest = (
                    min(order_key, previous["order_key"])
                    if previous
                    else order_key
                )
                affected = connection.execute(
                    "SELECT count(*) FROM messages WHERE dialog_id=? "
                    "AND order_key>=?",
                    (message.dialog_id, earliest),
                ).fetchone()[0]
                basis = "recomputed" if previous else "online"
                connection.execute(
                    "INSERT INTO jobs (dialog_id, message_id, revision_seq, "
                    "watermark, basis, state) SELECT dialog_id, message_id, "
                    "current_seq, ?, CASE WHEN message_id=? THEN ? "
                    "ELSE 'recomputed' END, 'pending' FROM messages "
                    "WHERE dialog_id=? AND order_key>=? "
                    "ON CONFLICT(dialog_id, message_id) DO UPDATE SET "
                    "revision_seq=excluded.revision_seq, "
                    "watermark=excluded.watermark, basis=excluded.basis, "
                    "state='pending', token=NULL, last_error=NULL",
                    (
                        seq,
                        message.message_id,
                        basis,
                        message.dialog_id,
                        earliest,
                    ),
                )
                connection.execute(
                    "UPDATE assessments SET is_current=0 WHERE dialog_id=? "
                    "AND message_id IN (SELECT message_id FROM messages "
                    "WHERE dialog_id=? AND order_key>=?)",
                    (message.dialog_id, message.dialog_id, earliest),
                )
                queued = affected
                status = "revised" if previous else "inserted"
            cursor = connection.execute(
                "INSERT OR IGNORE INTO observations (dialog_id, message_id, "
                "revision_seq, received_at, fingerprint, payload) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (*scope, seq, now, fingerprint, payload),
            )
            if same and not stale and cursor.rowcount:
                status = "metadata_updated"
            return IngestResult(*scope, revision, seq, status, queued)

    def mark_deleted(
        self, dialog_id: str, message_id: str, *, deleted_at: str | None = None
    ) -> IngestResult:
        """Keep text; absence from an export never calls this method."""
        with self._lock:
            message = self.get_message(dialog_id, message_id)
            if message.metadata.get("is_deleted"):
                return self.ingest(message)
            payload = message.to_dict()
            payload.update(is_deleted=True, deleted_at=deleted_at or utc_now())
            return self.ingest(Message.from_dict(payload))

    def get_message(
        self, dialog_id: str, message_id: str, *, revision: int | None = None
    ) -> Message:
        with self._lock:
            self._check()
            if revision is None:
                row = self._connection.execute(
                    "SELECT r.payload FROM messages m JOIN revisions r "
                    "ON r.seq=m.current_seq WHERE m.dialog_id=? "
                    "AND m.message_id=?",
                    (dialog_id, message_id),
                ).fetchone()
            else:
                row = self._connection.execute(
                    "SELECT payload FROM revisions WHERE dialog_id=? "
                    "AND message_id=? AND revision=?",
                    (dialog_id, message_id, revision),
                ).fetchone()
            if row is None:
                raise KeyError((dialog_id, message_id, revision))
            return Message.from_dict(strict_json(row["payload"]))

    def history(
        self, dialog_id: str, message_id: str
    ) -> Iterator[dict[str, Any]]:
        after = 0
        while True:
            with self._lock:
                self._check()
                rows = self._connection.execute(
                    "SELECT seq, revision, received_at, payload "
                    "FROM revisions "
                    "WHERE dialog_id=? AND message_id=? AND seq>? "
                    "ORDER BY seq LIMIT 16",
                    (dialog_id, message_id, after),
                ).fetchall()
            if not rows:
                break
            for row in rows:
                after = row["seq"]
                yield {
                    "revision_seq": after,
                    "revision": row["revision"],
                    "received_at": row["received_at"],
                    "message": strict_json(row["payload"]),
                }

    def statistics(self, dialog_id: str | None = None) -> dict[str, Any]:
        with self._lock:
            self._check()
            scope = (dialog_id, dialog_id)
            row = self._connection.execute(
                "SELECT "
                "(SELECT count(*) FROM dialogs "
                "WHERE ? IS NULL OR dialog_id=?) "
                "AS dialogs, "
                "(SELECT count(*) FROM messages "
                "WHERE ? IS NULL OR dialog_id=?) "
                "AS messages, "
                "(SELECT count(*) FROM revisions "
                "WHERE ? IS NULL OR dialog_id=?) "
                "AS revisions, "
                "(SELECT count(*) FROM observations WHERE ? IS NULL "
                "OR dialog_id=?) AS observations, "
                "(SELECT count(*) FROM jobs WHERE ? IS NULL OR dialog_id=?) "
                "AS outstanding_jobs, "
                "(SELECT count(*) FROM jobs WHERE state='failed' AND "
                "(? IS NULL OR dialog_id=?)) AS failed_jobs, "
                "(SELECT count(*) FROM assessments WHERE ? IS NULL "
                "OR dialog_id=?) AS archived_assessments, "
                "(SELECT count(*) FROM assessments WHERE is_current=1 AND "
                "(? IS NULL OR dialog_id=?)) AS current_assessments",
                scope * 8,
            ).fetchone()
            return {
                **dict(row),
                "dialog_id": dialog_id,
                "detector_available": False,
            }

    def bind_analysis(
        self, configuration: dict[str, Any], dialog_id: str | None = None
    ) -> int:
        """Persist a scoring configuration and invalidate changed dialogues.

        Initial pending work keeps its original knowledge watermarks. A model
        or context-policy change archives old current results and requeues
        the whole dialogue atomically, including any in-flight ownership.
        Threshold changes alone need no new scoring configuration.
        """
        if not isinstance(configuration, dict) or not configuration:
            raise ContractError("Expected a nonempty analysis configuration")
        if dialog_id is not None:
            identifier(dialog_id, "dialog_id")
        encoded = canonical_json(configuration)
        signature = hashlib.sha256(encoded.encode()).hexdigest()
        queued = 0
        last = ""
        with self._transaction() as connection:
            while True:
                row = connection.execute(
                    "SELECT d.dialog_id, b.signature FROM dialogs d "
                    "LEFT JOIN analysis_bindings b USING(dialog_id) "
                    "WHERE d.dialog_id>? AND (? IS NULL OR d.dialog_id=?) "
                    "AND (b.signature IS NULL OR b.signature<>?) "
                    "ORDER BY d.dialog_id LIMIT 1",
                    (last, dialog_id, dialog_id, signature),
                ).fetchone()
                if row is None:
                    break
                last = row["dialog_id"]
                prior = connection.execute(
                    "SELECT 1 FROM assessments WHERE dialog_id=? "
                    "AND is_current=1 LIMIT 1",
                    (last,),
                ).fetchone()
                changed = row["signature"] is not None or prior is not None
                if changed:
                    connection.execute(
                        "UPDATE assessments SET is_current=0 "
                        "WHERE dialog_id=?",
                        (last,),
                    )
                    connection.execute(
                        "DELETE FROM jobs WHERE dialog_id=?", (last,)
                    )
                cursor = connection.execute(
                    "INSERT OR IGNORE INTO jobs (dialog_id, message_id, "
                    "revision_seq, watermark, basis, state) "
                    "SELECT m.dialog_id, m.message_id, m.current_seq, "
                    "(SELECT max(seq) FROM revisions), ?, 'pending' "
                    "FROM messages m WHERE m.dialog_id=?",
                    ("recomputed" if changed else "online", last),
                )
                queued += cursor.rowcount
                connection.execute(
                    "INSERT INTO analysis_bindings VALUES (?, ?, ?) "
                    "ON CONFLICT(dialog_id) DO UPDATE SET "
                    "signature=excluded.signature, "
                    "configuration=excluded.configuration",
                    (last, signature, encoded),
                )
        return queued

    def claim_next(
        self,
        dialog_id: str | None = None,
        *,
        configuration_sha256: str | None = None,
    ) -> Job | None:
        """Claim work durably, preserving causal order within each dialogue."""
        with self._transaction() as connection:
            binding = (
                "AND EXISTS (SELECT 1 FROM analysis_bindings b "
                "WHERE b.dialog_id=d.dialog_id AND b.signature=?) "
                if configuration_sha256 is not None
                else ""
            )
            parameters = (dialog_id, dialog_id)
            if configuration_sha256 is not None:
                parameters += (configuration_sha256,)
            query = (
                "SELECT d.dialog_id FROM dialogs d WHERE "
                "(? IS NULL OR d.dialog_id=?) "
                + binding
                + "AND EXISTS (SELECT 1 FROM jobs j WHERE "
                "j.dialog_id=d.dialog_id AND j.state='pending') "
                "AND NOT EXISTS (SELECT 1 FROM jobs busy "
                "WHERE busy.dialog_id=d.dialog_id "
                "AND busy.state IN ('running','failed')) "
            )
            selected = connection.execute(
                query + "AND d.dialog_id>? ORDER BY d.dialog_id LIMIT 1",
                (*parameters, self._last_dialog or ""),
            ).fetchone()
            if selected is None:
                selected = connection.execute(
                    query + "ORDER BY d.dialog_id LIMIT 1", parameters
                ).fetchone()
            if selected is None:
                return None
            row = connection.execute(
                "SELECT j.* FROM jobs j JOIN messages m USING "
                "(dialog_id,message_id) WHERE j.dialog_id=? "
                "AND j.state='pending' ORDER BY j.watermark, "
                "m.order_key, m.first_seq LIMIT 1",
                (selected[0],),
            ).fetchone()
            if row is None:
                return None
            token = uuid.uuid4().hex
            connection.execute(
                "UPDATE jobs SET state='running', token=?, "
                "attempts=attempts+1 WHERE dialog_id=? AND message_id=?",
                (token, row["dialog_id"], row["message_id"]),
            )
            self._last_dialog = row["dialog_id"]
            return Job(
                row["dialog_id"],
                row["message_id"],
                row["revision_seq"],
                row["watermark"],
                row["basis"],
                token,
            )

    def _job_row(
        self, connection: sqlite3.Connection, job: Job
    ) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM jobs WHERE dialog_id=? AND message_id=? "
            "AND token=? AND state='running' AND revision_seq=? "
            "AND watermark=?",
            (
                job.dialog_id,
                job.message_id,
                job.token,
                job.revision_seq,
                job.watermark,
            ),
        ).fetchone()
        if row is None:
            raise StaleJob("Input changed or job is no longer owned")
        return row

    def job_message(self, job: Job) -> Message:
        """Read the exact owned revision, without substituting a newer edit."""
        with self._lock:
            self._check()
            self._job_row(self._connection, job)
            row = self._connection.execute(
                "SELECT payload FROM revisions WHERE seq=?",
                (job.revision_seq,),
            ).fetchone()
            return Message.from_dict(strict_json(row["payload"]))

    def complete_prediction(
        self,
        job: Job,
        prediction: Prediction,
        *,
        context: Context,
        configuration: dict[str, Any] | None = None,
        allow_test_fixture: bool = False,
        participant_exists: Callable[[str], bool] | None = None,
    ) -> int:
        """Validate typed model evidence before the existing atomic commit.

        An uncalibrated score record remains unthresholded in storage. Numeric
        test fixtures require explicit opt-in and cannot silently become real
        detector results.
        """
        if type(allow_test_fixture) is not bool:
            raise ContractError("allow_test_fixture must be boolean")
        if not isinstance(prediction, Prediction):
            raise ContractError("Expected a typed Prediction")
        if (
            prediction.model_purpose == "test_fixture"
            and not allow_test_fixture
        ):
            raise ContractError(
                "Test fixture output is not a trained prediction"
            )
        with self._lock:
            message = self.job_message(job)
            prediction.validate(
                message,
                (item.message for item in context.messages),
                participant_exists=participant_exists,
            )
            return self.complete_job(
                job,
                prediction.to_dict(),
                model_version=prediction.model_version,
                context=context,
                policy=context.policy,
                configuration=configuration,
            )

    def fail_job(self, job: Job, error: str) -> None:
        with self._transaction() as connection:
            self._job_row(connection, job)
            connection.execute(
                "UPDATE jobs SET state='failed', token=NULL, last_error=? "
                "WHERE dialog_id=? AND message_id=?",
                (str(error)[:4096], job.dialog_id, job.message_id),
            )

    def release_job(self, job: Job) -> None:
        """Return an interrupted owned job to pending without losing it."""
        with self._transaction() as connection:
            self._job_row(connection, job)
            connection.execute(
                "UPDATE jobs SET state='pending', token=NULL "
                "WHERE dialog_id=? AND message_id=?",
                (job.dialog_id, job.message_id),
            )

    def assessments(
        self,
        dialog_id: str | None = None,
        *,
        current: bool = True,
        after_id: int = 0,
        page_size: int = 128,
    ) -> Iterator[dict[str, Any]]:
        """Read bounded pages; never hold a read transaction across yields.

        Each page reflects committed data at its read time. A fixed upper ID
        excludes later inserts, so a growing writer cannot extend this stream
        indefinitely. The immutable assessment ID supports consumer replay.
        """
        if type(current) is not bool:
            raise ContractError("current must be boolean")
        if type(after_id) is not int or after_id < 0:
            raise ContractError("after_id must be a nonnegative integer")
        if type(page_size) is not int or not 1 <= page_size <= 1024:
            raise ContractError("page_size must be in [1, 1024]")
        if dialog_id is not None:
            identifier(dialog_id, "dialog_id")
        with self._lock:
            self._check()
            upper = self._connection.execute(
                "SELECT coalesce(max(assessment_id),0) FROM assessments"
            ).fetchone()[0]
        while after_id < upper:
            with self._lock:
                self._check()
                rows = self._connection.execute(
                    "SELECT * FROM assessments WHERE assessment_id>? "
                    "AND assessment_id<=? AND (? IS NULL OR dialog_id=?) "
                    "AND (?=0 OR is_current=1) "
                    "ORDER BY assessment_id LIMIT ?",
                    (
                        after_id,
                        upper,
                        dialog_id,
                        dialog_id,
                        int(current),
                        page_size,
                    ),
                ).fetchall()
            if not rows:
                break
            for row in rows:
                after_id = row["assessment_id"]
                yield {
                    **dict(row),
                    "configuration": strict_json(row["configuration"]),
                    "payload": strict_json(row["payload"]),
                }

    def retry_failed(self, dialog_id: str | None = None) -> int:
        with self._transaction() as connection:
            cursor = connection.execute(
                "UPDATE jobs SET state='pending', token=NULL, last_error=NULL "
                "WHERE state='failed' AND (? IS NULL OR dialog_id=?)",
                (dialog_id, dialog_id),
            )
            return cursor.rowcount

    def context(
        self,
        job: Job,
        *,
        policy: str = "marked_history",
        max_messages: int = 32,
        max_codepoints: int = 32_768,
        max_payload_bytes: int = 8 * 1024**2,
    ) -> Context:
        """Bounded explicit-reply and recent context at the job's watermark.

        Future messages and revisions received later than the recorded
        watermark are excluded. Oversize context is omitted with a flag;
        the current message remains intact for separate window processing.
        Semantic retrieval will be added with the trained model.
        """
        if policy not in CONTEXT_POLICIES:
            raise ContractError("Unknown context policy")
        if (
            type(max_messages) is not int
            or max_messages < 0
            or type(max_codepoints) is not int
            or max_codepoints < 0
            or type(max_payload_bytes) is not int
            or max_payload_bytes < 0
        ):
            raise ContractError("Context budgets must be nonnegative integers")
        with self._lock:
            self._check()
            target = self._connection.execute(
                "SELECT r.*, m.first_seq FROM revisions r JOIN messages m "
                "USING (dialog_id, message_id) WHERE r.seq=? "
                "AND r.dialog_id=? AND r.message_id=?",
                (job.revision_seq, job.dialog_id, job.message_id),
            ).fetchone()
            if target is None or target["seq"] > job.watermark:
                raise StaleJob("Invalid context target")
            current = Message.from_dict(strict_json(target["payload"]))
            reply = current.metadata.get("reply_to", {})
            reply_id = reply.get("message_id")
            local_reply = (
                reply.get("dialog_id", job.dialog_id) == job.dialog_id
            )
            query = (
                "SELECT r.*, m.first_seq FROM revisions r "
                "JOIN messages m USING "
                "(dialog_id, message_id) WHERE r.dialog_id=? AND "
                "r.seq=(SELECT max(v.seq) FROM revisions v "
                "WHERE v.dialog_id=r.dialog_id AND v.message_id=r.message_id "
                "AND v.seq<=?) AND (r.order_key<? OR "
                "(r.order_key=? AND m.first_seq<?)) "
            )
            parameters = (
                job.dialog_id,
                job.watermark,
                target["order_key"],
                target["order_key"],
                target["first_seq"],
            )
            selected = []
            ordering = {}
            used = 0
            used_payload = 0
            exhausted = False
            found_reply = not reply_id

            def consider(row: sqlite3.Row) -> None:
                nonlocal used, used_payload, exhausted, found_reply
                if row["seq"] in ordering:
                    return
                size = len(row["payload"].encode("utf-8"))
                if used_payload + size > max_payload_bytes:
                    exhausted = True
                    return
                message = Message.from_dict(strict_json(row["payload"]))
                deleted = message.metadata.get("is_deleted", False)
                if deleted and policy == "visible_history":
                    return
                if (
                    len(selected) >= max_messages
                    or used + len(message.text) > max_codepoints
                ):
                    exhausted = True
                    return
                selected.append(ContextMessage(message, row["seq"]))
                ordering[row["seq"]] = (row["order_key"], row["first_seq"])
                used += len(message.text)
                used_payload += size
                if local_reply and message.message_id == reply_id:
                    found_reply = True

            if reply_id and local_reply:
                replied = self._connection.execute(
                    query + "AND r.message_id=?", (*parameters, reply_id)
                ).fetchone()
                if replied:
                    consider(replied)
            rows = self._connection.execute(
                query + "ORDER BY r.order_key DESC, m.first_seq DESC",
                parameters,
            )
            # The SQL cursor streams rows; at most the configured context is
            # materialized. Deleted entries can be skipped without a RAM scan.
            try:
                for row in rows:
                    if row["seq"] in ordering:
                        continue
                    if len(selected) >= max_messages:
                        exhausted = True
                        break
                    consider(row)
            finally:
                rows.close()
            # Deterministic causal ordering, independent of reply priority.
            selected.sort(key=lambda item: ordering[item.revision_seq])
            return Context(
                tuple(selected),
                exhausted,
                not found_reply,
                any(
                    item.message.metadata.get("is_deleted", False)
                    for item in selected
                ),
                job.dialog_id,
                job.revision_seq,
                job.watermark,
                policy,
            )

    def complete_job(
        self,
        job: Job,
        payload: dict[str, Any],
        *,
        model_version: str,
        context: Context,
        policy: str = "marked_history",
        configuration: dict[str, Any] | None = None,
    ) -> int:
        """Atomically archive a backend result with exact input dependencies.

        This persistence method does not produce predictions, calibrate them,
        or certify a backend's output schema. A detector adapter must validate
        its own result before calling it.
        """
        identifier(model_version, "model_version")
        if policy not in CONTEXT_POLICIES:
            raise ContractError("Unknown context policy")
        if not isinstance(payload, dict):
            raise ContractError("Assessment payload must be an object")
        if (
            context.dialog_id != job.dialog_id
            or context.target_seq != job.revision_seq
            or context.watermark != job.watermark
            or context.policy != policy
        ):
            raise ContractError(
                "Context does not belong to this job and policy"
            )
        serialized = canonical_json(payload)
        if configuration is not None and not isinstance(configuration, dict):
            raise ContractError("Assessment configuration must be an object")
        config = canonical_json(configuration or {})
        dependencies = {item.revision_seq for item in context.messages}
        if len(dependencies) != len(context.messages):
            raise ContractError("Duplicate assessment dependencies")
        with self._transaction() as connection:
            self._job_row(connection, job)
            target = connection.execute(
                "SELECT r.order_key, m.first_seq FROM revisions r "
                "JOIN messages m USING (dialog_id, message_id) WHERE r.seq=?",
                (job.revision_seq,),
            ).fetchone()
            target_order = (target["order_key"], target["first_seq"])
            for item in context.messages:
                seq = item.revision_seq
                dependency = connection.execute(
                    "SELECT r.*, m.first_seq, (SELECT max(v.seq) "
                    "FROM revisions v WHERE v.dialog_id=r.dialog_id "
                    "AND v.message_id=r.message_id AND v.seq<=?) AS latest "
                    "FROM revisions r JOIN messages m USING "
                    "(dialog_id, message_id) WHERE r.seq=?",
                    (job.watermark, seq),
                ).fetchone()
                if (
                    dependency is None
                    or dependency["dialog_id"] != job.dialog_id
                    or seq > job.watermark
                    or seq == job.revision_seq
                    or dependency["latest"] != seq
                ):
                    raise ContractError("Invalid assessment dependency")
                if (
                    dependency["order_key"],
                    dependency["first_seq"],
                ) >= target_order:
                    raise ContractError("Future context is not permitted")
                if (
                    canonical_json(item.message.to_dict())
                    != dependency["payload"]
                ):
                    raise ContractError("Context differs from stored revision")
                if policy == "visible_history" and item.message.metadata.get(
                    "is_deleted"
                ):
                    raise ContractError(
                        "Deleted context is not visible history"
                    )
            connection.execute(
                "UPDATE assessments SET is_current=0 WHERE dialog_id=? "
                "AND message_id=?",
                (job.dialog_id, job.message_id),
            )
            cursor = connection.execute(
                "INSERT INTO assessments (dialog_id, message_id, "
                "revision_seq, "
                "watermark, basis, model_version, context_policy, "
                "configuration, created_at, is_current, payload) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)",
                (
                    job.dialog_id,
                    job.message_id,
                    job.revision_seq,
                    job.watermark,
                    job.basis,
                    model_version,
                    policy,
                    config,
                    utc_now(),
                    serialized,
                ),
            )
            assessment_id = cursor.lastrowid
            connection.executemany(
                "INSERT INTO dependencies VALUES (?, ?)",
                ((assessment_id, seq) for seq in sorted(dependencies)),
            )
            connection.execute(
                "DELETE FROM jobs WHERE dialog_id=? AND message_id=?",
                (job.dialog_id, job.message_id),
            )
            return assessment_id
