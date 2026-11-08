# SPDX-License-Identifier: Apache-2.0
"""Bounded streaming inference over durable, revision-aware dialogue work."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import logging
from pathlib import Path
import sqlite3
import threading
from typing import Any, Iterable, Iterator

from .core import (
    CONTEXT_POLICIES,
    ContractError,
    Message,
    canonical_json,
    identifier,
)
from .model import ModelError, OnnxWindowScorer
from .results import Prediction, filter_events
from .review import QualityProfile
from .store import Context, HistoryStore, StaleJob, StoreError


LOGGER = logging.getLogger("personal_threats")


@dataclass(frozen=True, slots=True)
class AnalysisSettings:
    context_policy: str = "marked_history"
    max_context_messages: int = 32
    max_context_codepoints: int = 32768
    max_context_payload_bytes: int = 8 * 1024**2
    window_batch_size: int = 8

    def __post_init__(self):
        if self.context_policy not in CONTEXT_POLICIES:
            raise ContractError("Unknown context policy")
        for name, upper in (
            ("max_context_messages", 256),
            ("max_context_codepoints", 65536),
            ("max_context_payload_bytes", 64 * 1024**2),
        ):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= upper:
                raise ContractError(f"{name} must be in [0, {upper}]")
        if (
            type(self.window_batch_size) is not int
            or not 1 <= self.window_batch_size <= 64
        ):
            raise ContractError("window_batch_size must be in [1, 64]")


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    dialog_id: str
    message_id: str
    revision_seq: int
    watermark: int
    basis: str
    status: str
    assessment_id: int | None = None
    prediction: Prediction | None = None
    decision: dict[str, Any] | None = None
    uncertainty: tuple[str, ...] = ()
    context: dict[str, Any] | None = None
    error_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "dialog-analysis-result-0.1",
            "dialog_id": self.dialog_id,
            "message_id": self.message_id,
            "revision_seq": self.revision_seq,
            "watermark": self.watermark,
            "basis": self.basis,
            "status": self.status,
            "assessment_id": self.assessment_id,
            "prediction": (
                self.prediction.to_dict() if self.prediction else None
            ),
            "decision": self.decision,
            "uncertain": bool(self.uncertainty),
            "uncertainty_reasons": list(self.uncertainty),
            "context": self.context,
            "error_type": self.error_type,
        }


def context_text(message: Message, context: Context, task: str) -> str:
    """Use the frozen training serializer; names come only from metadata."""
    from .model import quality_context_text, task_context_text

    messages = [item.message for item in context.messages]
    participants = {}
    for item in (*messages, message):
        sender = item.metadata.get("sender", {})
        name = sender.get("display_name")
        participants[item.sender_id] = {
            "participant_id": item.sender_id,
            "display_name": (
                name if isinstance(name, str) and len(name) <= 512 else None
            ),
        }
    payload = {
        "participants": list(participants.values()),
        "context_messages": [m.to_dict() for m in messages],
        "current_message": message.to_dict(),
        "context_policy": context.policy,
    }
    serializer = (
        quality_context_text
        if task == "window_features_v3"
        else task_context_text
    )
    return serializer(payload)


class _ResultAccess:
    """Read bounded current results without acquiring writer ownership."""

    def _result(self, row) -> AnalysisResult:
        prediction = Prediction.from_dict(row["payload"])
        flags = row["configuration"].get("context", {})
        reasons = []
        for name in (
            "context_insufficient",
            "prediction_uncertain",
            "context_truncated",
            "unmapped_content",
            "unknown_tokens",
            "normalization_changed",
        ):
            if getattr(prediction, name):
                reasons.append(name)
        for name, value in flags.items():
            if value is True:
                reasons.append(name)
        if prediction.analysis_status != "ok":
            reasons.append(prediction.analysis_status)
        if prediction.language == "und":
            reasons.append("language_undetermined")
        if self.profile is None:
            reasons.append("operating_profile_unavailable")
        decision = (
            self.profile.apply(
                prediction.scores, model_sha256=prediction.model_sha256
            )
            if self.profile is not None and prediction.scores is not None
            else None
        )
        if decision and decision["type_uncertain"]:
            reasons.append("type_uncertain")
        return AnalysisResult(
            prediction.dialog_id,
            prediction.message_id,
            row["revision_seq"],
            row["watermark"],
            row["basis"],
            "assessed",
            row["assessment_id"],
            prediction,
            decision,
            tuple(dict.fromkeys(reasons)),
            flags,
        )

    def results(
        self,
        dialog_id=None,
        *,
        after_id=0,
        recipient_id=None,
        include_unknown=False,
    ) -> Iterator[AnalysisResult]:
        self._check()
        if type(include_unknown) is not bool:
            raise ContractError("include_unknown must be boolean")
        if recipient_id is not None:
            identifier(recipient_id, "recipient_id")
        for row in self.store.assessments(dialog_id, after_id=after_id):
            if not self._accept_row(row):
                continue
            result = self._result(row)
            if recipient_id is not None:
                if result.prediction.extraction_status != "complete":
                    raise ModelError(
                        "Recipient filtering requires an event extractor"
                    )
                selected = filter_events(
                    result.prediction.events,
                    recipient_id,
                    include_unknown=include_unknown,
                )
                if not selected:
                    continue
                result = replace(
                    result,
                    prediction=replace(result.prediction, events=selected),
                )
            yield result

    def statistics(self, dialog_id=None):
        self._check()
        assessed = scored = positive = uncertain = unavailable = 0
        for result in self.results(dialog_id):
            assessed += 1
            scored += result.prediction.scores is not None
            positive += bool(
                result.decision and result.decision["threat_present"]
            )
            uncertain += bool(result.uncertainty)
            unavailable += result.prediction.scores is None
        storage = self.store.statistics(dialog_id)
        return {
            **storage,
            "model_available": hasattr(self, "scorer"),
            "detector_available": (
                hasattr(self, "scorer")
                and self._purpose == "trained_candidate"
            ),
            "model_purpose": self._purpose,
            "model_sha256": self._model_sha256,
            "analysis_configuration_sha256": self._signature,
            "profile": self.profile.name if self.profile else None,
            "calibration_available": self.profile is not None,
            "assessed_messages": assessed,
            "scored_messages": scored,
            "positive_messages": positive if self.profile else None,
            "uncertain_messages": uncertain,
            "unavailable_messages": unavailable,
            "unassessed_messages": (storage["messages"] - assessed),
            "production_ready_detector": False,
        }


class ThreatAnalyzer(_ResultAccess):
    """One scorer and writer; readers use independent HistoryStore objects.

    Inference runs outside the store transaction. Only a matching owned job
    may commit. The application keeps no unbounded history or prediction
    cache. A stream must be consumed to perform its remaining work.
    """

    def __init__(
        self,
        database: str | Path | HistoryStore,
        scorer: Any,
        *,
        profile: QualityProfile | None = None,
        settings: AnalysisSettings | None = None,
        allow_test_fixture: bool = False,
    ):
        if type(allow_test_fixture) is not bool:
            raise ContractError("allow_test_fixture must be boolean")
        if settings is not None and not isinstance(settings, AnalysisSettings):
            raise ContractError("Expected AnalysisSettings")
        self.settings = settings or AnalysisSettings()
        self.scorer = scorer
        self.profile = profile
        self._allow_test_fixture = allow_test_fixture
        self._worker_lock = threading.RLock()
        self._closed = False
        manifest = scorer.bundle.manifest
        self._model_sha256 = scorer.bundle.manifest_sha256
        self._model_version = manifest["model_version"]
        self._purpose = manifest["purpose"]
        self._task = manifest["task"]
        if self._purpose == "test_fixture" and not allow_test_fixture:
            raise ModelError("Test fixtures require explicit opt-in")
        if profile is not None and (
            not isinstance(profile, QualityProfile)
            or profile.model_sha256 != self._model_sha256
        ):
            raise ModelError("Operating profile belongs to another model")
        configuration = {
            "schema_version": "dialog-analysis-configuration-0.1",
            "settings": asdict(self.settings),
            "runtime": scorer.runtime_configuration(
                batch_size=self.settings.window_batch_size
            ),
        }
        self._configuration_json = canonical_json(configuration)
        self._signature = hashlib.sha256(
            self._configuration_json.encode()
        ).hexdigest()
        self._owns_store = not isinstance(database, HistoryStore)
        self.store = HistoryStore(database) if self._owns_store else database
        if self.store.read_only:
            raise StoreError("Analysis requires a writer store")

    @classmethod
    def from_bundle(
        cls,
        database,
        directory,
        *,
        report=None,
        profile="balanced",
        settings=None,
        device="auto",
        threads=0,
        allow_test_fixture=False,
    ):
        scorer = OnnxWindowScorer(
            directory,
            device=device,
            threads=threads,
            allow_test_fixture=allow_test_fixture,
        )
        operating = (
            QualityProfile.load(report, directory, name=profile)
            if report is not None
            else None
        )
        return cls(
            database,
            scorer,
            profile=operating,
            settings=settings,
            allow_test_fixture=allow_test_fixture,
        )

    @property
    def configuration(self):
        from .core import strict_json

        return strict_json(self._configuration_json)

    @property
    def configuration_sha256(self):
        return self._signature

    def _check(self):
        if self._closed:
            raise StoreError("Analyzer is closed")
        self.store._check()

    def close(self):
        with self._worker_lock:
            if self._owns_store:
                self.store.close()
            self._closed = True

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, *args):
        self.close()

    def ingest(self, message: Message):
        self._check()
        return self.store.ingest(message)

    def process_message(self, message: Message) -> Iterator[AnalysisResult]:
        """Yield the message and all affected dependent recomputations."""
        self.ingest(message)
        yield from self.process_pending(message.dialog_id)

    def process_messages(
        self,
        messages: Iterable[Message],
        *,
        batch_size=64,
        max_jobs=None,
        dialog_id=None,
    ) -> Iterator[AnalysisResult]:
        """Consume streaming input; retain only one input message at a time.

        A work limit bounds scoring, never input ingestion. Remaining input
        is still committed and queued. Re-running the same input is safe.
        """
        if type(batch_size) is not int or not 1 <= batch_size <= 1024:
            raise ContractError("batch_size must be in [1, 1024]")
        self._limit(max_jobs)
        count = attempted = 0

        def pending():
            return self.process_pending(
                dialog_id,
                max_jobs=(None if max_jobs is None else max_jobs - attempted),
            )

        try:
            for message in messages:
                if not isinstance(message, Message):
                    raise ContractError("Expected a Message")
                if dialog_id is not None and message.dialog_id != dialog_id:
                    raise ContractError("Input belongs to another dialogue")
                self.ingest(message)
                count += 1
                if count % batch_size == 0:
                    for result in pending():
                        attempted += 1
                        yield result
        except (ContractError, OSError, ValueError):
            LOGGER.error(
                "analysis_input_interrupted",
                extra={"fields": {"committed_records": count}},
            )
            raise
        for result in pending():
            yield result

    @staticmethod
    def _limit(value):
        if value is not None and (type(value) is not int or value < 0):
            raise ContractError("max_jobs must be a nonnegative integer")

    def _predict(self, message, context):
        if not message.text.strip():
            return Prediction(
                message.dialog_id,
                message.message_id,
                message.sender_id,
                self._model_version,
                self._model_sha256,
                None,
                analysis_status="input_error",
                context_insufficient=True,
                model_purpose=self._purpose,
            )
        rendered = context_text(message, context, self._task)
        prediction = self.scorer.score(
            message,
            context_text=rendered,
            batch_size=self.settings.window_batch_size,
        )
        if not isinstance(prediction, Prediction) or (
            prediction.model_sha256 != self._model_sha256
            or prediction.model_version != self._model_version
            or prediction.model_purpose != self._purpose
        ):
            raise ModelError("Backend returned another model's prediction")
        missing = context.budget_exhausted or context.missing_reply
        return replace(
            prediction,
            context_truncated=(
                prediction.context_truncated or len(rendered) >= 65536
            ),
            context_insufficient=prediction.context_insufficient or missing,
            prediction_uncertain=(
                prediction.prediction_uncertain
                or missing
                or message.metadata.get("has_unprocessed_media", False)
            ),
        )

    def _accept_row(self, row):
        return (
            row["configuration"].get("analysis_fingerprint") == self._signature
        )

    def process_pending(
        self,
        dialog_id=None,
        *,
        max_jobs=None,
    ) -> Iterator[AnalysisResult]:
        self._check()
        self._limit(max_jobs)
        if max_jobs == 0:
            return
        self.store.bind_analysis(self.configuration, dialog_id)
        attempted = 0
        while max_jobs is None or attempted < max_jobs:
            with self._worker_lock:
                self._check()
                job = self.store.claim_next(
                    dialog_id, configuration_sha256=self._signature
                )
                if job is None:
                    return
                attempted += 1
                try:
                    message = self.store.job_message(job)
                    context = self.store.context(
                        job,
                        policy=self.settings.context_policy,
                        max_messages=self.settings.max_context_messages,
                        max_codepoints=self.settings.max_context_codepoints,
                        max_payload_bytes=(
                            self.settings.max_context_payload_bytes
                        ),
                    )
                    prediction = self._predict(message, context)
                    flags = {
                        "current_message_deleted": message.metadata.get(
                            "is_deleted", False
                        ),
                        "context_budget_exhausted": context.budget_exhausted,
                        "missing_reply": context.missing_reply,
                        "reliance_on_deleted_context": (
                            context.reliance_on_deleted_context
                        ),
                        "has_unprocessed_media": message.metadata.get(
                            "has_unprocessed_media", False
                        ),
                        "context_names_omitted": any(
                            isinstance(
                                m.metadata.get("sender", {}).get(
                                    "display_name"
                                ),
                                str,
                            )
                            and len(m.metadata["sender"]["display_name"]) > 512
                            for m in (
                                message,
                                *(c.message for c in context.messages),
                            )
                        ),
                    }
                    configuration = {
                        "analysis_fingerprint": self._signature,
                        "scoring": self.configuration,
                        "context": flags,
                    }
                    result = self._result(
                        {
                            "payload": prediction.to_dict(),
                            "configuration": configuration,
                            "assessment_id": None,
                            "revision_seq": job.revision_seq,
                            "watermark": job.watermark,
                            "basis": job.basis,
                        }
                    )
                    assessment_id = self.store.complete_prediction(
                        job,
                        prediction,
                        context=context,
                        configuration=configuration,
                        allow_test_fixture=self._allow_test_fixture,
                    )
                    result = replace(result, assessment_id=assessment_id)
                except StaleJob:
                    result = AnalysisResult(
                        job.dialog_id,
                        job.message_id,
                        job.revision_seq,
                        job.watermark,
                        job.basis,
                        "superseded",
                        uncertainty=("input_or_configuration_changed",),
                    )
                except (StoreError, sqlite3.Error):
                    try:
                        self.store.release_job(job)
                    except (StoreError, sqlite3.Error):
                        pass
                    raise
                except Exception as error:
                    try:
                        self.store.fail_job(job, type(error).__name__)
                    except StaleJob:
                        status = "superseded"
                    else:
                        status = "failed"
                    result = AnalysisResult(
                        job.dialog_id,
                        job.message_id,
                        job.revision_seq,
                        job.watermark,
                        job.basis,
                        status,
                        uncertainty=("analysis_failed",),
                        error_type=type(error).__name__,
                    )
                except BaseException:
                    try:
                        self.store.release_job(job)
                    except (StoreError, sqlite3.Error):
                        pass
                    raise
            LOGGER.log(
                logging.ERROR if result.status == "failed" else logging.INFO,
                "analysis_" + result.status,
                extra={
                    "fields": {
                        "dialog_id": result.dialog_id,
                        "message_id": result.message_id,
                        "revision_seq": result.revision_seq,
                        "assessment_id": result.assessment_id,
                        "error_type": result.error_type,
                    }
                },
            )
            yield result


class AnalysisReader(_ResultAccess):
    """Read committed predictions with no writer lock or model session."""

    def __init__(self, database, *, model_sha256=None, profile=None):
        if profile is not None and not isinstance(profile, QualityProfile):
            raise ContractError("Expected a QualityProfile")
        if model_sha256 is None and profile is not None:
            model_sha256 = profile.model_sha256
        if profile is not None and profile.model_sha256 != model_sha256:
            raise ModelError("Profile and selected model differ")
        self.profile = profile
        self._model_sha256 = model_sha256
        self._purpose = None
        self._signature = None
        self._closed = False
        self._owns_store = not isinstance(database, HistoryStore)
        self.store = (
            HistoryStore(database, read_only=True)
            if self._owns_store
            else database
        )

    @classmethod
    def from_bundle(
        cls,
        database,
        directory,
        *,
        report=None,
        profile="balanced",
        allow_test_fixture=False,
    ):
        from .model import ModelBundle

        bundle = ModelBundle.load(directory)
        if bundle.manifest["purpose"] == "test_fixture" and not (
            allow_test_fixture
        ):
            raise ModelError("Test fixtures require explicit opt-in")
        operating = (
            QualityProfile.load(report, directory, name=profile)
            if report is not None
            else None
        )
        result = cls(
            database, model_sha256=bundle.manifest_sha256, profile=operating
        )
        result._purpose = bundle.manifest["purpose"]
        return result

    def _check(self):
        if self._closed:
            raise StoreError("Reader is closed")
        self.store._check()

    def close(self):
        if self._owns_store:
            self.store.close()
        self._closed = True

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, *args):
        self.close()

    def _accept_row(self, row):
        return (
            isinstance(row["configuration"].get("analysis_fingerprint"), str)
            and row["payload"].get("schema_version") == "prediction-0.1"
            and (
                self._model_sha256 is None
                or row["payload"].get("model_sha256") == self._model_sha256
            )
        )
