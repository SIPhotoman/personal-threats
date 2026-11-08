# SPDX-License-Identifier: Apache-2.0
"""Validated scores, event evidence and strict recipient filtering."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import re
from typing import Any, Callable, Iterable

from .core import (
    THREAT_TYPES,
    ContractError,
    Message,
    canonical_json,
    identifier,
)

TAXONOMY_VERSION = "personal-threats-0.6"


def _record(payload: Any, fields: Iterable[str]) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != set(fields):
        raise ContractError("Result object has missing or unknown fields")
    return dict(payload)


def probability(value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ContractError("Score must be a finite number in [0, 1]")
    return float(value)


def _boolean(value: Any, name: str) -> None:
    if type(value) is not bool:
        raise ContractError(f"{name} must be boolean")


@dataclass(frozen=True, slots=True)
class Scores:
    """Seven numeric scores; values alone do not establish calibration."""

    binary: float
    types: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "binary", probability(self.binary))
        values = tuple(probability(value) for value in self.types)
        if len(values) != len(THREAT_TYPES):
            raise ContractError("Exactly six type scores are required")
        object.__setattr__(self, "types", values)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Scores:
        if not isinstance(payload, dict) or set(payload) != {
            "binary",
            "types",
        }:
            raise ContractError("Invalid score object")
        by_type = payload["types"]
        if not isinstance(by_type, dict) or set(by_type) != set(THREAT_TYPES):
            raise ContractError("Type score names differ from the taxonomy")
        return cls(
            payload["binary"], tuple(by_type[name] for name in THREAT_TYPES)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "binary": self.binary,
            "types": dict(zip(THREAT_TYPES, self.types)),
        }


@dataclass(frozen=True, slots=True)
class PersonReference:
    kind: str
    participant_id: str | None = None
    mention: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("participant", "third_person", "unknown"):
            raise ContractError("Unknown person reference kind")
        if self.kind == "participant":
            identifier(self.participant_id, "participant_id")
        elif self.participant_id is not None:
            raise ContractError("Only a participant can have participant_id")
        if self.kind == "unknown" and self.mention is not None:
            raise ContractError("Unknown reference must have mention=null")
        if self.mention is not None and (
            not isinstance(self.mention, str) or not self.mention
        ):
            raise ContractError("Mention must be a nonempty string or null")
        if self.kind == "third_person" and self.mention is None:
            raise ContractError("A third person requires an explicit mention")
        canonical_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "participant_id": self.participant_id,
            "mention": self.mention,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PersonReference:
        return cls(**_record(payload, cls.__dataclass_fields__))


@dataclass(frozen=True, slots=True)
class Fragment:
    message_id: str
    start: int
    end: int
    text: str

    def __post_init__(self) -> None:
        identifier(self.message_id, "fragment.message_id")
        if (
            type(self.start) is not int
            or type(self.end) is not int
            or not 0 <= self.start < self.end
        ):
            raise ContractError("Fragment requires a nonempty code point span")
        if (
            not isinstance(self.text, str)
            or len(self.text) != self.end - self.start
        ):
            raise ContractError("Fragment text length differs from its span")
        canonical_json(self.text)

    def validate(self, message: Message) -> None:
        start, end = self.start, self.end
        if (
            self.message_id != message.message_id
            or self.end > len(message.text)
            or message.text[start:end] != self.text
        ):
            raise ContractError("Fragment differs from the original message")

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "start": self.start,
            "end": self.end,
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Fragment:
        return cls(**_record(payload, cls.__dataclass_fields__))


@dataclass(frozen=True, slots=True)
class ThreatEvent:
    event_id: str
    author_id: str | None
    source_mode: str
    asserted_actor: PersonReference
    types: tuple[str, ...]
    type_uncertain: bool
    explicitness: str
    temporal_status: str
    targets: tuple[PersonReference, ...]
    fragments: tuple[Fragment, ...]
    supporting_message_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        identifier(self.event_id, "event_id")
        identifier(self.author_id, "author_id", nullable=True)
        if self.source_mode not in ("direct", "reported"):
            raise ContractError("Invalid source_mode")
        if self.explicitness not in ("explicit", "implicit"):
            raise ContractError("Invalid explicitness")
        if self.temporal_status not in ("announced", "ongoing", "completed"):
            raise ContractError("Invalid temporal_status")
        _boolean(self.type_uncertain, "type_uncertain")
        object.__setattr__(self, "types", tuple(self.types))
        object.__setattr__(self, "targets", tuple(self.targets))
        object.__setattr__(self, "fragments", tuple(self.fragments))
        object.__setattr__(
            self, "supporting_message_ids", tuple(self.supporting_message_ids)
        )
        if any(not isinstance(value, str) for value in self.types):
            raise ContractError("Event types must be strings")
        if len(set(self.types)) != len(self.types) or set(self.types) - set(
            THREAT_TYPES
        ):
            raise ContractError("Unknown or repeated event type")
        if not self.types and not self.type_uncertain:
            raise ContractError("Missing event types require type_uncertain")
        if not isinstance(self.asserted_actor, PersonReference):
            raise ContractError("asserted_actor must be a PersonReference")
        if not self.targets or any(
            not isinstance(item, PersonReference) for item in self.targets
        ):
            raise ContractError("Event requires at least one target")
        if not self.fragments or any(
            not isinstance(item, Fragment) for item in self.fragments
        ):
            raise ContractError("Event requires original-text fragments")
        if len(set(self.fragments)) != len(self.fragments):
            raise ContractError("Repeated fragment")
        if len(set(self.targets)) != len(self.targets):
            raise ContractError("Repeated target")
        for key in self.supporting_message_ids:
            identifier(key, "supporting_message_id")
        if len(set(self.supporting_message_ids)) != len(
            self.supporting_message_ids
        ):
            raise ContractError("Repeated supporting message ID")

    def validate(
        self,
        message: Message,
        context: Iterable[Message] = (),
        *,
        participant_exists: Callable[[str], bool] | None = None,
    ) -> None:
        if self.author_id != message.sender_id:
            raise ContractError("Event author must equal the current sender")
        messages = (message, *context)
        if any(item.dialog_id != message.dialog_id for item in messages):
            raise ContractError(
                "Cross-dialogue event evidence is not permitted"
            )
        available = {item.message_id for item in messages}
        if not set(self.supporting_message_ids) <= available:
            raise ContractError("Unavailable supporting message")
        known = {
            item.sender_id for item in messages if item.sender_id is not None
        }
        for item in messages:
            for entity in item.metadata.get("entities", ()):
                if entity.get("participant_id"):
                    known.add(entity["participant_id"])
        for reference in (self.asserted_actor, *self.targets):
            if reference.kind == "participant":
                if reference.participant_id not in known and (
                    participant_exists is None
                    or not participant_exists(reference.participant_id)
                ):
                    raise ContractError(
                        "Participant ID is absent from metadata"
                    )
            if reference.mention is not None and not any(
                reference.mention in item.text for item in messages
            ):
                raise ContractError("Person mention is absent from input text")
        for fragment in self.fragments:
            fragment.validate(message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "author_id": self.author_id,
            "source_mode": self.source_mode,
            "asserted_actor": self.asserted_actor.to_dict(),
            "types": list(self.types),
            "type_uncertain": self.type_uncertain,
            "explicitness": self.explicitness,
            "temporal_status": self.temporal_status,
            "targets": [item.to_dict() for item in self.targets],
            "fragments": [item.to_dict() for item in self.fragments],
            "supporting_message_ids": list(self.supporting_message_ids),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ThreatEvent:
        fields = _record(payload, cls.__dataclass_fields__)
        for name in (
            "types",
            "targets",
            "fragments",
            "supporting_message_ids",
        ):
            if not isinstance(fields[name], list):
                raise ContractError(f"{name} must be a JSON array")
        fields["asserted_actor"] = PersonReference.from_dict(
            fields["asserted_actor"]
        )
        fields["targets"] = tuple(
            PersonReference.from_dict(value) for value in fields["targets"]
        )
        fields["fragments"] = tuple(
            Fragment.from_dict(value) for value in fields["fragments"]
        )
        return cls(**fields)


@dataclass(frozen=True, slots=True)
class Prediction:
    """Uncalibrated model output with explicit extraction capabilities."""

    dialog_id: str
    message_id: str
    author_id: str | None
    model_version: str
    model_sha256: str
    scores: Scores | None
    events: tuple[ThreatEvent, ...] = ()
    extraction_status: str = "not_available"
    analysis_status: str = "ok"
    language: str = "und"
    detected_languages: tuple[str, ...] = ()
    context_insufficient: bool = False
    prediction_uncertain: bool = True
    window_count: int = 0
    context_truncated: bool = False
    unmapped_content: bool = False
    unknown_tokens: bool = False
    normalization_changed: bool = False
    model_purpose: str = "trained_candidate"

    def __post_init__(self) -> None:
        for name in ("dialog_id", "message_id", "model_version"):
            identifier(getattr(self, name), name)
        identifier(self.author_id, "author_id", nullable=True)
        if not isinstance(self.model_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", self.model_sha256
        ):
            raise ContractError("Invalid model SHA-256")
        object.__setattr__(self, "events", tuple(self.events))
        object.__setattr__(
            self, "detected_languages", tuple(self.detected_languages)
        )
        if self.analysis_status not in (
            "ok",
            "partial_language_support",
            "unsupported_language",
            "input_error",
        ):
            raise ContractError("Invalid analysis_status")
        if self.extraction_status not in ("complete", "not_available"):
            raise ContractError("Invalid extraction_status")
        if self.extraction_status == "not_available" and self.events:
            raise ContractError("Unavailable extraction cannot contain events")
        if self.model_purpose not in ("trained_candidate", "test_fixture"):
            raise ContractError("Invalid model purpose")
        failed = self.analysis_status in (
            "unsupported_language",
            "input_error",
        )
        if (
            failed
            and (self.scores is not None or self.events)
            or not failed
            and not isinstance(self.scores, Scores)
        ):
            raise ContractError("Scores contradict analysis status")
        if self.language not in ("ru", "en", "mixed", "other", "und"):
            raise ContractError("Invalid language")
        if any(
            not isinstance(code, str) or not re.fullmatch("[a-z]{2}", code)
            for code in self.detected_languages
        ):
            raise ContractError("Invalid detected language code")
        if len(set(self.detected_languages)) != len(self.detected_languages):
            raise ContractError("Repeated language code")
        languages = set(self.detected_languages)
        if self.language == "mixed" and not {"ru", "en"} <= languages:
            raise ContractError("Mixed language requires ru and en")
        if (
            self.analysis_status == "unsupported_language"
            and self.language != "other"
        ):
            raise ContractError("Unsupported language must be marked other")
        if self.analysis_status == "partial_language_support" and (
            not languages & {"ru", "en"} or not languages - {"ru", "en"}
        ):
            raise ContractError(
                "Partial coverage requires supported and other languages"
            )
        if type(self.window_count) is not int or self.window_count < 0:
            raise ContractError("window_count must be a nonnegative integer")
        for name in (
            "context_insufficient",
            "prediction_uncertain",
            "context_truncated",
            "unmapped_content",
            "unknown_tokens",
            "normalization_changed",
        ):
            _boolean(getattr(self, name), name)
        if any(not isinstance(event, ThreatEvent) for event in self.events):
            raise ContractError("Invalid event object")
        if len({event.event_id for event in self.events}) != len(self.events):
            raise ContractError("Repeated event ID")

    def validate(
        self,
        message: Message,
        context: Iterable[Message] = (),
        *,
        participant_exists: Callable[[str], bool] | None = None,
    ) -> None:
        if (
            self.dialog_id != message.dialog_id
            or self.message_id != message.message_id
            or self.author_id != message.sender_id
        ):
            raise ContractError("Prediction belongs to another message")
        context = tuple(context)
        for event in self.events:
            event.validate(
                message, context, participant_exists=participant_exists
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "prediction-0.1",
            "taxonomy_version": TAXONOMY_VERSION,
            "dialog_id": self.dialog_id,
            "message_id": self.message_id,
            "author_id": self.author_id,
            "model_version": self.model_version,
            "model_sha256": self.model_sha256,
            "model_purpose": self.model_purpose,
            "analysis_status": self.analysis_status,
            "language": self.language,
            "detected_languages": list(self.detected_languages),
            "partial_language_support": self.analysis_status
            == "partial_language_support",
            "scores": (
                self.scores.to_dict() if self.scores is not None else None
            ),
            "calibration_status": "uncalibrated",
            "threat_present": None,
            "extraction_status": self.extraction_status,
            "events": [event.to_dict() for event in self.events],
            "context_insufficient": self.context_insufficient,
            "prediction_uncertain": self.prediction_uncertain,
            "window_count": self.window_count,
            "context_truncated": self.context_truncated,
            "unmapped_content": self.unmapped_content,
            "unknown_tokens": self.unknown_tokens,
            "normalization_changed": self.normalization_changed,
            "score_aggregation": "max_logit",
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Prediction:
        extras = {
            "schema_version",
            "taxonomy_version",
            "partial_language_support",
            "calibration_status",
            "threat_present",
            "score_aggregation",
        }
        fields = _record(payload, set(cls.__dataclass_fields__) | extras)
        if (
            fields.pop("schema_version") != "prediction-0.1"
            or fields.pop("taxonomy_version") != TAXONOMY_VERSION
            or fields.pop("calibration_status") != "uncalibrated"
            or fields.pop("threat_present") is not None
            or fields.pop("score_aggregation") != "max_logit"
        ):
            raise ContractError("Unsupported prediction contract")
        partial = fields.pop("partial_language_support")
        _boolean(partial, "partial_language_support")
        if partial != (
            fields["analysis_status"] == "partial_language_support"
        ):
            raise ContractError("Partial language flags disagree")
        if not isinstance(fields["events"], list) or not isinstance(
            fields["detected_languages"], list
        ):
            raise ContractError("Events and languages must be JSON arrays")
        fields["events"] = tuple(
            ThreatEvent.from_dict(value) for value in fields["events"]
        )
        if fields["scores"] is not None:
            fields["scores"] = Scores.from_dict(fields["scores"])
        return cls(**fields)


def filter_events(
    events: Iterable[ThreatEvent],
    participant_id: str | None = None,
    *,
    include_unknown: bool = False,
) -> tuple[ThreatEvent, ...]:
    """Match any known target; unknown means kind=unknown, not third_person."""
    _boolean(include_unknown, "include_unknown")
    if participant_id is None:
        return tuple(events)
    identifier(participant_id, "participant_id")
    return tuple(
        event
        for event in events
        if any(
            (
                target.kind == "participant"
                and target.participant_id == participant_id
            )
            or (include_unknown and target.kind == "unknown")
            for target in event.targets
        )
    )


def merge_identical_events(
    events: Iterable[ThreatEvent],
) -> tuple[ThreatEvent, ...]:
    """Deduplicate exact evidence only; never fuse similar or nearby events."""
    unique = []
    seen = set()
    for event in events:
        payload = event.to_dict()
        payload.pop("event_id")
        payload["types"] = sorted(payload["types"])
        for key in ("targets", "fragments"):
            payload[key] = sorted(payload[key], key=canonical_json)
        payload["supporting_message_ids"] = sorted(
            payload["supporting_message_ids"]
        )
        key = canonical_json(payload)
        if key not in seen:
            seen.add(key)
            unique.append(
                replace(event, event_id=f"event-{len(unique)+1:06d}")
            )
    return tuple(unique)
