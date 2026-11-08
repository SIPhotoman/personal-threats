# SPDX-License-Identifier: Apache-2.0
"""Validated messages and lossless Unicode utilities for streaming input."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator

MAX_TEXT_CODEPOINTS = 65_536
MAX_RECORD_BYTES = 4 * 1024 * 1024
THREAT_TYPES = (
    "physical_harm",
    "coercion",
    "stalking",
    "doxxing",
    "reputational_harm",
    "property_harm",
)
CONTEXT_POLICIES = ("marked_history", "visible_history")
OPTIONAL_FIELDS = frozenset(
    {
        "timestamp",
        "observed_at",
        "edited_at",
        "deleted_at",
        "revision_id",
        "is_deleted",
        "sender",
        "reply_to",
        "topic_id",
        "root_message_id",
        "forward_origin",
        "entities",
        "source",
        "content_kind",
        "caption",
        "media_group_id",
        "has_unprocessed_media",
        "service_event",
        "source_metadata",
    }
)
REQUIRED_FIELDS = frozenset({"dialog_id", "message_id", "sender_id", "text"})


class ContractError(ValueError):
    """Input cannot be represented without loss or ambiguity."""


class UniqueObject(dict):
    """Reject ambiguous duplicate keys in both JSON parsers."""

    def __setitem__(self, key: str, value: Any) -> None:
        if key in self:
            raise ContractError(f"Duplicate JSON key: {key!r}")
        super().__setitem__(key, value)


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = UniqueObject()
    for key, value in pairs:
        result[key] = value
    return dict(result)


def _invalid_constant(value: str) -> None:
    raise ContractError(f"Non-finite JSON number: {value}")


def validate_json(value: Any, depth: int = 0) -> None:
    """Accept finite JSON values and Unicode scalar strings only."""
    if depth > 64:
        raise ContractError("JSON nesting exceeds 64 levels")
    if isinstance(value, str):
        if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
            raise ContractError("Unpaired Unicode surrogate")
    elif value is None or isinstance(value, (bool, int)):
        return
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ContractError("Non-finite JSON number")
    elif isinstance(value, (list, tuple)):
        for item in value:
            validate_json(item, depth + 1)
    elif isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ContractError("JSON object keys must be strings")
            validate_json(key, depth + 1)
            validate_json(item, depth + 1)
    else:
        raise ContractError(f"Not a JSON value: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    validate_json(value)
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def strict_json(text: str | bytes) -> Any:
    try:
        result = json.loads(
            text,
            object_pairs_hook=_object_pairs,
            parse_constant=_invalid_constant,
        )
        validate_json(result)
        return result
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ContractError(f"Invalid JSON: {error}") from error


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def utc_timestamp(value: str) -> str:
    """Canonical UTC; a timezone must be supplied explicitly."""
    if not isinstance(value, str):
        raise ContractError("Timestamp must be an ISO 8601 string")
    try:
        moment = datetime.fromisoformat(value)
        if moment.tzinfo is None or moment.utcoffset() is None:
            raise ValueError("timezone is required")
        return moment.astimezone(timezone.utc).isoformat(
            timespec="microseconds"
        )
    except (ValueError, OverflowError) as error:
        raise ContractError(f"Invalid timestamp: {value!r}") from error


def identifier(value: Any, name: str, *, nullable: bool = False) -> None:
    if value is None and nullable:
        return
    if not isinstance(value, str) or not value or len(value) > 512:
        raise ContractError(f"{name} must be a string of 1..512 characters")
    if any(ord(character) < 32 for character in value):
        raise ContractError(f"Control character in {name}")


@dataclass(frozen=True, slots=True)
class Message:
    """One original message; optional JSON fields are stored in metadata.

    ``sender_id=None`` explicitly represents unavailable source metadata.
    Text is never normalized, rewritten, or silently truncated.
    """

    dialog_id: str
    message_id: str
    sender_id: str | None
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        identifier(self.dialog_id, "dialog_id")
        identifier(self.message_id, "message_id")
        identifier(self.sender_id, "sender_id", nullable=True)
        if not isinstance(self.text, str):
            raise ContractError("text must be a string")
        if len(self.text) > MAX_TEXT_CODEPOINTS:
            raise ContractError("text exceeds 65 536 Unicode code points")
        if not isinstance(self.metadata, dict):
            raise ContractError("metadata must be a JSON object")
        unknown = set(self.metadata) - OPTIONAL_FIELDS
        if unknown:
            raise ContractError(f"Unknown message fields: {sorted(unknown)}")
        metadata = strict_json(canonical_json(self.metadata))
        for name in ("timestamp", "observed_at", "edited_at", "deleted_at"):
            if metadata.get(name) is not None:
                metadata[name] = utc_timestamp(metadata[name])
        for name in ("is_deleted", "has_unprocessed_media"):
            if name in metadata and type(metadata[name]) is not bool:
                raise ContractError(f"{name} must be boolean")
        for name in ("topic_id", "root_message_id", "media_group_id"):
            if metadata.get(name) is not None:
                identifier(metadata[name], name)
        for name in (
            "source",
            "sender",
            "reply_to",
            "forward_origin",
            "source_metadata",
        ):
            if name in metadata and not isinstance(metadata[name], dict):
                raise ContractError(f"{name} must be an object")
        reply = metadata.get("reply_to", {})
        if "message_id" in reply:
            identifier(reply["message_id"], "reply_to.message_id")
        if "dialog_id" in reply:
            identifier(reply["dialog_id"], "reply_to.dialog_id")
        if "entities" in metadata:
            if not isinstance(metadata["entities"], list):
                raise ContractError("entities must be an array")
            for entity in metadata["entities"]:
                if not isinstance(entity, dict):
                    raise ContractError("Entity must be an object")
                start, end = entity.get("start"), entity.get("end")
                if (
                    type(start) is not int
                    or type(end) is not int
                    or not 0 <= start <= end <= len(self.text)
                ):
                    raise ContractError("Invalid code point entity span")
                identifier(entity.get("type"), "entity.type")
        if metadata.get("caption") is not None:
            if not isinstance(metadata["caption"], str):
                raise ContractError("caption must be a string")
            if len(metadata["caption"]) > MAX_TEXT_CODEPOINTS:
                raise ContractError("caption exceeds the text limit")
        object.__setattr__(self, "metadata", metadata)
        if len(canonical_json(self.to_dict()).encode()) > MAX_RECORD_BYTES:
            raise ContractError("Message JSON exceeds 4 MiB")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Message:
        if not isinstance(payload, dict):
            raise ContractError("Message must be an object")
        if REQUIRED_FIELDS - payload.keys():
            raise ContractError("Missing required message fields")
        return cls(
            **{name: payload[name] for name in REQUIRED_FIELDS},
            metadata={
                name: value
                for name, value in payload.items()
                if name not in REQUIRED_FIELDS
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.metadata,
            "dialog_id": self.dialog_id,
            "message_id": self.message_id,
            "sender_id": self.sender_id,
            "text": self.text,
        }

    def semantic_hash(self) -> str:
        """Ignore transport time, raw metadata, and upstream version IDs."""
        payload = self.to_dict()
        for name in ("observed_at", "source_metadata", "revision_id"):
            payload.pop(name, None)
        payload.setdefault("is_deleted", False)
        payload.setdefault("has_unprocessed_media", False)
        return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class TextWindow:
    start: int
    end: int
    text: str


def text_windows(
    text: str, *, size: int = 2048, overlap: int = 256
) -> Iterator[TextWindow]:
    """Cover all original code points with overlapping character windows."""
    if type(size) is not int or type(overlap) is not int:
        raise ContractError("Window size and overlap must be integers")
    if not 0 <= overlap < size:
        raise ContractError("Require 0 <= overlap < size")
    if not isinstance(text, str) or len(text) > MAX_TEXT_CODEPOINTS:
        raise ContractError("Invalid text or text exceeds the message limit")
    validate_json(text)
    start = 0
    while True:
        end = min(start + size, len(text))
        yield TextWindow(start, end, text[start:end])
        if end == len(text):
            break
        start = end - overlap


def utf16_span(text: str, offset: int, length: int) -> tuple[int, int]:
    """Convert UTF-16 units, rejecting boundaries inside surrogate pairs."""
    if not isinstance(text, str) or len(text) > MAX_TEXT_CODEPOINTS:
        raise ContractError("Invalid text for UTF-16 conversion")
    validate_json(text)
    if type(offset) is not int or type(length) is not int:
        raise ContractError("UTF-16 offsets must be integers")
    if offset < 0 or length < 0:
        raise ContractError("Negative UTF-16 offset")
    positions = {0: 0}
    units = 0
    for index, character in enumerate(text, 1):
        units += 2 if ord(character) > 0xFFFF else 1
        positions[units] = index
    try:
        return positions[offset], positions[offset + length]
    except KeyError as error:
        raise ContractError(
            "UTF-16 span splits a character or exceeds text"
        ) from error
