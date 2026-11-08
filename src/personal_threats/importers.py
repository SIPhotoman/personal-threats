# SPDX-License-Identifier: Apache-2.0
"""Streaming JSONL and Telegram Desktop export conversion."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Iterator

import ijson
from ijson.common import ObjectBuilder

from .core import (
    MAX_RECORD_BYTES,
    MAX_TEXT_CODEPOINTS,
    ContractError,
    Message,
    UniqueObject,
    canonical_json,
    strict_json,
    utc_timestamp,
    utf16_span,
    validate_json,
)


def read_jsonl(path: str | Path) -> Iterator[Message]:
    """Read one bounded UTF-8 object per line, failing at the first bad row."""
    with Path(path).open("rb") as stream:
        line_number = 0
        while True:
            line = stream.readline(MAX_RECORD_BYTES + 2)
            if not line:
                break
            line_number += 1
            try:
                if len(line.rstrip(b"\r\n")) > MAX_RECORD_BYTES:
                    raise ContractError("JSONL record exceeds 4 MiB")
                if not line.strip():
                    raise ContractError("Empty JSONL line")
                yield Message.from_dict(strict_json(line.decode("utf-8")))
            except (ContractError, UnicodeError) as error:
                raise ContractError(
                    f"{path}:{line_number}: {error}"
                ) from error


def _object(events: Iterator, first: tuple) -> dict[str, Any]:
    builder = ObjectBuilder(map_type=UniqueObject)
    depth = 0
    approximate_bytes = 0
    item = first
    while True:
        _, event, value = item
        if event in ("start_map", "start_array"):
            depth += 1
            if depth > 64:
                raise ContractError("Telegram record nesting exceeds 64")
        if isinstance(value, str):
            approximate_bytes += len(value.encode("utf-8"))
            if approximate_bytes > MAX_RECORD_BYTES:
                raise ContractError("Telegram record exceeds 4 MiB")
        builder.event(event, value)
        if event in ("end_map", "end_array"):
            depth -= 1
            if depth == 0:
                validate_json(builder.value)
                return builder.value
        item = _next_event(events)


def _next_event(events: Iterator) -> tuple:
    try:
        return next(events)
    except StopIteration as error:
        raise ContractError("Unexpected end of Telegram JSON") from error


def _chat_id(chat: dict[str, Any]) -> str:
    value = chat.get("id")
    if not isinstance(value, (str, int)) or isinstance(value, bool):
        raise ContractError("Telegram chat id must precede its messages")
    kind = chat.get("type")
    if not isinstance(kind, str) or not kind:
        raise ContractError("Telegram chat type must precede its messages")
    return f"telegram:{kind}:{value}"


def _source_id(value: Any) -> str:
    if not isinstance(value, (str, int)) or isinstance(value, bool):
        raise ContractError("Invalid Telegram source identifier")
    if not str(value):
        raise ContractError("Empty Telegram source identifier")
    return str(value)


def _sender_id(value: Any) -> str | None:
    return None if value is None else "telegram:" + _source_id(value)


def _date(record: dict[str, Any], name: str) -> str | None:
    unix = record.get(name + "_unixtime")
    if unix is not None:
        try:
            if isinstance(unix, bool) or str(int(unix)) != str(unix):
                raise ValueError("Unix time must be an integer")
            return datetime.fromtimestamp(int(unix), timezone.utc).isoformat(
                timespec="microseconds"
            )
        except (ValueError, TypeError, OverflowError, OSError) as error:
            raise ContractError("Invalid Telegram Unix timestamp") from error
    value = record.get(name)
    if value:
        try:
            return utc_timestamp(value)
        except ContractError:
            # Telegram's local dates often have no timezone. Preserve those
            # in the raw record; do not invent a UTC offset.
            if isinstance(value, str):
                try:
                    moment = datetime.fromisoformat(value)
                except ValueError as error:
                    raise ContractError(
                        "Invalid Telegram local date"
                    ) from error
                if moment.tzinfo is None:
                    return None
            raise
    return None


class _Renderer:
    """Render only textual fields, with pointers back into the raw record."""

    def __init__(self) -> None:
        self.parts = []
        self.length = 0
        self.mapping = []
        self.entities = []
        self.warnings = set()

    def append(self, text: str, pointer: str | None) -> None:
        if not isinstance(text, str):
            raise ContractError("Telegram text fragment must be a string")
        if not text:
            return
        start = self.length
        self.length += len(text)
        if self.length > MAX_TEXT_CODEPOINTS:
            raise ContractError(
                "Telegram rendered text exceeds 65 536 characters"
            )
        self.parts.append(text)
        self.mapping.append(
            {
                "start": start,
                "end": self.length,
                "source_pointer": pointer,
                "source_start": 0 if pointer is not None else None,
                "source_end": len(text) if pointer is not None else None,
            }
        )

    def regular(self, value: Any, pointer: str = "/text") -> None:
        if isinstance(value, str):
            self.append(value, pointer)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                self.regular(item, f"{pointer}/{index}")
        elif isinstance(value, dict) and isinstance(value.get("text"), str):
            start = self.length
            self.append(value["text"], pointer + "/text")
            self.entity(value, start)
        else:
            raise ContractError("Unsupported Telegram text representation")

    def entity(self, value: dict[str, Any], start: int) -> None:
        kind = value.get("type", "plain")
        if kind in ("plain", "concat", "empty") or start == self.length:
            return
        entity = {"type": kind, "start": start, "end": self.length}
        for key in ("href", "user_id", "language", "document_id", "collapsed"):
            if key in value:
                entity[key] = value[key]
        if kind == "mention_name" and "user_id" in value:
            entity["participant_id"] = "telegram:user" + str(value["user_id"])
        self.entities.append(entity)

    def rich(self, value: Any, pointer: str, *, separator: str = "") -> None:
        if value is None:
            return
        if isinstance(value, str):
            self.append(value, pointer)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if index and separator:
                    self.append(separator, None)
                self.rich(item, f"{pointer}/{index}")
        elif isinstance(value, dict):
            start = self.length
            kind = value.get("type", value.get("kind"))
            if kind in (
                "photo",
                "video",
                "audio",
                "file",
                "inline_image",
                "embed",
                "embed_post",
                "map",
                "input_map",
                "collage",
                "slideshow",
            ):
                self.warnings.add("unprocessed_rich_media")
            if value.get("unsupported") or value.get("type") in (
                "unsupported",
                "unknown",
                "diff",
                "inline_image",
            ):
                self.warnings.add("partially_unsupported_rich_content")
            found = False
            for key, item in value.items():
                if key in (
                    "text",
                    "title",
                    "caption",
                    "credit",
                    "author",
                    "label",
                    "formula",
                ):
                    if found and self.length > start:
                        self.append("\n", None)
                    self.rich(item, f"{pointer}/{key}")
                    found = True
                elif key in (
                    "blocks",
                    "block",
                    "items",
                    "rows",
                    "cells",
                    "buttons",
                    "articles",
                ):
                    if found and self.length > start:
                        self.append("\n", None)
                    self.rich(item, f"{pointer}/{key}", separator="\n")
                    found = True
                elif key == "source" and value.get("type") == "math":
                    self.rich(item, f"{pointer}/{key}")
                    found = True
            if not found and value.get("type") not in (
                None,
                "empty",
                "divider",
            ):
                self.warnings.add("partially_unsupported_rich_content")
            # Rich formatting nodes use type/text; block types use kind.
            if "type" in value:
                self.entity(value, start)
        else:
            raise ContractError("Unsupported rich text value")

    @property
    def text(self) -> str:
        return "".join(self.parts)


def telegram_message(
    record: dict[str, Any],
    chat: dict[str, Any],
    *,
    dialog_id: str | None = None,
) -> Message:
    """Convert one record without inferring missing identity or media text."""
    namespace = _chat_id(chat)
    message_id = namespace + ":" + _source_id(record.get("id"))
    renderer = _Renderer()
    if "rich_message" in record:
        renderer.rich(record["rich_message"], "/rich_message")
    else:
        renderer.regular(record.get("text", ""))
    text = renderer.text
    if "text_entities" in record and "rich_message" not in record:
        entities_renderer = _Renderer()
        entities_renderer.regular(record["text_entities"], "/text_entities")
        if entities_renderer.text != text:
            raise ContractError("Telegram text and text_entities disagree")
        renderer.entities = entities_renderer.entities
    # Some interchange records include Bot API style UTF-16 spans.
    for raw in record.get("entities", []):
        if not isinstance(raw, dict):
            raise ContractError("Telegram entity must be an object")
        start, end = utf16_span(text, raw.get("offset"), raw.get("length"))
        entity = {
            **raw,
            "start": start,
            "end": end,
            "source_coordinate_system": "utf16",
        }
        renderer.entities.append(entity)
    sender = record.get("from_id", record.get("actor_id"))
    kind = record.get("type", "message")
    media = any(
        name in record
        for name in (
            "photo",
            "file",
            "media_type",
            "contact_information",
            "location_information",
            "poll",
            "game_information",
        )
    )
    if sender is None:
        renderer.warnings.add("sender_unavailable")
    metadata: dict[str, Any] = {
        "source": {
            "platform": "telegram",
            "source_chat_id": str(chat["id"]),
            "source_chat_type": chat["type"],
            "source_message_id": str(record["id"]),
            "export_format": "telegram_desktop_json",
        },
        "sender": {
            "display_name": record.get("from", record.get("actor")),
            "author_signature": record.get("author"),
        },
        "entities": renderer.entities,
        "content_kind": (
            "service"
            if kind == "service"
            else (
                "unsupported"
                if kind == "unsupported"
                else "media" if media else "text"
            )
        ),
        "has_unprocessed_media": media
        or bool(renderer.warnings - {"sender_unavailable"})
        or kind == "unsupported",
    }
    for source, target in (("date", "timestamp"), ("edited", "edited_at")):
        timestamp = _date(record, source)
        if timestamp is not None:
            metadata[target] = timestamp
        elif record.get(source):
            renderer.warnings.add(source + "_timezone_unavailable")
    if "reply_to_message_id" in record:
        peer = record.get("reply_to_peer_id")
        local = peer is None
        # Peer IDs include a type prefix whereas chat IDs can be bare.
        expected = {"telegram:" + str(peer), str(peer)}
        source_peer = {
            "personal_chat": "user",
            "private_group": "chat",
            "public_supergroup": "channel",
            "private_supergroup": "channel",
            "public_channel": "channel",
            "private_channel": "channel",
        }.get(chat["type"])
        if source_peer:
            local = local or source_peer + str(chat["id"]) in expected
        metadata["reply_to"] = {
            "message_id": (
                namespace + ":" + _source_id(record["reply_to_message_id"])
                if local
                else "telegram:external:"
                + _source_id(peer)
                + ":"
                + _source_id(record["reply_to_message_id"])
            ),
            "dialog_id": (
                (dialog_id or namespace)
                if local
                else "telegram:external:" + _source_id(peer)
            ),
        }
        if peer is not None:
            metadata["reply_to"]["source_chat_id"] = str(peer)
    for source, target in (
        ("topic_id", "topic_id"),
        ("reply_to_top_id", "root_message_id"),
        ("media_group_id", "media_group_id"),
    ):
        if record.get(source) is not None:
            metadata[target] = _source_id(record[source])
    if "forwarded_from" in record or "forwarded_from_id" in record:
        metadata["forward_origin"] = {
            "display_name": record.get("forwarded_from"),
            "sender_id": _sender_id(record.get("forwarded_from_id")),
        }
    if kind == "service":
        metadata["service_event"] = {"type": record.get("action", "unknown")}
    if kind not in ("message", "service", "unsupported"):
        raise ContractError(f"Unknown Telegram record type: {kind!r}")
    metadata["source_metadata"] = {
        "telegram": record,
        "chat": chat,
        "text_map": renderer.mapping,
        "import_warnings": sorted(renderer.warnings),
    }
    return Message(
        dialog_id or namespace, message_id, _sender_id(sender), text, metadata
    )


def _telegram_stream(
    stream: BinaryIO, *, dialog_id: str | None = None
) -> Iterator[Message]:
    events = iter(ijson.parse(stream, use_float=True))
    chat_prefixes = ("", "chats.list.item", "left_chats.list.item")
    chats: dict[str, dict[str, Any]] = {}
    object_keys = []
    seen_chat = None
    recognized = False
    for prefix, event, value in events:
        if event == "start_map":
            object_keys.append(set())
        elif event == "map_key":
            if value in object_keys[-1]:
                raise ContractError(f"Duplicate JSON key: {value!r}")
            object_keys[-1].add(value)
        elif event == "end_map":
            object_keys.pop()
        if (
            prefix in ("chats.list", "left_chats.list")
            and event == "start_array"
        ):
            recognized = True
        if prefix in chat_prefixes and event == "start_map":
            chats[prefix] = {}
        for base in chat_prefixes:
            stem = base + "." if base else ""
            if prefix in (stem + "id", stem + "type", stem + "name"):
                if event in ("string", "number", "null") and base in chats:
                    field_start = len(stem)
                    chats[base][prefix[field_start:]] = value
            if prefix != stem + "messages" or event != "start_array":
                continue
            chat = chats.get(base, {})
            namespace = _chat_id(chat)
            if dialog_id and seen_chat and seen_chat != namespace:
                raise ContractError(
                    "--dialog-id requires a single-chat export"
                )
            seen_chat = namespace
            recognized = True
            while True:
                item = _next_event(events)
                if item[0] == prefix and item[1] == "end_array":
                    break
                if item[1] != "start_map":
                    raise ContractError("Telegram messages must be objects")
                record = _object(events, item)
                yield telegram_message(record, dict(chat), dialog_id=dialog_id)
    if not recognized:
        raise ContractError("No Telegram Desktop messages array found")


def read_telegram(
    path: str | Path, *, dialog_id: str | None = None
) -> Iterator[Message]:
    """Read result.json from one chat or a complete account export folder."""
    path = Path(path)
    if path.is_dir():
        path = path / "result.json"
    try:
        with path.open("rb") as stream:
            yield from _telegram_stream(stream, dialog_id=dialog_id)
    except (ijson.JSONError, StopIteration, UnicodeError) as error:
        raise ContractError(
            f"Invalid Telegram JSON in {path}: {error}"
        ) from error


def write_jsonl(messages: Iterator[Message], stream: Any) -> int:
    count = 0
    for message in messages:
        stream.write(canonical_json(message.to_dict()) + "\n")
        count += 1
    return count
