"""AI_RELAY text protocol parsing and serialization."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from uuid import uuid4

PROTOCOL_MARKER = "AI_RELAY/1"
LEGACY_BEGIN = "----- AI_RELAY_BEGIN -----"
LEGACY_END = "----- AI_RELAY_END -----"


class RelayProtocolError(ValueError):
    """Raised when an AI_RELAY message is malformed."""


class MessageType(Enum):
    TASK = "TASK"
    RESPONSE = "RESPONSE"


class ProtocolFormat(Enum):
    V1 = "v1"
    LEGACY_WEB = "legacy_web"


@dataclass(frozen=True, slots=True)
class RelayMessage:
    message_id: str
    source: str
    target: str
    message_type: MessageType
    body: str
    in_reply_to: str | None = None
    protocol_format: ProtocolFormat = ProtocolFormat.V1
    round_number: int = 0
    max_rounds: int = 3


def parse_message(text: str) -> RelayMessage:
    """Parse one protocol message while preserving the body verbatim."""
    normalized = text.replace("\r\n", "\n")

    if normalized.strip().startswith(LEGACY_BEGIN):
        return _parse_legacy_web_message(normalized)

    header, separator, body = normalized.partition("\n\n")
    lines = header.splitlines()

    if not lines or lines[0].strip() != PROTOCOL_MARKER:
        raise RelayProtocolError("missing AI_RELAY/1 marker")

    if not separator:
        raise RelayProtocolError("missing blank line before message body")

    fields: dict[str, str] = {}
    for line in lines[1:]:
        key, colon, value = line.partition(":")
        if not (colon and key.strip() and value.strip()):
            raise RelayProtocolError(f"invalid protocol header: {line}")
        normalized_key = key.strip().upper()
        if normalized_key in fields:
            raise RelayProtocolError(f"duplicate protocol header: {normalized_key}")
        fields[normalized_key] = value.strip()

    required = ("MESSAGE_ID", "SOURCE", "TARGET", "TYPE")
    missing = [name for name in required if name not in fields]
    if missing:
        raise RelayProtocolError(f"missing protocol headers: {', '.join(missing)}")

    if not body.strip():
        raise RelayProtocolError("message body must not be empty")

    try:
        message_type = MessageType(fields["TYPE"].upper())
    except ValueError as exc:
        raise RelayProtocolError(f"unsupported message type: {fields['TYPE']}") from exc

    round_number, max_rounds = _parse_rounds(fields)

    return RelayMessage(
        message_id=fields["MESSAGE_ID"],
        source=fields["SOURCE"].upper(),
        target=fields["TARGET"].upper(),
        message_type=message_type,
        body=body,
        in_reply_to=fields.get("IN_REPLY_TO"),
        protocol_format=ProtocolFormat.V1,
        round_number=round_number,
        max_rounds=max_rounds,
    )


def wrap_response(
    body: str,
    in_reply_to: str,
    protocol_format: ProtocolFormat,
    round_number: int,
    max_rounds: int,
) -> str:
    """Wrap a Reasonix reply for the ChatGPT endpoint."""
    if not body.strip():
        raise RelayProtocolError("response body must not be empty")
    if not in_reply_to.strip():
        raise RelayProtocolError("in_reply_to must not be empty")
    if round_number < 0 or max_rounds < 1 or round_number > max_rounds:
        raise RelayProtocolError("invalid ROUND/MAX_ROUNDS")

    if protocol_format is ProtocolFormat.LEGACY_WEB:
        return "\n".join(
            (
                LEGACY_BEGIN,
                "SOURCE: EXECUTOR",
                "TARGET: CHATGPT",
                "TYPE: RESPONSE",
                f"TASK_ID: {in_reply_to}",
                f"ROUND: {round_number}",
                f"MAX_ROUNDS: {max_rounds}",
                f"TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "CONTENT:",
                body,
                LEGACY_END,
            )
        )
    return "\n".join(
        (
            PROTOCOL_MARKER,
            f"MESSAGE_ID: {uuid4()}",
            f"IN_REPLY_TO: {in_reply_to}",
            "SOURCE: EXECUTOR",
            "TARGET: CHATGPT",
            "TYPE: RESPONSE",
            f"ROUND: {round_number}",
            f"MAX_ROUNDS: {max_rounds}",
            "",
            body,
        )
    )


def _parse_legacy_web_message(text: str) -> RelayMessage:
    stripped = text.strip()
    if not stripped.endswith(LEGACY_END):
        raise RelayProtocolError("missing AI_RELAY_END marker")

    payload = stripped[len(LEGACY_BEGIN):-len(LEGACY_END)].strip("\n")
    lines = payload.splitlines()
    fields: dict[str, str] = {}
    body_lines: list[str] | None = None

    for line in lines:
        if body_lines is not None:
            body_lines.append(line)
            continue
        key, colon, value = line.partition(":")
        normalized_key = key.strip().upper()
        if not (colon and normalized_key):
            raise RelayProtocolError(f"invalid legacy protocol header: {line}")
        if normalized_key == "CONTENT":
            body_lines = [value.lstrip()] if value.strip() else []
        elif normalized_key in fields:
            raise RelayProtocolError(f"duplicate protocol header: {normalized_key}")
        else:
            fields[normalized_key] = value.strip()

    if body_lines is None:
        raise RelayProtocolError("missing protocol header: CONTENT")

    required = ("TASK_ID", "SOURCE", "TARGET", "TYPE")
    missing = [name for name in required if not fields.get(name)]
    if missing:
        raise RelayProtocolError(f"missing protocol headers: {', '.join(missing)}")

    body = "\n".join(body_lines).strip()
    if not body:
        raise RelayProtocolError("message body must not be empty")

    try:
        message_type = MessageType(fields["TYPE"].upper())
    except ValueError as exc:
        raise RelayProtocolError(f"unsupported message type: {fields['TYPE']}") from exc

    round_number, max_rounds = _parse_rounds(fields)

    return RelayMessage(
        message_id=fields["TASK_ID"],
        source=fields["SOURCE"].upper(),
        target=fields["TARGET"].upper(),
        message_type=message_type,
        body=body,
        in_reply_to=fields.get("IN_REPLY_TO"),
        protocol_format=ProtocolFormat.LEGACY_WEB,
        round_number=round_number,
        max_rounds=max_rounds,
    )


def _parse_rounds(fields: dict[str, str]) -> tuple[int, int]:
    try:
        round_number = int(fields.get("ROUND", "0"))
        max_rounds = int(fields.get("MAX_ROUNDS", "3"))
    except ValueError as exc:
        raise RelayProtocolError("ROUND and MAX_ROUNDS must be integers") from exc

    if round_number < 0 or max_rounds < 1 or round_number > max_rounds:
        raise RelayProtocolError("invalid ROUND/MAX_ROUNDS")

    return round_number, max_rounds
