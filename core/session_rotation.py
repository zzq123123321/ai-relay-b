"""Automatic OpenChamber session rotation after N successful auto tasks."""

from __future__ import annotations

import logging
import re

from core.openchamber import OpenChamberClient
from core.relay import directory_key
from core.relay_settings import RelaySettings

LOGGER = logging.getLogger("ai_relay_b")

# Fallback base title when neither a persisted rotation base name nor the
# current fixed session's title is available (requirement section 3.5).
DEFAULT_ROTATION_BASE_TITLE = "任务会话"
# Separator between the base title and the sequential suffix:
# "Task 14H" -> "Task 14H - 第一个".
ROTATION_TITLE_SEPARATOR = " - "
# Trailing " - 第<中文数字>个" suffix left over from an earlier rotation.
ROTATION_SUFFIX_PATTERN = re.compile(
    r"^(?P<base>.+?)\s*-\s*第[零一二三四五六七八九十百]+\s*个$"
)
_CN_DIGITS = "零一二三四五六七八九"


def rotation_sequence_label(index: int) -> str:
    """Chinese sequential label of a rotation: 1 -> 第一个, 10 -> 第十个,
    11 -> 第十一个, 20 -> 第二十个, 21 -> 第二十一个, 100 -> 第一百个.

    General integer conversion (no third-party dependency); at least the
    1..100 range is exact, larger values keep the same 百/十/个 grammar.
    """
    if isinstance(index, bool) or not isinstance(index, int) or index < 1:
        raise ValueError(f"rotation sequence must be a positive integer: {index!r}")
    n = index
    parts: list[str] = []
    has_hundreds = False
    if n >= 100:
        has_hundreds = True
        parts.append(_CN_DIGITS[n // 100] + "百")
        n %= 100
        if n == 0:
            return f"第{''.join(parts)}个"
        if n < 10:
            parts.append("零")
    if n >= 10:
        tens = n // 10
        # 10/11/19 read 十/十一/十九 (no leading 一 before 十) ONLY when
        # there is no hundreds place; 110 reads 一百一十.
        parts.append(_CN_DIGITS[tens] + "十" if (tens != 1 or has_hundreds) else "十")
        n %= 10
        if n:
            parts.append(_CN_DIGITS[n])
    elif n > 0:
        parts.append(_CN_DIGITS[n])
    return f"第{''.join(parts)}个"


def strip_rotation_suffix(title: str) -> str:
    """Remove a trailing " - 第<中文数字>个" rotation suffix from a title.

    "Task 14H - 第二个" -> "Task 14H"; a title without the suffix is
    returned trimmed but otherwise untouched.  This is how a base name is
    recovered from the CURRENT session's title without nesting suffixes
    ("Task 14H - 第二个 - 第一个" must never be produced).
    """
    stripped = title.strip()
    match = ROTATION_SUFFIX_PATTERN.fullmatch(stripped)
    if match is None:
        return stripped
    return match.group("base").strip()


class SessionRotation:
    """Per-project success counters driving automatic session rotation.

    Only the configured OpenChamber project directory participates; every
    directory keeps its own independent counter (count resets are owned by
    the caller: toggle on/off, manual switch / manual new session, directory
    change and successful rotation).  A failed rotation leaves the counter
    at the threshold so the next successful task retries the milestone.

    New rotated sessions are named ``<基础名称> - 第N个`` (Chinese numeral,
    e.g. "Task 14H - 第一个"): the per-directory sequence N and the base
    title are PERSISTED in relay settings (``auto_rotate_sequences`` /
    ``auto_rotate_base_titles``) keyed by the normalized project directory,
    so the sequence survives restarts, is never re-derived from the session
    list, is never reused after sessions are deleted, and manual session
    switches do not reset it.  The sequence is committed only after the new
    session was created and the mapping saved; a failed creation leaves both
    the sequence and the session mappings untouched.
    """

    def __init__(
        self,
        settings: RelaySettings,
        openchamber: OpenChamberClient | None = None,
    ):
        self.settings = settings
        self._openchamber = openchamber
        self._counts: dict[str, int] = {}

    def _client(self) -> OpenChamberClient:
        if self._openchamber is not None:
            return self._openchamber
        return OpenChamberClient(self.settings.openchamber_url)

    @staticmethod
    def _key(directory: str) -> str:
        return directory_key(directory)

    def count(self, directory: str) -> int:
        return self._counts.get(self._key(directory), 0)

    def note_auto_success(self, directory: str) -> bool:
        """Record one completed automatic OpenChamber task.

        Returns True exactly when the cumulative count reaches the
        configured threshold; the count is NOT cleared here, so a failed
        rotation retry keeps the milestone.  With auto rotation disabled
        nothing is ever counted (so no rotation -- and no sequence
        increment -- can ever be triggered).
        """
        if not self.settings.auto_rotate_enabled:
            return False
        key = self._key(directory)
        self._counts[key] = self._counts.get(key, 0) + 1
        return self._counts[key] >= self.settings.auto_rotate_threshold

    def reset(self, directory: str) -> None:
        self._counts.pop(self._key(directory), None)

    def reset_all(self) -> None:
        self._counts.clear()

    def _base_title(self, directory: str, hint: str | None = None) -> str:
        """The rotation base name for this project directory.

        Priority: the PERSISTED base title (once saved it is reused for all
        later rotations, so a suffix-carrying current session can never be
        re-derived into nested titles) > the caller's hint (the current
        fixed session's title, with any old rotation suffix stripped) >
        the generic fallback name.
        """
        key = self._key(directory)
        base = str(self.settings.auto_rotate_base_titles.get(key, "") or "").strip()
        if base:
            return base
        if hint:
            base = strip_rotation_suffix(hint)
            if base:
                return base
        return DEFAULT_ROTATION_BASE_TITLE

    def rotate(
        self,
        directory: str,
        previous_session_id: str | None = None,
        inherit_auto_accept: bool = False,
        title: str | None = None,
        base_title_hint: str | None = None,
    ) -> str:
        """Create a fresh session, persist it, and return the new id.

        The new session is named ``<base> - 第N个`` (N = this project's
        persisted sequence + 1) unless an explicit ``title`` is given, and
        replaces the directory's saved session in ``openchamber_sessions``
        plus, for the default directory, ``openchamber_session_id``; the
        new sequence value and the base title are committed to the settings
        and persisted (when the settings have a backing path) ONLY after the
        creation succeeded, so a failed creation increments nothing.
        When ``inherit_auto_accept`` the previous session's permission
        auto-accept flag is best-effort copied onto the new session: a
        snapshot/set failure is logged, never aborts the rotation.
        """
        key = self._key(directory)
        sequence = int(self.settings.auto_rotate_sequences.get(key, 0) or 0) + 1
        base = self._base_title(directory, base_title_hint)
        created_title = title or (
            f"{base}{ROTATION_TITLE_SEPARATOR}{rotation_sequence_label(sequence)}"
        )
        client = self._client()
        new_session_id = client.create_session(created_title, directory)
        if inherit_auto_accept and previous_session_id:
            self._inherit_auto_accept(
                client, previous_session_id, new_session_id, directory
            )
        self.settings.openchamber_sessions[key] = new_session_id
        default_directory = self.settings.openchamber_directory.strip()
        if default_directory and self._key(default_directory) == key:
            self.settings.openchamber_session_id = new_session_id
        # Commit the sequence and the base title only now that the session
        # exists and the mappings are saved: a creation failure (raised
        # before this point) leaves both untouched.
        self.settings.auto_rotate_sequences[key] = sequence
        self.settings.auto_rotate_base_titles[key] = base
        if self.settings._path is not None:
            self.settings.save()
        LOGGER.info(
            "自动轮换会话创建成功 directory=%s old_session_id=%s "
            "new_session_id=%s title=%s sequence=%d",
            directory,
            previous_session_id or "-",
            new_session_id,
            created_title,
            sequence,
        )
        return new_session_id

    def _inherit_auto_accept(
        self,
        client: OpenChamberClient,
        previous_session_id: str,
        new_session_id: str,
        directory: str,
    ) -> None:
        enabled = False
        try:
            snapshot = client.auto_accept_snapshot()
            enabled = bool(snapshot.get(previous_session_id, False))
        except Exception as exc:
            LOGGER.warning(
                "rotation auto-accept snapshot failed "
                "(defaulting new session off): %s",
                exc,
            )
        try:
            client.set_session_auto_accept(new_session_id, enabled, directory)
        except Exception as exc:
            LOGGER.warning(
                "rotation auto-accept inheritance failed for %s: %s",
                new_session_id, exc,
            )