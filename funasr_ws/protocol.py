"""FunASR runtime 2pass WebSocket protocol: frame parsing and partial-text deltas.

The wire format follows the FunASR `funasr-wss-server-2pass` runtime so existing
FunASR clients (including Jarvis) work unchanged:

- Client sends one JSON start frame, then binary 16 kHz s16le mono PCM frames, then
  `{"is_speaking": false}` to end the utterance.
- Server replies `{"mode": "2pass-online", ...}` with incremental partial text and
  `{"mode": "2pass-offline", ...}` with the corrected text of a finished sentence.
- The Jarvis extension `speech_noise_thres` is acknowledged with a `config` frame.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

# FunASR Nano language prompt names, keyed by BCP-47 primary subtag.
LANGUAGE_NAMES = {"zh": "中文", "en": "英文", "ja": "日文"}
AUTO_LANGUAGES = {"", "auto"}


class ProtocolError(ValueError):
    """The client sent a frame the protocol does not allow."""


@dataclass(frozen=True)
class StartFrame:
    wav_name: str
    language: str  # FunASR Nano prompt name, or "" for automatic detection
    hotwords: tuple[str, ...]
    speech_noise_threshold: float | None
    is_speaking: bool = True
    extras: dict[str, Any] = field(default_factory=dict)


def parse_language(value: Any) -> str:
    """Map a client language tag such as `en`, `zh-CN` or `auto` to a Nano prompt name."""
    if value is None:
        return ""
    tag = str(value).strip().lower()
    if tag in AUTO_LANGUAGES:
        return ""
    primary = tag.split("-")[0]
    try:
        return LANGUAGE_NAMES[primary]
    except KeyError as err:
        raise ProtocolError(f"unsupported language: {value!r}") from err


def parse_hotwords(value: Any) -> tuple[str, ...]:
    """FunASR sends hotwords as a JSON-encoded string `{"word": weight}`; weights are ignored."""
    if not value:
        return ()
    if isinstance(value, dict):
        words = value.keys()
    else:
        try:
            decoded = json.loads(value)
        except (TypeError, ValueError) as err:
            raise ProtocolError("hotwords must be a JSON object string") from err
        if not isinstance(decoded, dict):
            raise ProtocolError("hotwords must be a JSON object string")
        words = decoded.keys()
    return tuple(str(w).strip() for w in words if str(w).strip())


def parse_speech_noise_threshold(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ProtocolError("speech_noise_thres must be a number")
    if not 0 <= value < 1:
        raise ProtocolError("speech_noise_thres must satisfy 0 <= x < 1")
    return float(value)


def parse_start_frame(raw: str | bytes) -> StartFrame:
    try:
        data = json.loads(raw)
    except ValueError as err:
        raise ProtocolError("start frame is not valid JSON") from err
    if not isinstance(data, dict):
        raise ProtocolError("start frame must be a JSON object")
    mode = data.get("mode", "2pass")
    if mode not in ("2pass", "online", "offline"):
        raise ProtocolError(f"unsupported mode: {mode!r}")
    if data.get("wav_format", "pcm") != "pcm":
        raise ProtocolError("only wav_format=pcm is supported")
    if int(data.get("audio_fs", 16000)) != 16000:
        raise ProtocolError("only audio_fs=16000 is supported")
    return StartFrame(
        wav_name=str(data.get("wav_name") or "stream"),
        language=parse_language(data.get("svs_lang")),
        hotwords=parse_hotwords(data.get("hotwords")),
        speech_noise_threshold=parse_speech_noise_threshold(data.get("speech_noise_thres")),
        is_speaking=bool(data.get("is_speaking", True)),
        extras={k: v for k, v in data.items() if k not in ("wav_name", "svs_lang", "hotwords")},
    )


def parse_control_frame(raw: str | bytes) -> dict[str, Any]:
    """A JSON frame after the start frame; only `is_speaking` is meaningful."""
    try:
        data = json.loads(raw)
    except ValueError as err:
        raise ProtocolError("control frame is not valid JSON") from err
    if not isinstance(data, dict):
        raise ProtocolError("control frame must be a JSON object")
    if "speech_noise_thres" in data:
        raise ProtocolError("speech_noise_thres can only be set in the start frame")
    return data


def config_frame(threshold: float) -> dict[str, Any]:
    return {"mode": "config", "text": "", "speech_noise_thres": threshold}


def error_frame(message: str) -> dict[str, Any]:
    return {"mode": "config", "text": "", "error": message}


def online_frame(wav_name: str, text: str) -> dict[str, Any]:
    return {"mode": "2pass-online", "wav_name": wav_name, "text": text, "is_final": False}


def offline_frame(wav_name: str, text: str, is_final: bool) -> dict[str, Any]:
    return {"mode": "2pass-offline", "wav_name": wav_name, "text": text, "is_final": is_final}


class PartialTracker:
    """Turn full re-decodes of a growing segment into FunASR-style incremental deltas.

    FunASR online frames carry only newly appended text. A re-decoded LLM transcript can
    also revise earlier words; a revision that does not extend the emitted prefix is held
    back until the transcript grows past it, so the client never sees duplicated text.
    """

    def __init__(self) -> None:
        self.emitted = ""

    def delta(self, full_text: str) -> str:
        if full_text.startswith(self.emitted):
            new = full_text[len(self.emitted) :]
            self.emitted = full_text
            return new
        if len(full_text) > len(self.emitted):
            new = full_text[len(self.emitted) :]
            self.emitted = full_text
            return new
        return ""

    def reset(self) -> None:
        self.emitted = ""
