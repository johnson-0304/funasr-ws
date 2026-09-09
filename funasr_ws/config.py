"""Runtime settings, read once from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ASR_MODEL_NAME = "sherpa-onnx-funasr-nano-int8-2025-12-30"
SAMPLE_RATE = 16000
VAD_WINDOW = 512  # Silero VAD frame size at 16 kHz


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    model_dir: Path
    asr_model_name: str
    num_threads: int
    language: str  # "" = automatic, otherwise a Nano prompt name such as "中文"
    itn: bool
    vad_threshold: float
    vad_min_silence_s: float
    vad_min_speech_s: float
    vad_max_speech_s: float
    partial_interval_ms: int
    partial_window_s: float
    pre_roll_ms: int  # audio kept before a VAD segment start so soft onsets are not clipped
    post_roll_ms: int
    max_http_audio_s: float
    host: str
    port: int
    log_level: str

    @property
    def asr_dir(self) -> Path:
        return self.model_dir / self.asr_model_name

    @property
    def vad_model(self) -> Path:
        return self.model_dir / "silero_vad.onnx"


def load_settings() -> Settings:
    from .protocol import parse_language

    return Settings(
        model_dir=Path(os.environ.get("MODEL_DIR", "models")),
        asr_model_name=os.environ.get("ASR_MODEL_NAME", ASR_MODEL_NAME),
        num_threads=_env_int("NUM_THREADS", 4),
        language=parse_language(os.environ.get("ASR_LANGUAGE", "")),
        itn=_env_bool("ASR_ITN", True),
        vad_threshold=_env_float("VAD_THRESHOLD", 0.5),
        vad_min_silence_s=_env_float("VAD_MIN_SILENCE_S", 0.4),
        vad_min_speech_s=_env_float("VAD_MIN_SPEECH_S", 0.2),
        vad_max_speech_s=_env_float("VAD_MAX_SPEECH_S", 15.0),
        partial_interval_ms=_env_int("PARTIAL_INTERVAL_MS", 800),
        partial_window_s=_env_float("PARTIAL_WINDOW_S", 12.0),
        pre_roll_ms=_env_int("PRE_ROLL_MS", 300),
        post_roll_ms=_env_int("POST_ROLL_MS", 150),
        max_http_audio_s=_env_float("MAX_HTTP_AUDIO_S", 600.0),
        host=os.environ.get("HOST", "127.0.0.1"),
        port=_env_int("PORT", 10095),
        log_level=os.environ.get("LOG_LEVEL", "info"),
    )
