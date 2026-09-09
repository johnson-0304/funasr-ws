"""sherpa-onnx wrappers: one shared Fun-ASR-Nano recognizer and per-connection VADs."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .config import SAMPLE_RATE, VAD_WINDOW, Settings

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Segment:
    start_s: float
    end_s: float
    text: str


class Vad(Protocol):
    """The subset of sherpa_onnx.VoiceActivityDetector used by the streaming session."""

    def accept_waveform(self, samples: np.ndarray) -> None: ...
    def is_speech_detected(self) -> bool: ...
    def empty(self) -> bool: ...
    def pop(self) -> None: ...
    def flush(self) -> None: ...
    @property
    def front(self): ...  # SpeechSegment with .start (samples) and .samples
    @property
    def current_segment(self): ...


class Engine:
    """Loads the models once; `transcribe` is serialized because one recognizer is shared."""

    def __init__(self, settings: Settings) -> None:

        self.settings = settings
        asr = settings.asr_dir
        for name in ("encoder_adaptor.int8.onnx", "llm.int8.onnx", "embedding.int8.onnx"):
            if not (asr / name).is_file():
                raise FileNotFoundError(f"ASR model file missing: {asr / name}")
        if not settings.vad_model.is_file():
            raise FileNotFoundError(f"VAD model file missing: {settings.vad_model}")
        self._lock = threading.Lock()
        self._recognizers: dict[str, object] = {}
        self._denoiser = self._make_denoiser() if settings.denoise else None
        self._recognizer(settings.language)
        probe = self._recognizer(settings.language).create_stream()
        self.supports_stream_hotwords = probe.has_option("hotwords")
        log.info(
            "engine ready (threads=%d, default language=%r, denoise=%s, hotwords=%s)",
            settings.num_threads,
            settings.language or "auto",
            settings.denoise,
            self.supports_stream_hotwords,
        )

    def _make_denoiser(self):
        import sherpa_onnx

        model = self.settings.denoise_model
        if not model.is_file():
            raise FileNotFoundError(f"denoiser model file missing: {model}")
        config = sherpa_onnx.OfflineSpeechDenoiserConfig(
            model=sherpa_onnx.OfflineSpeechDenoiserModelConfig(
                gtcrn=sherpa_onnx.OfflineSpeechDenoiserGtcrnModelConfig(model=str(model)),
                num_threads=1,
            )
        )
        denoiser = sherpa_onnx.OfflineSpeechDenoiser(config)
        if denoiser.sample_rate != SAMPLE_RATE:
            raise RuntimeError(f"denoiser sample rate {denoiser.sample_rate} != {SAMPLE_RATE}")
        log.info("speech denoiser enabled (%s)", model.name)
        return denoiser

    def _recognizer(self, language: str):
        """One recognizer per language prompt, created on first use (~1 GB RAM each)."""
        import sherpa_onnx

        recognizer = self._recognizers.get(language)
        if recognizer is not None:
            return recognizer
        asr = self.settings.asr_dir
        started = time.perf_counter()
        recognizer = sherpa_onnx.OfflineRecognizer.from_funasr_nano(
            encoder_adaptor=str(asr / "encoder_adaptor.int8.onnx"),
            llm=str(asr / "llm.int8.onnx"),
            embedding=str(asr / "embedding.int8.onnx"),
            tokenizer=str(asr / "Qwen3-0.6B"),
            num_threads=self.settings.num_threads,
            language=language,
            itn=self.settings.itn,
        )
        self._recognizers[language] = recognizer
        log.info(
            "loaded %s for language %r in %.1fs",
            self.settings.asr_model_name,
            language or "auto",
            time.perf_counter() - started,
        )
        return recognizer

    def transcribe(self, samples: np.ndarray, language: str = "", hotwords: str = "") -> str:
        """Decode one utterance of float32 16 kHz mono samples. Blocks; safe from any thread.

        `language` is a Nano prompt name ("中文", "英文", "日文") or "" for the default.
        """
        if samples.size == 0:
            return ""
        with self._lock:
            if self._denoiser is not None:
                samples = np.ascontiguousarray(
                    self._denoiser.run(np.ascontiguousarray(samples), SAMPLE_RATE).samples,
                    dtype=np.float32,
                )
            recognizer = self._recognizer(language or self.settings.language)
            stream = recognizer.create_stream()
            if hotwords and self.supports_stream_hotwords:
                stream.set_option("hotwords", hotwords)
            stream.accept_waveform(SAMPLE_RATE, samples)
            recognizer.decode_stream(stream)
            return stream.result.text.strip()

    def make_vad(self, threshold: float | None = None):
        import sherpa_onnx

        s = self.settings
        config = sherpa_onnx.VadModelConfig(
            silero_vad=sherpa_onnx.SileroVadModelConfig(
                model=str(s.vad_model),
                threshold=s.vad_threshold if threshold is None else threshold,
                min_silence_duration=s.vad_min_silence_s,
                min_speech_duration=s.vad_min_speech_s,
                window_size=VAD_WINDOW,
                max_speech_duration=s.vad_max_speech_s,
            ),
            sample_rate=SAMPLE_RATE,
            num_threads=1,
        )
        return sherpa_onnx.VoiceActivityDetector(
            config, buffer_size_in_seconds=s.vad_max_speech_s + 10
        )

    def transcribe_file(self, samples: np.ndarray, language: str = "") -> list[Segment]:
        """Split a whole recording with VAD and decode each speech segment."""
        vad = self.make_vad()
        segments: list[Segment] = []
        pre = self.settings.pre_roll_ms * SAMPLE_RATE // 1000
        post = self.settings.post_roll_ms * SAMPLE_RATE // 1000

        def drain() -> None:
            while not vad.empty():
                seg = vad.front
                length = len(seg.samples)
                audio = samples[
                    max(seg.start - pre, 0) : min(seg.start + length + post, len(samples))
                ]
                text = self.transcribe(audio, language)
                if text:
                    start = seg.start / SAMPLE_RATE
                    segments.append(Segment(start, start + length / SAMPLE_RATE, text))
                vad.pop()

        for i in range(0, len(samples) - VAD_WINDOW + 1, VAD_WINDOW):
            vad.accept_waveform(samples[i : i + VAD_WINDOW])
            drain()
        vad.flush()
        drain()
        return segments
