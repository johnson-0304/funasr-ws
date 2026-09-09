"""One streaming recognition session: PCM in, FunASR 2pass frames out.

Silero VAD finds sentence boundaries. While a sentence is in progress the audio so far is
re-decoded every `partial_interval_ms` and the new text is sent as a `2pass-online` frame;
when the VAD closes the sentence its full audio is decoded once more and sent as the
`2pass-offline` frame. Decodes run in a worker thread so the socket keeps reading audio.

VAD segments start at the frame where speech was confirmed, which clips soft onsets
("Please" became "Lee's"), so a short history of raw audio is kept and every decode is
padded with `pre_roll` samples before and `post_roll` samples after the VAD boundaries.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import SAMPLE_RATE, VAD_WINDOW
from .protocol import PartialTracker, offline_frame, online_frame

log = logging.getLogger(__name__)

Send = Callable[[dict], Awaitable[None]]
Transcribe = Callable[[np.ndarray], str]  # blocking; language/hotwords already bound


@dataclass(frozen=True)
class SessionOptions:
    wav_name: str
    partial_interval_samples: int
    partial_window_samples: int
    pre_roll_samples: int = 0
    post_roll_samples: int = 0
    history_samples: int = 30 * SAMPLE_RATE  # must cover max_speech_duration + padding
    dump_dir: Path | None = None  # debug: write every finalized sentence as wav + txt


class StreamSession:
    def __init__(self, vad, transcribe: Transcribe, send: Send, options: SessionOptions) -> None:
        self._vad = vad
        self._transcribe = transcribe
        self._send = send
        self._opt = options
        self._pending = np.zeros(0, dtype=np.float32)  # samples not yet a full VAD window
        self._tail = b""  # odd trailing byte of the last PCM frame
        self._history = np.zeros(0, dtype=np.float32)  # most recent raw audio fed to the VAD
        self._fed = 0  # total samples fed to the VAD; VAD segment starts index this timeline
        self._tracker = PartialTracker()
        self._speaking = False
        self._since_partial = 0
        self._partial_task: asyncio.Task | None = None
        self._generation = 0  # bumps on every sentence end; stale partials are dropped
        self._finalizing = asyncio.Lock()

    async def feed(self, pcm: bytes) -> None:
        data = self._tail + pcm
        if len(data) % 2:
            data, self._tail = data[:-1], data[-1:]
        else:
            self._tail = b""
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        buf = np.concatenate([self._pending, samples])
        n_full = (len(buf) // VAD_WINDOW) * VAD_WINDOW
        self._pending = buf[n_full:]
        for i in range(0, n_full, VAD_WINDOW):
            self._push_window(buf[i : i + VAD_WINDOW])
            await self._after_window()

    async def end_utterance(self, is_final: bool = True) -> None:
        """Client said `is_speaking: false`: close the open sentence and emit its text."""
        if self._pending.size:
            padded = np.zeros(VAD_WINDOW, dtype=np.float32)
            padded[: self._pending.size] = self._pending
            self._pending = np.zeros(0, dtype=np.float32)
            self._push_window(padded)
        self._vad.flush()
        await self._drain(is_final=is_final)
        self._speaking = False

    async def close(self) -> None:
        if self._partial_task and not self._partial_task.done():
            self._partial_task.cancel()

    def _push_window(self, window: np.ndarray) -> None:
        self._vad.accept_waveform(window)
        self._fed += len(window)
        self._history = np.concatenate([self._history, window])[-self._opt.history_samples :]

    def _padded(self, start: int, length: int) -> np.ndarray:
        """Audio for VAD range [start, start+length) plus pre/post roll, from the history."""
        history_start = self._fed - len(self._history)
        a = max(start - self._opt.pre_roll_samples, history_start)
        b = min(start + length + self._opt.post_roll_samples, self._fed)
        return self._history[a - history_start : b - history_start]

    async def _after_window(self) -> None:
        speaking = self._vad.is_speech_detected()
        if speaking and not self._speaking:
            self._tracker.reset()
            self._since_partial = 0
        self._speaking = speaking
        if speaking:
            self._since_partial += VAD_WINDOW
            if self._since_partial >= self._opt.partial_interval_samples and self._partial_idle():
                self._since_partial = 0
                self._partial_task = asyncio.create_task(self._emit_partial(self._generation))
        await self._drain(is_final=False)

    def _partial_idle(self) -> bool:
        return self._partial_task is None or self._partial_task.done()

    async def _emit_partial(self, generation: int) -> None:
        seg = self._vad.current_segment
        audio = self._padded(seg.start, len(seg.samples))
        if audio.size > self._opt.partial_window_samples:
            audio = audio[-self._opt.partial_window_samples :]
        try:
            text = await asyncio.to_thread(self._transcribe, audio)
        except Exception:
            log.exception("partial decode failed")
            return
        if generation != self._generation:
            return  # the sentence already ended; its offline frame carries the text
        delta = self._tracker.delta(text)
        if delta:
            await self._send(online_frame(self._opt.wav_name, delta))

    async def _drain(self, is_final: bool) -> None:
        async with self._finalizing:
            while not self._vad.empty():
                seg = self._vad.front
                audio = self._padded(seg.start, len(seg.samples))
                self._vad.pop()
                self._generation += 1
                self._tracker.reset()
                text = await asyncio.to_thread(self._transcribe, audio)
                log.info("sentence %.2fs -> %r", len(audio) / SAMPLE_RATE, text)
                if self._opt.dump_dir:
                    dump_sentence(self._opt.dump_dir, audio, text)
                await self._send(offline_frame(self._opt.wav_name, text, is_final))


def dump_sentence(directory: Path, audio: np.ndarray, text: str) -> None:
    """Debug aid: keep the exact audio a sentence was decoded from, next to its transcript."""
    import soundfile as sf

    directory.mkdir(parents=True, exist_ok=True)
    stem = directory / f"{time.strftime('%Y%m%d-%H%M%S')}-{int(time.time() * 1000) % 1000:03d}"
    sf.write(f"{stem}.wav", audio, SAMPLE_RATE, subtype="PCM_16")
    Path(f"{stem}.txt").write_text(text + "\n", encoding="utf-8")
