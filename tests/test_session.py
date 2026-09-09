"""StreamSession behaviour with a scripted VAD and a deterministic fake decoder."""

import asyncio

import numpy as np
import pytest

from funasr_ws.config import VAD_WINDOW
from funasr_ws.session import SessionOptions, StreamSession


class FakeSegment:
    def __init__(self, start: int, samples: np.ndarray) -> None:
        self.start = start
        self.samples = samples


class ScriptedVad:
    """Speech is 'detected' while the incoming window has any non-zero sample.

    A sentence ends when silence follows speech; flush ends an open sentence.
    """

    def __init__(self) -> None:
        self._speaking = False
        self._current: list[np.ndarray] = []
        self._start = 0
        self._fed = 0
        self._queue: list[FakeSegment] = []

    def accept_waveform(self, samples: np.ndarray) -> None:
        loud = bool(np.any(samples != 0))
        if loud and not self._speaking:
            self._speaking = True
            self._start = self._fed
            self._current = []
        if self._speaking and loud:
            self._current.append(samples.copy())
        if not loud and self._speaking:
            self._close()
        self._fed += len(samples)

    def _close(self) -> None:
        self._queue.append(FakeSegment(self._start, np.concatenate(self._current)))
        self._speaking = False
        self._current = []

    def is_speech_detected(self) -> bool:
        return self._speaking

    def empty(self) -> bool:
        return not self._queue

    @property
    def front(self) -> FakeSegment:
        return self._queue[0]

    def pop(self) -> None:
        self._queue.pop(0)

    def flush(self) -> None:
        if self._speaking:
            self._close()

    @property
    def current_segment(self) -> FakeSegment:
        return FakeSegment(self._start, np.concatenate(self._current))


def fake_transcribe(audio: np.ndarray) -> str:
    """Text length grows with audio length so partials produce deltas."""
    n = len(audio) // VAD_WINDOW
    return "".join(str(i % 10) for i in range(n))


def pcm(windows: int, loud: bool) -> bytes:
    value = 1000 if loud else 0
    return np.full(windows * VAD_WINDOW, value, dtype=np.int16).tobytes()


@pytest.fixture
def frames():
    return []


@pytest.fixture
def session(frames):
    async def send(frame):
        frames.append(frame)

    return StreamSession(
        vad=ScriptedVad(),
        transcribe=fake_transcribe,
        send=send,
        options=SessionOptions(
            "t", partial_interval_samples=4 * VAD_WINDOW, partial_window_samples=10**6
        ),
    )


async def settle(session: StreamSession) -> None:
    for _ in range(20):
        await asyncio.sleep(0)
    if session._partial_task:
        await session._partial_task


@pytest.mark.asyncio
async def test_partials_then_final(session, frames):
    await session.feed(pcm(4, True))
    await settle(session)
    await session.feed(pcm(4, True))
    await settle(session)
    online = [f for f in frames if f["mode"] == "2pass-online"]
    assert "".join(f["text"] for f in online) == "01234567"
    await session.feed(pcm(1, False))
    await settle(session)
    offline = [f for f in frames if f["mode"] == "2pass-offline"]
    assert offline == [
        {"mode": "2pass-offline", "wav_name": "t", "text": "01234567", "is_final": False}
    ]


@pytest.mark.asyncio
async def test_end_utterance_flushes_open_sentence(session, frames):
    await session.feed(pcm(3, True))
    await session.end_utterance()
    await settle(session)
    assert frames[-1] == {"mode": "2pass-offline", "wav_name": "t", "text": "012", "is_final": True}


@pytest.mark.asyncio
async def test_odd_byte_and_sub_window_frames_are_buffered(session, frames):
    data = pcm(2, True)
    for i in range(0, len(data), 333):
        await session.feed(data[i : i + 333])
    await session.feed(pcm(1, False))
    await settle(session)
    assert [f["text"] for f in frames if f["mode"] == "2pass-offline"] == ["01"]


@pytest.mark.asyncio
async def test_silence_emits_nothing(session, frames):
    await session.feed(pcm(10, False))
    await session.end_utterance()
    await settle(session)
    assert frames == []


@pytest.mark.asyncio
async def test_pre_and_post_roll_pad_the_decoded_audio(frames):
    lengths = []

    def transcribe(audio):
        lengths.append(len(audio))
        return "x"

    async def send(frame):
        frames.append(frame)

    session = StreamSession(
        vad=ScriptedVad(),
        transcribe=transcribe,
        send=send,
        options=SessionOptions(
            "t",
            partial_interval_samples=10**6,
            partial_window_samples=10**6,
            pre_roll_samples=VAD_WINDOW,
            post_roll_samples=VAD_WINDOW,
            history_samples=8 * VAD_WINDOW,
        ),
    )
    await session.feed(pcm(2, False))
    await session.feed(pcm(3, True))
    await session.feed(pcm(2, False))
    await settle(session)
    # 3 speech windows + 1 window before + 1 window after the VAD boundaries.
    assert lengths == [5 * VAD_WINDOW]
    assert frames[-1]["mode"] == "2pass-offline"
