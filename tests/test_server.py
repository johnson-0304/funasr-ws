"""HTTP and WebSocket endpoints with a fake engine (no model files needed)."""

import io
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from funasr_ws.config import SAMPLE_RATE, VAD_WINDOW, Settings
from funasr_ws.engine import Segment
from funasr_ws.server import create_app, decode_audio
from tests.test_session import ScriptedVad


class FakeEngine:
    supports_stream_language = True
    supports_stream_hotwords = False

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.calls: list[tuple[int, str, str]] = []

    def transcribe(self, samples, language="", hotwords=""):
        self.calls.append((len(samples), language, hotwords))
        return f"text{len(samples) // VAD_WINDOW}"

    def make_vad(self, threshold=None):
        self.threshold = threshold
        return ScriptedVad()

    def transcribe_file(self, samples, language=""):
        return [Segment(0.0, len(samples) / SAMPLE_RATE, self.transcribe(samples, language))]


def settings(tmp_path: Path) -> Settings:
    return Settings(
        model_dir=tmp_path,
        asr_model_name="m",
        num_threads=1,
        language="",
        itn=True,
        vad_threshold=0.5,
        vad_min_silence_s=0.4,
        vad_min_speech_s=0.2,
        vad_max_speech_s=15,
        partial_interval_ms=800,
        partial_window_s=12,
        pre_roll_ms=0,
        post_roll_ms=0,
        max_http_audio_s=5,
        host="127.0.0.1",
        port=0,
        log_level="info",
    )


@pytest.fixture
def client(tmp_path):
    app = create_app(settings(tmp_path), engine_factory=FakeEngine)
    with TestClient(app) as c:
        yield c


def wav_bytes(seconds: float, sr: int = SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.full(int(seconds * sr), 0.1, dtype=np.float32), sr, format="WAV")
    return buf.getvalue()


def test_health_and_models(client):
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/v1/models").json()["data"][0]["id"] == "fun-asr-nano"


def test_transcription_json_and_verbose(client):
    r = client.post(
        "/v1/audio/transcriptions", files={"file": ("a.wav", wav_bytes(1.0), "audio/wav")}
    )
    assert r.status_code == 200
    assert r.json()["text"].startswith("text")
    r = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", wav_bytes(1.0), "audio/wav")},
        data={"response_format": "verbose_json", "language": "zh"},
    )
    body = r.json()
    assert body["language"] == "中文"
    assert body["segments"][0]["end"] == pytest.approx(1.0)


def test_transcription_rejects_bad_input(client):
    r = client.post(
        "/v1/audio/transcriptions", files={"file": ("a.bin", b"nope", "application/octet-stream")}
    )
    assert r.status_code == 400
    r = client.post(
        "/v1/audio/transcriptions", files={"file": ("a.wav", wav_bytes(6.0), "audio/wav")}
    )
    assert r.status_code == 413
    r = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", wav_bytes(1.0), "audio/wav")},
        data={"language": "fr"},
    )
    assert r.status_code == 400


def test_decode_audio_resamples_to_16k():
    samples = decode_audio(wav_bytes(0.5, sr=24000))
    assert len(samples) == SAMPLE_RATE // 2
    assert samples.dtype == np.float32


def test_websocket_2pass_round_trip(client):
    with client.websocket_connect("/") as ws:
        ws.send_text(
            json.dumps(
                {"mode": "2pass", "wav_name": "w1", "svs_lang": "en", "speech_noise_thres": 0.7}
            )
        )
        assert ws.receive_json() == {"mode": "config", "text": "", "speech_noise_thres": 0.7}
        ws.send_bytes(np.full(3 * VAD_WINDOW, 1000, dtype=np.int16).tobytes())
        ws.send_bytes(np.zeros(VAD_WINDOW, dtype=np.int16).tobytes())
        frame = ws.receive_json()
        assert frame == {
            "mode": "2pass-offline",
            "wav_name": "w1",
            "text": "text3",
            "is_final": False,
        }
        ws.send_text(json.dumps({"is_speaking": False}))
    engine = client.app.state.engine
    assert engine.threshold == 0.7
    assert engine.calls[-1][1] == "英文"


def test_websocket_rejects_bad_start(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"audio_fs": 8000}))
        assert "error" in ws.receive_json()
