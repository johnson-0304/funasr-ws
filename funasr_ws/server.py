"""FastAPI application: FunASR-compatible WebSocket plus an OpenAI-compatible HTTP API."""

from __future__ import annotations

import asyncio
import io
import logging
from contextlib import asynccontextmanager
from functools import partial
from typing import Annotated

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, PlainTextResponse

from .config import SAMPLE_RATE, Settings, load_settings
from .engine import Engine
from .protocol import (
    ProtocolError,
    config_frame,
    error_frame,
    parse_control_frame,
    parse_language,
    parse_start_frame,
)
from .session import SessionOptions, StreamSession

log = logging.getLogger(__name__)
WS_POLICY_VIOLATION = 1008


def create_app(settings: Settings | None = None, engine_factory=Engine) -> FastAPI:
    settings = settings or load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        app.state.engine = await asyncio.to_thread(engine_factory, settings)
        yield

    app = FastAPI(title="funasr-ws", lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "model": settings.asr_model_name,
            "language": settings.language or "auto",
            "denoise": settings.denoise,
            "num_threads": settings.num_threads,
        }

    @app.get("/v1/models")
    async def models():
        return {
            "object": "list",
            "data": [{"id": "fun-asr-nano", "object": "model", "owned_by": "funasr-ws"}],
        }

    @app.post("/v1/audio/transcriptions")
    async def transcriptions(
        file: Annotated[UploadFile, File()],
        model: Annotated[str, Form()] = "fun-asr-nano",
        language: Annotated[str | None, Form()] = None,
        response_format: Annotated[str, Form()] = "json",
    ):
        if response_format not in ("json", "text", "verbose_json"):
            raise HTTPException(400, "response_format must be json, text or verbose_json")
        try:
            lang = parse_language(language)
        except ProtocolError as err:
            raise HTTPException(400, str(err)) from err
        raw = await file.read()
        try:
            samples = decode_audio(raw)
        except (RuntimeError, ValueError) as err:
            raise HTTPException(400, f"cannot decode audio: {err}") from err
        duration = len(samples) / SAMPLE_RATE
        if duration > settings.max_http_audio_s:
            raise HTTPException(413, f"audio longer than {settings.max_http_audio_s:.0f}s")
        engine: Engine = app.state.engine
        segments = await asyncio.to_thread(engine.transcribe_file, samples, lang)
        text = (
            "".join(s.text for s in segments)
            if lang == "中文"
            else " ".join(s.text for s in segments)
        )
        if response_format == "text":
            return PlainTextResponse(text)
        if response_format == "verbose_json":
            return {
                "task": "transcribe",
                "language": lang or "auto",
                "duration": duration,
                "text": text,
                "segments": [
                    {"id": i, "start": s.start_s, "end": s.end_s, "text": s.text}
                    for i, s in enumerate(segments)
                ],
            }
        return {"text": text}

    async def websocket_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        engine: Engine = app.state.engine
        try:
            first = await ws.receive()
        except WebSocketDisconnect:
            return
        if "text" not in first:
            await ws.send_json(error_frame("first frame must be the JSON start frame"))
            await ws.close(code=WS_POLICY_VIOLATION)
            return
        try:
            start = parse_start_frame(first["text"])
        except ProtocolError as err:
            await ws.send_json(error_frame(str(err)))
            await ws.close(code=WS_POLICY_VIOLATION)
            return
        if start.hotwords and not engine.supports_stream_hotwords:
            log.warning(
                "%s: hotwords unsupported by this model build; ignoring %s",
                start.wav_name,
                start.hotwords,
            )
        if start.speech_noise_threshold is not None:
            await ws.send_json(config_frame(start.speech_noise_threshold))
        transcribe = partial(
            engine.transcribe, language=start.language, hotwords=",".join(start.hotwords)
        )
        session = StreamSession(
            vad=engine.make_vad(start.speech_noise_threshold),
            transcribe=transcribe,
            send=ws.send_json,
            options=SessionOptions(
                wav_name=start.wav_name,
                partial_interval_samples=settings.partial_interval_ms * SAMPLE_RATE // 1000,
                partial_window_samples=int(settings.partial_window_s * SAMPLE_RATE),
                pre_roll_samples=settings.pre_roll_ms * SAMPLE_RATE // 1000,
                post_roll_samples=settings.post_roll_ms * SAMPLE_RATE // 1000,
                history_samples=int((settings.vad_max_speech_s + 5) * SAMPLE_RATE),
                dump_dir=settings.dump_dir,
            ),
        )
        log.info(
            "ws open %s language=%r threshold=%s",
            start.wav_name,
            start.language or "auto",
            start.speech_noise_threshold,
        )
        try:
            while True:
                message = await ws.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                if "bytes" in message and message["bytes"] is not None:
                    await session.feed(message["bytes"])
                elif "text" in message and message["text"] is not None:
                    try:
                        control = parse_control_frame(message["text"])
                    except ProtocolError as err:
                        await ws.send_json(error_frame(str(err)))
                        await ws.close(code=WS_POLICY_VIOLATION)
                        return
                    if control.get("is_speaking") is False:
                        await session.end_utterance()
        except WebSocketDisconnect:
            pass
        finally:
            await session.close()
            log.info("ws closed %s", start.wav_name)

    app.add_api_websocket_route("/", websocket_endpoint)
    app.add_api_websocket_route("/ws", websocket_endpoint)

    @app.exception_handler(Exception)
    async def unhandled(_, exc: Exception):
        log.exception("unhandled error", exc_info=exc)
        return JSONResponse(status_code=500, content={"error": str(exc)})

    return app


def decode_audio(raw: bytes) -> np.ndarray:
    """Decode any libsndfile-supported container to float32 mono 16 kHz."""
    data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != SAMPLE_RATE:
        n = int(round(len(mono) * SAMPLE_RATE / sr))
        mono = np.interp(
            np.linspace(0, len(mono) - 1, n, dtype=np.float64),
            np.arange(len(mono), dtype=np.float64),
            mono,
        ).astype(np.float32)
    return np.ascontiguousarray(mono, dtype=np.float32)
