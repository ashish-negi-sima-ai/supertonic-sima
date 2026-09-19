"""Small standard-library HTTP server around one persistent TTS engine."""

from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from queue import Empty
from threading import Lock
from typing import TYPE_CHECKING, Any

from .audio import wav_bytes
from .broadcast import AudioBroadcast
from .text import AVAILABLE_LANGUAGES, AVAILABLE_VOICES, MAX_SPEED, MIN_SPEED

if TYPE_CHECKING:
    from .engine import SupertonicModalix

_LISTENER_HTML = Path(__file__).with_name("listener.html").read_bytes()


class SpeechSuperseded(Exception):
    """The queued request belongs to an interrupted response."""


class SpeechApplication:
    def __init__(self, engine: SupertonicModalix, index_html: str | None = None) -> None:
        self.engine = engine
        self.index_html = index_html
        self.lock = Lock()
        self.broadcast = AudioBroadcast()

    def synthesize(
        self, payload: dict[str, Any], *, generation: int | None = None,
    ) -> tuple[bytes, dict[str, str]]:
        text = payload.get("input")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("'input' must be a non-empty string")
        response_format = payload.get("response_format", "wav")
        if response_format != "wav":
            raise ValueError("only response_format='wav' is supported")
        voice = str(payload.get("voice", "M1"))
        language = str(payload.get("language", payload.get("lang", "en")))
        speed = float(payload.get("speed", 1.0))
        seed = int(payload.get("seed", 1101))
        if voice not in AVAILABLE_VOICES:
            raise ValueError(f"unsupported voice {voice!r}")
        if language not in AVAILABLE_LANGUAGES:
            raise ValueError(f"unsupported language {language!r}")
        if not MIN_SPEED <= speed <= MAX_SPEED:
            raise ValueError(f"speed must be between {MIN_SPEED} and {MAX_SPEED}")
        with self.lock:
            if generation is not None:
                settings = self.broadcast.settings_for_response(generation)
                if settings is None:
                    raise SpeechSuperseded
                voice = settings.get("voice", voice)
                language = settings.get("language", language)
                speed = settings.get("speed", speed)
            result = self.engine.synthesize(
                text, voice=voice, language=language, speed=speed, seed=seed
            )
        body = wav_bytes(result.waveform, result.sample_rate)
        headers = {
            "X-Voice": voice,
            "X-Language": language,
            "X-Speed": str(speed),
            "X-Audio-Length-Seconds": f"{result.audio_seconds:.6f}",
            "X-Generation-Length-Seconds": f"{result.generation_seconds:.6f}",
            "X-Real-Time-Factor": f"{result.real_time_factor:.6f}",
            "X-Latent-Length": str(result.latent_length),
            "X-Text-Length": str(result.text_length),
            "X-Denoising-Steps": str(self.engine.steps),
        }
        return body, headers


def create_server(
    host: str,
    port: int,
    application: SpeechApplication,
) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        server_version = "SupertonicSiMa/1"

        def _send(
            self,
            status: HTTPStatus,
            body: bytes,
            content_type: str,
            headers: dict[str, str] | None = None,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for name, value in (headers or {}).items():
                self.send_header(name, value)
            try:
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True  # Interrupted synthesis requests are expected.

        def _json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            self._send(
                status,
                json.dumps(payload, sort_keys=True).encode(),
                "application/json",
            )

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._json(
                    HTTPStatus.OK,
                    {
                        "status": "ok",
                        "vocoder_backend": application.engine.vocoder_backend,
                        "steps": application.engine.steps,
                        "browser_listeners": application.broadcast.listener_count,
                    },
                )
            elif self.path == "/listen":
                self._send(HTTPStatus.OK, _LISTENER_HTML, "text/html; charset=utf-8")
            elif self.path == "/listen/events":
                self._listen()
            elif self.path == "/listen/settings":
                self._json(HTTPStatus.OK, {
                    **application.broadcast.settings,
                    "voices": AVAILABLE_VOICES,
                    "languages": sorted(AVAILABLE_LANGUAGES),
                    "min_speed": MIN_SPEED,
                    "max_speed": MAX_SPEED,
                })
            elif self.path == "/config":
                self._json(
                    HTTPStatus.OK,
                    {
                        "languages": sorted(AVAILABLE_LANGUAGES),
                        "voices": AVAILABLE_VOICES,
                        "min_speed": MIN_SPEED,
                        "max_speed": MAX_SPEED,
                        "steps": application.engine.steps,
                    },
                )
            elif self.path == "/" and application.index_html is not None:
                self._send(
                    HTTPStatus.OK,
                    application.index_html.encode(),
                    "text/html; charset=utf-8",
                )
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def _listen(self) -> None:
            try:
                listener = application.broadcast.subscribe()
            except ValueError as error:
                self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": str(error)})
                return
            try:
                self.connection.settimeout(10)
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Accel-Buffering", "no")
                self.end_headers()
                ready = json.dumps({
                    "generation": application.broadcast.generation,
                    "settings": application.broadcast.settings,
                }).encode()
                self.wfile.write(b"event: ready\ndata: " + ready + b"\n\n")
                self.wfile.flush()
                while True:
                    try:
                        event = listener.get(timeout=5)
                    except Empty:
                        event = b": keepalive\n\n"
                    if event is None:
                        self.wfile.write(b"event: reset\ndata: {}\n\n")
                        self.wfile.flush()
                        return
                    self.wfile.write(event)
                    self.wfile.flush()
            except OSError:
                pass  # The browser closed the page or its connection stalled.
            finally:
                application.broadcast.unsubscribe(listener)

        def do_POST(self) -> None:  # noqa: N802
            broadcast = self.path == "/v1/speech/broadcast"
            if self.path not in {
                "/v1/speech", "/v1/speech/broadcast", "/v1/speech/interrupt", "/listen/settings",
            }:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                if content_length <= 0 or content_length > 1_000_000:
                    raise ValueError("invalid request body length")
                payload = json.loads(self.rfile.read(content_length))
                if not isinstance(payload, dict):
                    raise ValueError("request body must be a JSON object")
                if self.path == "/listen/settings":
                    self._json(HTTPStatus.OK, application.broadcast.update_settings(payload))
                    return
                if self.path == "/v1/speech/interrupt":
                    stream_id, sequence = payload.get("stream_id"), payload.get("sequence")
                    if not isinstance(stream_id, str) or not 1 <= len(stream_id) <= 128:
                        raise ValueError(
                            "stream_id must be a non-empty string of at most 128 characters"
                        )
                    if type(sequence) is not int or sequence < 1:
                        raise ValueError("sequence must be a positive integer")
                    generation = application.broadcast.interrupt(stream_id, sequence)
                    self._json(
                        HTTPStatus.ACCEPTED if generation is None else HTTPStatus.OK,
                        {"superseded": True} if generation is None else {"generation": generation},
                    )
                    return
                generation = None
                if broadcast:
                    generation = payload.get("generation", application.broadcast.generation)
                    if type(generation) is not int or generation < 0:
                        raise ValueError("generation must be a non-negative integer")
                    if generation != application.broadcast.generation:
                        raise SpeechSuperseded
                if broadcast and application.broadcast.listener_count == 0:
                    self._json(HTTPStatus.CONFLICT, {
                        "error": "No browser listeners. Open /listen on your host "
                                 "and click Enable audio.",
                    })
                    return
                raw_text = payload.get("input", "")
                raw_chars = len(raw_text) if isinstance(raw_text, str) else 0
                logged_text = json.dumps(raw_text, ensure_ascii=False)
                chunk_index = payload.get("chunk_index", 1)
                chunk_count = payload.get("chunk_count", 1)
                source_chars = payload.get("source_chars", raw_chars)
                boundary = str(payload.get("split_boundary", "single"))
                boundary = boundary.replace("\r", "").replace("\n", "")[:24]
                self.log_message(
                    "synthesis start chunk=%s/%s boundary=%s "
                    "raw_chars=%d source_chars=%s requested_language=%s requested_voice=%s "
                    "requested_speed=%s seed=%s steps=%d text=%s",
                    chunk_index,
                    chunk_count,
                    boundary,
                    raw_chars,
                    source_chars,
                    payload.get("language", payload.get("lang", "en")),
                    payload.get("voice", "M1"),
                    payload.get("speed", 1.0),
                    payload.get("seed", 1101),
                    application.engine.steps,
                    logged_text,
                )
                body, headers = application.synthesize(payload, generation=generation)
                self.log_message(
                    "synthesis done chunk=%s/%s voice=%s language=%s speed=%s processed_chars=%s "
                    "latent_frames=%s audio_s=%s generation_s=%s rtf=%s",
                    chunk_index,
                    chunk_count,
                    headers["X-Voice"],
                    headers["X-Language"],
                    headers["X-Speed"],
                    headers["X-Text-Length"],
                    headers["X-Latent-Length"],
                    headers["X-Audio-Length-Seconds"],
                    headers["X-Generation-Length-Seconds"],
                    headers["X-Real-Time-Factor"],
                )
                if broadcast:
                    listeners = application.broadcast.publish(
                        body, raw_text, generation=generation,
                    )
                    if listeners is None:
                        raise SpeechSuperseded
                    if listeners == 0:
                        self._json(HTTPStatus.CONFLICT, {
                            "error": "Browser listener disconnected. "
                                     "Open /listen and enable audio again.",
                        })
                    else:
                        self._json(HTTPStatus.OK, {"listeners": listeners})
                else:
                    self._send(HTTPStatus.OK, body, "audio/wav", headers)
            except SpeechSuperseded:
                self._json(HTTPStatus.ACCEPTED, {"superseded": True})
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            except Exception as error:  # noqa: BLE001
                self.log_error("synthesis failed: %s", error)
                self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(error)})

    class SpeechServer(ThreadingHTTPServer):
        def server_close(self) -> None:
            application.broadcast.close()
            super().server_close()

    return SpeechServer((host, port), Handler)
