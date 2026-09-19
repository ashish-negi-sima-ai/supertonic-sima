"""Bounded, live-only audio delivery to connected browser listeners."""

from __future__ import annotations

import base64
from collections import OrderedDict
import json
from queue import Empty, Full, Queue
from threading import Lock

from .text import AVAILABLE_VOICES


class AudioBroadcast:
    def __init__(self, *, max_listeners: int = 16, queue_size: int = 8) -> None:
        self._lock = Lock()
        self._listeners: set[Queue[bytes | None]] = set()
        self._max_listeners = max_listeners
        self._queue_size = queue_size
        self._generation = 0
        self._current: tuple[str, int] | None = None
        self._sequences: OrderedDict[str, int] = OrderedDict()
        self._voice: str | None = None
        self._response_voice: str | None = None
        self._settings_revision = 0

    def _settings(self) -> dict:
        return {"voice": self._voice, "revision": self._settings_revision}

    @property
    def settings(self) -> dict:
        with self._lock:
            return self._settings()

    def set_voice(self, voice: str | None) -> dict:
        if voice is not None and (not isinstance(voice, str) or voice not in AVAILABLE_VOICES):
            raise ValueError("voice must be F1–F5, M1–M5, or null for the Jarvic voice")
        with self._lock:
            if voice != self._voice:
                self._voice = voice
                self._settings_revision += 1
                if self._current is None:
                    self._response_voice = voice
                self._deliver(self._event("settings", self._settings()))
            return self._settings()

    @property
    def response_voice(self) -> str | None:
        """The selected voice is held constant for all chunks of one response."""
        with self._lock:
            return self._response_voice

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    def interrupt(self, stream_id: str, sequence: int) -> int | None:
        """Flush current speech and reject delayed/repeated control requests."""
        with self._lock:
            if sequence <= self._sequences.get(stream_id, 0):
                return self._generation if self._current == (stream_id, sequence) else None
            self._sequences[stream_id] = sequence
            self._sequences.move_to_end(stream_id)
            if len(self._sequences) > 64:
                self._sequences.popitem(last=False)
            self._current = (stream_id, sequence)
            self._generation += 1
            self._response_voice = self._voice
            event = self._event("interrupt", {
                "generation": self._generation, "settings": self._settings(),
            })
            for listener in self._listeners:
                self._clear(listener)
                listener.put_nowait(event)
            return self._generation

    @property
    def listener_count(self) -> int:
        with self._lock:
            return len(self._listeners)

    def subscribe(self) -> Queue[bytes | None]:
        with self._lock:
            if len(self._listeners) >= self._max_listeners:
                raise ValueError("too many browser listeners")
            listener: Queue[bytes | None] = Queue(maxsize=self._queue_size)
            self._listeners.add(listener)
            return listener

    def unsubscribe(self, listener: Queue[bytes | None]) -> None:
        with self._lock:
            self._listeners.discard(listener)

    @staticmethod
    def _clear(listener: Queue[bytes | None]) -> None:
        # Producers are serialized by _lock; a consumer may still drain items.
        while True:
            try:
                listener.get_nowait()
            except Empty:
                break

    @classmethod
    def _stop(cls, listener: Queue[bytes | None]) -> None:
        cls._clear(listener)
        listener.put_nowait(None)

    @staticmethod
    def _event(name: str, payload: dict) -> bytes:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        return b"event: " + name.encode("ascii") + b"\ndata: " + data + b"\n\n"

    def _deliver(self, event: bytes) -> int:
        """Deliver while holding _lock; disconnect listeners that fall behind."""
        delivered = 0
        for listener in tuple(self._listeners):
            try:
                listener.put_nowait(event)
                delivered += 1
            except Full:
                self._listeners.remove(listener)
                self._stop(listener)
        return delivered

    def publish(self, audio: bytes, text: str, *, generation: int | None = None) -> int | None:
        payload = {
            "audio": base64.b64encode(audio).decode("ascii"),
            "text": text,
        }
        with self._lock:
            if generation is not None and generation != self._generation:
                return None  # Synthesis completed after its response was interrupted.
            event = self._event("audio", {**payload, "generation": self._generation})
            return self._deliver(event)

    def close(self) -> None:
        with self._lock:
            for listener in self._listeners:
                self._stop(listener)
            self._listeners.clear()
