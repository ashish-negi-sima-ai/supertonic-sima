"""HTTP/browser transport checks; no ONNX Runtime or Modalix device required.

Run with: PYTHONPATH=app python -m unittest discover -s tests
"""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
import http.client
import json
from queue import Empty
from threading import Event, Thread
from types import SimpleNamespace
import unittest

import numpy as np

from supertonic_sima.broadcast import AudioBroadcast
from supertonic_sima.server import SpeechApplication, create_server


class FakeEngine:
    steps = 8
    vocoder_backend = "test"

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.voices: list[str] = []
        self.options: list[dict] = []

    def synthesize(self, text: str, **kwargs):
        self.calls.append(text)
        self.voices.append(kwargs["voice"])
        self.options.append(kwargs)
        if text == "overflow":
            raise ValueError("predicted latent length 210 exceeds static limit 192")
        return SimpleNamespace(
            waveform=np.zeros(4410, dtype=np.float32), sample_rate=44100,
            audio_seconds=.1, generation_seconds=.01, real_time_factor=.1,
            latent_length=2, text_length=len(text),
        )


class BroadcastTests(unittest.TestCase):
    def test_live_only_delivery_reaches_each_current_listener(self):
        hub = AudioBroadcast()
        self.assertEqual(hub.publish(b"old", "old speech"), 0)
        first, second = hub.subscribe(), hub.subscribe()
        self.assertEqual(hub.publish(b"wave", "hello"), 2)
        self.assertEqual(first.get_nowait(), second.get_nowait())
        hub.unsubscribe(first)
        self.assertEqual(hub.publish(b"next", "next"), 1)
        with self.assertRaises(Empty):
            first.get_nowait()

    def test_slow_listener_is_removed_without_blocking_other_listeners(self):
        hub = AudioBroadcast(queue_size=1)
        slow, fast = hub.subscribe(), hub.subscribe()
        hub.publish(b"one", "one")
        fast.get_nowait()
        self.assertEqual(hub.publish(b"two", "two"), 1)
        self.assertIsNone(slow.get_nowait())
        self.assertIn(b'"text": "two"', fast.get_nowait())
        self.assertEqual(hub.listener_count, 1)

    def test_listener_limit_and_close(self):
        hub = AudioBroadcast(max_listeners=1)
        listener = hub.subscribe()
        with self.assertRaisesRegex(ValueError, "too many"):
            hub.subscribe()
        hub.publish(b"pending", "pending")
        hub.close()
        self.assertEqual(hub.listener_count, 0)
        self.assertIsNone(listener.get_nowait())

    def test_interrupt_flushes_queues_and_rejects_old_results_and_controls(self):
        hub = AudioBroadcast()
        listener = hub.subscribe()
        first = hub.interrupt("client", 1)
        hub.publish(b"old", "old", generation=first)
        second = hub.interrupt("client", 2)
        self.assertIn(b'event: interrupt', listener.get_nowait())
        self.assertIsNone(hub.publish(b"late", "late", generation=first))
        self.assertIsNone(hub.interrupt("client", 1))
        self.assertEqual(hub.interrupt("client", 2), second)  # Idempotent retry, no flush.
        with self.assertRaises(Empty):
            listener.get_nowait()
        self.assertEqual(hub.publish(b"new", "new", generation=second), 1)
        self.assertIn(b'"text": "new"', listener.get_nowait())
        hub.interrupt("another-client", 1)
        self.assertIsNone(hub.interrupt("client", 2))


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.engine = FakeEngine()
        self.app = SpeechApplication(self.engine, index_html="<p>Existing GUI</p>")
        self.server = create_server("127.0.0.1", 0, self.app)
        self.thread = Thread(target=self.server.serve_forever, kwargs={"poll_interval": .01})
        self.thread.start()
        self.connections: list[http.client.HTTPConnection] = []

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        for connection in self.connections:
            connection.close()

    def connection(self):
        connection = http.client.HTTPConnection(*self.server.server_address, timeout=2)
        self.connections.append(connection)
        return connection

    def request(self, path, payload=None):
        connection = self.connection()
        connection.request(
            "POST" if payload is not None else "GET", path,
            body=json.dumps(payload) if payload is not None else None,
            headers={"Content-Type": "application/json"} if payload is not None else {},
        )
        response = connection.getresponse()
        return response.status, response.read(), dict(response.getheaders())

    def subscribe(self):
        connection = self.connection()
        connection.request("GET", "/listen/events")
        response = connection.getresponse()
        self.assertEqual(response.status, 200)
        self.assertIn("text/event-stream", response.getheader("Content-Type"))
        self.assertEqual(
            self.event(response), ("ready", {
                "generation": self.app.broadcast.generation,
                "settings": self.app.broadcast.settings,
            }),
        )
        self.addCleanup(response.close)
        return response

    @staticmethod
    def event(response):
        event, data = "", ""
        while True:
            line = response.readline().decode("utf-8").rstrip("\r\n")
            if not line:
                return event, json.loads(data)
            if line.startswith("event: "):
                event = line[7:]
            if line.startswith("data: "):
                data = line[6:]

    def test_pages_and_health(self):
        status, html, _ = self.request("/listen")
        self.assertEqual(status, 200)
        self.assertIn(b"Enable audio", html)
        self.assertIn(b"EventSource('/listen/events')", html)
        self.assertEqual(self.request("/")[1], b"<p>Existing GUI</p>")
        self.assertEqual(json.loads(self.request("/health")[1])["browser_listeners"], 0)

    def test_no_listener_skips_inference(self):
        status, body, _ = self.request("/v1/speech/broadcast", {"input": "hello"})
        self.assertEqual(status, 409)
        self.assertIn("Enable audio", json.loads(body)["error"])
        self.assertEqual(self.engine.calls, [])

    def test_broadcast_synthesizes_once_and_streams_wav_to_two_browsers(self):
        first, second = self.subscribe(), self.subscribe()
        self.assertEqual(json.loads(self.request("/health")[1])["browser_listeners"], 2)
        status, body, _ = self.request("/v1/speech/broadcast", {"input": "Hello, café!"})
        self.assertEqual((status, json.loads(body)), (200, {"listeners": 2}))
        one, two = self.event(first), self.event(second)
        self.assertEqual(one, two)
        self.assertEqual(one[0], "audio")
        self.assertEqual(one[1]["text"], "Hello, café!")
        audio = base64.b64decode(one[1]["audio"])
        self.assertEqual(audio[:4], b"RIFF")
        self.assertEqual(audio[8:12], b"WAVE")
        self.assertEqual(self.engine.calls, ["Hello, café!"])

    def test_direct_speech_returns_wav_without_broadcast(self):
        listener = self.app.broadcast.subscribe()
        status, body, headers = self.request("/v1/speech", {"input": "direct"})
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "audio/wav")
        self.assertEqual(headers["X-Denoising-Steps"], "8")
        self.assertEqual(body[:4], b"RIFF")
        with self.assertRaises(Empty):
            listener.get_nowait()

    def test_profile_errors_remain_retryable_without_emitting_audio(self):
        listener = self.app.broadcast.subscribe()
        status, body, _ = self.request("/v1/speech/broadcast", {"input": "overflow"})
        self.assertEqual(status, 400)
        self.assertIn("exceeds static limit", json.loads(body)["error"])
        with self.assertRaises(Empty):
            listener.get_nowait()

    def test_closed_listeners_are_not_replayed_audio(self):
        first = self.subscribe()
        self.request("/v1/speech/broadcast", {"input": "earlier"})
        self.assertEqual(self.event(first)[1]["text"], "earlier")
        self.app.broadcast.close()
        self.assertEqual(self.event(first), ("reset", {}))
        second = self.subscribe()
        self.request("/v1/speech/broadcast", {"input": "new speech"})
        self.assertEqual(self.event(second)[1]["text"], "new speech")

    def test_interrupt_while_synthesis_runs_flushes_listener_and_drops_late_wav(self):
        listener = self.subscribe()
        _, body, _ = self.request("/v1/speech/interrupt", {"stream_id": "chat", "sequence": 1})
        first = json.loads(body)["generation"]
        self.assertEqual(self.event(listener), ("interrupt", {
            "generation": first, "settings": self.app.broadcast.settings,
        }))
        started, release = Event(), Event()
        synthesize = self.engine.synthesize

        def blocked(text, **kwargs):
            started.set()
            if not release.wait(timeout=2):
                raise RuntimeError("test did not release old synthesis")
            return synthesize(text, **kwargs)

        self.engine.synthesize = blocked
        with ThreadPoolExecutor(max_workers=1) as pool:
            old = pool.submit(self.request, "/v1/speech/broadcast", {
                "input": "old response", "generation": first,
            })
            try:
                self.assertTrue(started.wait(timeout=2))
                # Must work while engine.lock is held by the old synthesis.
                status, body, _ = self.request("/v1/speech/interrupt", {
                    "stream_id": "chat", "sequence": 2,
                })
                second = json.loads(body)["generation"]
                self.assertEqual(status, 200)
                self.assertEqual(self.event(listener), ("interrupt", {
                    "generation": second, "settings": self.app.broadcast.settings,
                }))
            finally:
                release.set()
            status, body, _ = old.result(timeout=2)
        self.assertEqual((status, json.loads(body)), (202, {"superseded": True}))
        # A delayed old control and delayed old chunk cannot interrupt or synthesize.
        for path, payload in (
            ("/v1/speech/interrupt", {"stream_id": "chat", "sequence": 1}),
            ("/v1/speech/broadcast", {"input": "old queued", "generation": first}),
        ):
            self.assertEqual(self.request(path, payload)[0], 202)
        self.request("/v1/speech/broadcast", {"input": "new response", "generation": second})
        event, payload = self.event(listener)
        self.assertEqual((event, payload["text"], payload["generation"]), (
            "audio", "new response", second,
        ))
        self.assertEqual(self.engine.calls, ["old response", "new response"])

    def test_interrupted_request_waiting_for_engine_does_not_run_inference(self):
        self.app.broadcast.subscribe()
        generation = self.app.broadcast.interrupt("chat", 1)
        queued = Event()
        synthesize = self.app.synthesize

        def waiting(payload, **kwargs):
            queued.set()
            return synthesize(payload, **kwargs)

        self.app.synthesize = waiting
        with ThreadPoolExecutor(max_workers=1) as pool:
            self.app.lock.acquire()
            old = pool.submit(self.request, "/v1/speech/broadcast", {
                "input": "queued old response", "generation": generation,
            })
            try:
                self.assertTrue(queued.wait(timeout=2))
                self.request("/v1/speech/interrupt", {"stream_id": "chat", "sequence": 2})
            finally:
                self.app.lock.release()
            self.assertEqual(old.result(timeout=2)[0], 202)
        self.assertEqual(self.engine.calls, [])

    def test_interrupt_without_listeners_succeeds_and_does_not_synthesize(self):
        status, body, _ = self.request(
            "/v1/speech/interrupt", {"stream_id": "chat", "sequence": 1},
        )
        self.assertEqual((status, json.loads(body)), (200, {"generation": 1}))
        self.assertEqual(self.engine.calls, [])

    def test_invalid_interrupt_does_not_advance_generation(self):
        for payload in ({}, {"stream_id": "chat", "sequence": -1},
                        {"stream_id": "chat", "sequence": True}):
            self.assertEqual(self.request("/v1/speech/interrupt", payload)[0], 400)
        self.assertEqual(self.app.broadcast.generation, 0)

    def test_speech_settings_are_validated_and_shared_with_connected_listeners(self):
        status, body, _ = self.request("/listen/settings")
        settings = json.loads(body)
        self.assertEqual(status, 200)
        self.assertIsNone(settings["voice"])
        self.assertIsNone(settings["language"])
        self.assertIsNone(settings["speed"])
        self.assertIn("hi", settings["languages"])
        self.assertIn("en", settings["languages"])
        self.assertEqual((settings["min_speed"], settings["max_speed"]), (.7, 2.0))
        self.assertEqual(
            settings["voices"], [f"{gender}{index}" for gender in "FM" for index in range(1, 6)],
        )
        first, second = self.subscribe(), self.subscribe()
        # Selecting settings must not wait for current MLA inference to finish.
        with self.app.lock:
            status, body, _ = self.request("/listen/settings", {
                "voice": "F3", "language": "fr", "speed": 1.25,
            })
        saved = {"voice": "F3", "language": "fr", "speed": 1.25, "revision": 1}
        self.assertEqual((status, json.loads(body)), (200, saved))
        self.assertEqual(self.event(first), ("settings", saved))
        self.assertEqual(self.event(second), ("settings", saved))
        self.assertEqual(json.loads(self.request("/listen/settings")[1])["voice"], "F3")
        self.assertEqual(self.app.broadcast.generation, 0)
        self.assertEqual(self.engine.calls, [])
        for invalid in (
            {"voice": "F6"}, {"voice": []}, {"voice": 1}, {}, {"voices": "M2"},
            {"language": "xx"}, {"language": []}, {"language": True},
            {"speed": .69}, {"speed": 2.01}, {"speed": True}, {"speed": "1.2"},
            {"speed": []}, {"speed": float("inf")}, {"speed": float("nan")},
            {"voice": "M2", "language": "de", "speed": 3},
        ):
            with self.subTest(invalid=invalid):
                self.assertEqual(self.request("/listen/settings", invalid)[0], 400)
                self.assertEqual(self.app.broadcast.settings, saved)

    def test_partial_settings_updates_preserve_other_preferences_and_allow_speed_bounds(self):
        self.request("/listen/settings", {"voice": "F1", "language": "de"})
        for speed in (.7, 2, 1.25, None):
            status, body, _ = self.request("/listen/settings", {"speed": speed})
            saved = json.loads(body)
            self.assertEqual(status, 200)
            self.assertEqual(
                (saved["voice"], saved["language"], saved["speed"]), ("F1", "de", speed),
            )
            # Repeating a setting does not increment its revision.
            self.assertEqual(
                json.loads(self.request("/listen/settings", {"speed": speed})[1]), saved,
            )

    def test_browser_settings_are_stable_per_response_and_can_restore_client_defaults(self):
        self.app.broadcast.subscribe()

        def start(sequence):
            status, body, _ = self.request("/v1/speech/interrupt", {
                "stream_id": "chat", "sequence": sequence,
            })
            self.assertEqual(status, 200)
            return json.loads(body)["generation"]

        def chunk(generation, text, voice="M1"):
            self.assertEqual(self.request("/v1/speech/broadcast", {
                "input": text, "generation": generation, "voice": voice,
                "language": "es", "speed": 1.3,
            })[0], 200)

        self.request("/listen/settings", {"voice": "F2", "language": "de", "speed": .8})
        first = start(1)
        chunk(first, "First chunk")
        self.request("/listen/settings", {"voice": "M3", "language": "hi", "speed": 1.6})
        chunk(first, "Same response keeps its settings")
        # An idempotent interrupt retry must not change a response's settings either.
        self.assertEqual(start(1), first)
        chunk(first, "Still the same settings")
        chunk(start(2), "Next response changes settings")
        # Direct WAV requests (including the original GUI) keep their requested settings.
        status, _, headers = self.request("/v1/speech", {
            "input": "Direct speech", "voice": "M5", "language": "fr", "speed": 1.1,
        })
        self.assertEqual(
            (status, headers["X-Voice"], headers["X-Language"], headers["X-Speed"]),
            (200, "M5", "fr", "1.1"),
        )
        self.request("/listen/settings", {"language": None})
        chunk(start(3), "Restore only the configured Jarvic language", voice="F4")
        self.request("/listen/settings", {"voice": None, "speed": None})
        chunk(start(4), "Restore all configured Jarvic settings", voice="F4")
        self.assertEqual(
            [(options["voice"], options["language"], options["speed"])
             for options in self.engine.options],
            [("F2", "de", .8)] * 3
            + [("M3", "hi", 1.6), ("M5", "fr", 1.1), ("M3", "es", 1.6), ("F4", "es", 1.3)],
        )


if __name__ == "__main__":
    unittest.main()
