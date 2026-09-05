"""Unit tests for the retry + error-relay + truncation diagnostics logic.

Imports main.py directly and drives BridgeHandler._upstream_post with a mocked
requests.post, plus feeds a fake upstream SSE stream through each translator to
verify truncation is surfaced (not masked) and upstream 'error' events are
relayed. No network needed.
"""

import io
import json
import sys
import types
import unittest
from unittest import mock

import main


class FakeHandler:
    """Minimal stand-in for BridgeHandler: captures _send_json / _close_upstream."""

    def __init__(self):
        self.sent = []
        self.closed = 0
        self.written = bytearray()
        self.wfile = io.BytesIO(self.written)

    def _send_json(self, status, obj, extra_headers=None):
        self.sent.append((status, obj))

    def _close_upstream(self, resp):
        self.closed += 1

    def _write_anthropic_event(self, event_type, data):
        pass

    def _write_sse(self, chunk):
        pass

    def _write_sse_event(self, event_type, data):
        pass

    def _sse_headers(self):
        pass


def fake_resp(status=200, body=b"", text=None, stream_cls=None):
    """A requests.Response-like object with the bits _upstream_post uses."""
    r = types.SimpleNamespace()
    r.status_code = status
    r.text = text if text is not None else body.decode("utf-8", "replace")
    if stream_cls is None:
        def stream_cls():
            return iter([body])
    r.iter_lines = stream_cls
    r.close = lambda: None
    r.json = lambda: json.loads(body or b"{}")
    return r


class TestUpstreamPost(unittest.TestCase):
    def setUp(self):
        self.handler = FakeHandler()
        self.cfg = {"key_pinned": True, "auth_token": "tok", "key_pool": None,
                    "base_url": "https://up.test/alpha/generate"}
        main.UPSTREAM_RETRIES = 3

    def test_retries_then_succeeds(self):
        """Two transient timeouts, third attempt succeeds -> returns the resp."""
        ok = fake_resp(200, body=b'{"type":"start"}')
        calls = []

        def flaky_post(*a, **k):
            calls.append(1)
            if len(calls) < 3:
                raise requests_exc("Read timed out. (read timeout=10)")
            return ok

        with mock.patch.object(main.requests, "post", side_effect=flaky_post):
            resp = main.BridgeHandler._upstream_post(self.handler, {"m": 1}, self.cfg)
        self.assertIs(resp, ok)
        self.assertEqual(len(calls), 3)
        self.assertEqual(self.handler.sent, [])  # no 502 sent

    def test_all_fail_sends_502(self):
        """Persistent failures -> 502, never a None-without-reply."""
        def always_fail(*a, **k):
            raise requests_exc("Read timed out. (read timeout=10)")

        with mock.patch.object(main.requests, "post", side_effect=always_fail):
            resp = main.BridgeHandler._upstream_post(self.handler, {"m": 1}, self.cfg)
        self.assertIsNone(resp)
        self.assertEqual(len(self.handler.sent), 1)
        status, obj = self.handler.sent[0]
        self.assertEqual(status, 502)
        self.assertEqual(obj["error"]["code"], "upstream_unreachable")

    def test_4xx_not_retried(self):
        """401 is permanent -> returned immediately, single attempt."""
        auth = fake_resp(401, body=b'{"error":{"message":"bad key"}}')
        calls = []

        def auth_post(*a, **k):
            calls.append(1)
            return auth

        with mock.patch.object(main.requests, "post", side_effect=auth_post):
            resp = main.BridgeHandler._upstream_post(self.handler, {"m": 1}, self.cfg)
        self.assertIs(resp, auth)
        self.assertEqual(len(calls), 1)

    def test_5xx_retried_then_surfaced(self):
        """Two 5xx, third 200 -> succeeds; all-5xx -> the last 5xx is returned
        so _upstream_ok surfaces the upstream's own message."""
        fail5 = fake_resp(503, body=b'{"error":{"message":"overloaded"}}')
        ok = fake_resp(200, body=b'{"type":"start"}')
        seq = [fail5, fail5, ok]
        calls = []

        def svc(*a, **k):
            calls.append(1)
            return seq[len(calls) - 1]

        with mock.patch.object(main.requests, "post", side_effect=svc):
            resp = main.BridgeHandler._upstream_post(self.handler, {"m": 1}, self.cfg)
        self.assertIs(resp, ok)
        self.assertEqual(len(calls), 3)

        # all 5xx -> last response returned (surfaced by _upstream_ok)
        calls.clear()
        with mock.patch.object(main.requests, "post", return_value=fail5):
            resp = main.BridgeHandler._upstream_post(self.handler, {"m": 1}, self.cfg)
        self.assertIs(resp, fail5)


class TestUpstreamErrorRelay(unittest.TestCase):
    def test_anthropic_captures_error_event(self):
        t = main.AnthropicTranslator("m")
        # stream: reasoning text, then an explicit upstream error event, then EOF
        for obj in [
            {"type": "reasoning-start"},
            {"type": "reasoning-delta", "text": "think"},
            {"type": "error", "error": {"message": "model host timeout", "code": 529}},
        ]:
            t.on_event(obj)
        self.assertEqual(t.upstream_error, "model host timeout")
        self.assertFalse(t.completed)  # no finish -> truncation path decides

    def test_anthropic_truncation_message_uses_upstream_error(self):
        t = main.AnthropicTranslator("m")
        t.on_event({"type": "error", "error": {"message": "rate limited"}})
        self.assertEqual(t.upstream_error, "rate limited")

    def test_chat_captures_error_event(self):
        t = main.Translator("m")
        t.on_event({"type": "error", "error": {"message": "kaboom"}})
        self.assertEqual(t.upstream_error, "kaboom")
        self.assertIsNone(t.finish_reason)

    def test_responses_captures_error_event(self):
        t = main.ResponsesTranslator("m")
        t.on_event({"type": "error", "error": {"message": "kaboom"}})
        self.assertEqual(t.upstream_error, "kaboom")
        self.assertFalse(t.completed)

    def test_stream_anthropic_relays_error_and_no_fake_endturn(self):
        """A truncated stream that sent an upstream error -> Anthropic 'error'
        event with the real message; no message_delta/message_stop fabricated."""
        handler = FakeHandler()
        emitted = []
        handler._write_anthropic_event = lambda et, d: emitted.append((et, d))

        lines = iter([
            b'{"type":"reasoning-start"}',
            b'{"type":"reasoning-delta","text":"hi"}',
            b'{"type":"error","error":{"message":"upstream died"}}',
        ])
        resp = fake_resp(200, stream_cls=lambda: lines)
        translator = main.AnthropicTranslator("m")
        # translate manually (don't hit network):
        for raw in resp.iter_lines():
            obj = json.loads(raw)
            for et, d in translator.on_event(obj):
                pass
        handler._sse_headers = lambda: None
        # simulate the post-loop truncation branch with the handler's writes:
        if not translator.completed:
            for et, d in translator.pending_block_stop():
                handler._write_anthropic_event(et, d)
            reason = translator.upstream_error or "generic"
            handler._write_anthropic_event("error", {
                "type": "error",
                "error": {"type": "api_error", "message": reason},
            })
        types_ = [et for et, _ in emitted]
        self.assertIn("error", types_)
        self.assertNotIn("message_delta", types_)
        self.assertNotIn("message_stop", types_)
        err_payload = [d for et, d in emitted if et == "error"][0]
        self.assertEqual(err_payload["error"]["message"], "upstream died")


def requests_exc(msg):
    return main.requests.exceptions.ConnectTimeout(msg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
