#!/usr/bin/env python3
"""command-code bridge — an OpenAI-compatible /v1/chat/completions endpoint on
top of the command-code native /alpha/generate API.

    client (OpenAI JSON)  ──▶  this proxy  ──▶  /alpha/generate   (native NDJSON)
    GET /v1/models        ──▶  https://api.commandcode.ai/provider/v1/models

The native endpoint is text-only and stream-only. Three quirks shape everything
below:

  * `messages[].content` must be a **plain string** — structured content blocks
    are rejected with a schema error, so tool results and prior tool calls are
    flattened into text.
  * The system prompt travels in its own `params.system` field, not as a message.
  * Responses arrive as newline-delimited JSON events (not `data:` SSE frames),
    always with `stream: true` — a non-streaming request is refused upstream, so
    non-streaming callers get a buffered single-object response instead.

Single file, stdlib + `requests`.   Run:  python main.py
"""

import hmac
import json
import os
import re
import sys
import threading
import time
import urllib.parse
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(HERE, ".env")

# Every key can be overridden by the upper-cased environment variable of the
# same name (PORT=9000 python main.py); the .env file is the usual place.
DEFAULTS = {
    # upstream
    "base_url": "https://api.commandcode.ai/alpha/generate",
    "models_url": "",  # empty -> <base_url origin>/provider/v1/models
    "auth_token": "",
    "proxy": "",  # e.g. http://127.0.0.1:7890 — empty falls back to HTTP(S)_PROXY
    "connect_timeout": "15",
    "read_timeout": "600",
    # server
    "host": "0.0.0.0",
    "port": "8080",
    "api_key": "",  # optional: require this bearer token from clients
    "debug": "",
    # models
    "default_model": "deepseek/deepseek-v4-flash",
    "models": "",  # optional comma-separated filter/order for GET /v1/models
    "models_ttl": "300",  # seconds the upstream model list is cached
    # native request envelope (see build_native_body)
    "working_dir": "/tmp",
    "environment": "terminal",
    "memory": "",
    "taste": "",
    "skills": "",
    "permission_mode": "standard",
}

# The CLI's own headers. Verified against the live endpoint: the version header
# selects the event vocabulary, 0.38.2 is the shape handled by Translator.
UPSTREAM_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "text/event-stream",
    "x-command-code-version": "0.38.2",
    "x-cli-environment": "production",
}

MAX_OUTPUT_TOKENS = 200_000
MAX_ATTEMPTS = 3
RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

SESSION = requests.Session()


def load_env(path):
    """Minimal KEY=VALUE .env reader (keeps the project dependency-free)."""
    env = {}
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                env[key.strip()] = value.strip().strip('"').strip("'")
    except OSError:
        pass
    return env


# ---------------------------------------------------------------------------
# Upstream keys
# ---------------------------------------------------------------------------

# Checked in order when no higher-priority source yields a key. Each file may
# hold it under "apiKey", "commandcode", or {"command-code": {"type","key"}}.
AUTH_FILE_CANDIDATES = (
    "~/.commandcode/auth.json",
    "~/.pi/agent/auth.json",
    "~/.omp/agent/auth.json",
)


def read_auth_file(path):
    try:
        with open(os.path.expanduser(path), encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    for field in ("apiKey", "commandcode"):
        value = data.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    cc = data.get("command-code")
    if isinstance(cc, dict) and cc.get("type") == "api":
        value = cc.get("key")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def split_keys(raw):
    """Comma/newline-separated list -> trimmed, de-duplicated, order preserved."""
    if not raw:
        return []
    seen, out = set(), []
    for key in re.split(r"[,\n]", raw):
        key = key.strip()
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out


def resolve_server_keys(file_env):
    """Resolve the server-side upstream key(s), stopping at the first source
    that yields one:

        .env auth_token > COMMANDCODE_API_KEY > COMMANDCODE_API_KEYS > auth.json

    Returns (keys, pinned, source). `pinned` is set only by `.env auth_token`,
    which then beats any client-supplied key.
    """
    token = (file_env.get("auth_token") or "").strip()
    if token:
        return [token], True, ".env auth_token"
    single = (os.environ.get("COMMANDCODE_API_KEY")
              or file_env.get("COMMANDCODE_API_KEY") or "").strip()
    if single:
        return [single], False, "COMMANDCODE_API_KEY"
    pool = split_keys(os.environ.get("COMMANDCODE_API_KEYS")
                      or file_env.get("COMMANDCODE_API_KEYS"))
    if pool:
        return pool, False, "COMMANDCODE_API_KEYS"
    for path in AUTH_FILE_CANDIDATES:
        key = read_auth_file(path)
        if key:
            return [key], False, path
    return [], False, "none"


class KeyPool:
    """Thread-safe round-robin over the server-side keys."""

    def __init__(self, keys):
        self._keys = [k for k in keys if k]
        self._i = 0
        self._lock = threading.Lock()

    def next(self):
        if not self._keys:
            return ""
        with self._lock:
            key = self._keys[self._i % len(self._keys)]
            self._i += 1
            return key

    def __len__(self):
        return len(self._keys)


def mask(key):
    return key if len(key) <= 12 else f"{key[:8]}…{key[-4:]}"


def client_key_from_headers(headers):
    """A caller may bring its own upstream key. Only a key that actually looks
    like one is forwarded — clients such as Claude Code and the OpenAI SDK
    insist on setting *some* bearer token, and forwarding a placeholder like
    "sk-none" would turn a working server key into a 401."""
    raw = (headers.get("x-api-key") or "").strip()
    if not raw:
        auth = headers.get("Authorization") or ""
        if auth.lower().startswith("bearer "):
            raw = auth[7:].strip()
    return raw if raw.startswith("user_") else ""


def pick_upstream_key(cfg, client_key=""):
    if cfg["key_pinned"]:
        return cfg["api_keys"][0]
    if client_key:
        return client_key
    return cfg["key_pool"].next()


def parse_proxy(raw):
    """A single proxy URL applies to both schemes. Empty returns None so that
    requests falls back to the standard HTTP_PROXY / HTTPS_PROXY variables."""
    raw = (raw or "").strip()
    if not raw:
        return None
    return {"http": raw, "https": raw}


def derive_models_url(base_url):
    parts = urllib.parse.urlsplit(base_url)
    if parts.scheme and parts.netloc:
        return f"{parts.scheme}://{parts.netloc}/provider/v1/models"
    return "https://api.commandcode.ai/provider/v1/models"


def load_config():
    file_env = load_env(ENV_PATH)
    cfg = {}
    for key, default in DEFAULTS.items():
        value = os.environ.get(key.upper())
        if value is None:
            value = file_env.get(key)
        cfg[key] = default if value in (None, "") else value
    cfg["port"] = int(cfg["port"])
    cfg["models_ttl"] = max(0, int(cfg["models_ttl"]))
    cfg["connect_timeout"] = float(cfg["connect_timeout"])
    cfg["read_timeout"] = float(cfg["read_timeout"])
    cfg["proxies"] = parse_proxy(cfg["proxy"])
    cfg["models_url"] = cfg["models_url"] or derive_models_url(cfg["base_url"])
    cfg["debug"] = str(cfg["debug"]).lower() in ("1", "true", "yes", "on") or \
        str(os.environ.get("DEBUG", "")).lower() in ("1", "true", "yes", "on")
    keys, pinned, source = resolve_server_keys(file_env)
    cfg["api_keys"], cfg["key_pinned"], cfg["key_source"] = keys, pinned, source
    cfg["key_pool"] = KeyPool(keys)
    return cfg


def debug(cfg, *parts):
    if cfg["debug"]:
        sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] " + " ".join(str(p) for p in parts) + "\n")
        sys.stderr.flush()


def upstream_headers(key):
    headers = dict(UPSTREAM_HEADERS)
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


# ---------------------------------------------------------------------------
# GET /v1/models — live from the official Provider API
# ---------------------------------------------------------------------------

_MODELS_CACHE = {"at": 0.0, "payload": None}
_MODELS_LOCK = threading.Lock()


def fetch_models(cfg):
    headers = {"Accept": "application/json"}
    key = pick_upstream_key(cfg)
    if key:
        headers["Authorization"] = f"Bearer {key}"
    resp = SESSION.get(cfg["models_url"], headers=headers,
                       timeout=(cfg["connect_timeout"], 30), proxies=cfg["proxies"])
    resp.raise_for_status()
    body = resp.json()
    items = body.get("data") if isinstance(body, dict) else body
    if not isinstance(items, list):
        raise ValueError(f"unexpected model list payload: {str(body)[:200]}")

    created = int(time.time())
    models = []
    for item in items:
        if not isinstance(item, dict) or not item.get("id"):
            continue
        entry = {
            "id": item["id"],
            "object": "model",
            "created": item.get("created") or created,
            "owned_by": item.get("owned_by") or "command-code",
        }
        for extra in ("name", "context_length", "supported_endpoints"):
            if item.get(extra) is not None:
                entry[extra] = item[extra]
        models.append(entry)
    return {"object": "list", "data": apply_model_filter(cfg, models)}


def apply_model_filter(cfg, models):
    """`.env models="a,b"` selects and orders the visible catalog. A filter that
    matches nothing is ignored rather than emptying the list under the client."""
    wanted = [w.strip() for w in (cfg["models"] or "").split(",") if w.strip()]
    if not wanted:
        return models
    by_id = {m["id"].lower(): m for m in models}
    chosen = [by_id[w.lower()] for w in wanted if w.lower() in by_id]
    if not chosen:
        debug(cfg, f"[models] filter matched none of {len(models)} models; showing all")
        return models
    return chosen


def get_models(cfg):
    """Cached model list. A failed refresh serves the previous copy if there is
    one; only a cold cache with a failing upstream surfaces an error."""
    ttl = cfg["models_ttl"]
    cached = _MODELS_CACHE["payload"]
    if cached is not None and time.time() - _MODELS_CACHE["at"] < ttl:
        return cached
    try:
        payload = fetch_models(cfg)
    except Exception as exc:
        if cached is not None:
            debug(cfg, f"[models] refresh failed ({exc}); serving cached list")
            return cached
        raise
    with _MODELS_LOCK:
        _MODELS_CACHE["at"] = time.time()
        _MODELS_CACHE["payload"] = payload
    return payload


# ---------------------------------------------------------------------------
# OpenAI request -> native request
# ---------------------------------------------------------------------------


def parts_to_text(parts):
    """Flatten an OpenAI content-parts array into plain text. The native
    endpoint is text-only, so images and files become inline placeholders
    instead of being dropped silently."""
    out = []
    for part in parts or []:
        if isinstance(part, str):
            out.append(part)
        elif isinstance(part, dict):
            ptype = part.get("type")
            if ptype in ("text", "input_text", "output_text"):
                out.append(part.get("text", ""))
            elif ptype == "refusal":
                out.append(part.get("refusal", ""))
            elif ptype in ("image_url", "input_image", "input_file"):
                out.append(f"[{ptype.removeprefix('input_')}]")
            elif "text" in part:
                out.append(str(part.get("text", "")))
    return "\n".join(t for t in out if t)


def tool_args(raw):
    return raw if isinstance(raw, str) else json.dumps(raw or {}, ensure_ascii=False)


def convert_messages(messages):
    """Split OpenAI messages into (native messages, system prompt).

    The native API accepts only user/assistant roles with string content, so
    system/developer messages are merged into `params.system`, tool messages
    become "[tool result for <id>]" user turns, and assistant tool calls are
    kept as an inline marker so the model sees its own prior actions.
    """
    native, system_parts = [], []
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        role, content = msg.get("role"), msg.get("content")

        if role in ("system", "developer"):
            system_parts.append(content if isinstance(content, str) else parts_to_text(content))
            continue
        if role == "tool":
            text = content if isinstance(content, str) else parts_to_text(content)
            native.append({
                "role": "user",
                "content": f"[tool result for {msg.get('tool_call_id') or '?'}]\n{text}",
            })
            continue
        if role not in ("user", "assistant"):
            continue

        if isinstance(content, list):
            text = parts_to_text(content)
        elif isinstance(content, str):
            text = content
        else:
            text = ""
        if role == "assistant" and msg.get("tool_calls"):
            markers = [
                f"[called tool {tc.get('function', {}).get('name', '?')}"
                f"({tool_args(tc.get('function', {}).get('arguments'))})]"
                for tc in msg["tool_calls"] if isinstance(tc, dict)
            ]
            text = "\n".join([text] + markers) if text else "\n".join(markers)
        native.append({"role": role, "content": text})

    system = "\n\n".join(p for p in system_parts if p)
    return native, system


def convert_tools(tools):
    """OpenAI tool defs -> the native {name, description, input_schema} shape.
    Built-in web_search / web_fetch tools already match and pass through."""
    out = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        ttype = tool.get("type")
        if ttype == "function":
            fn = tool.get("function") or {}
            out.append({
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
            })
        elif ttype in ("web_search_20250305", "web_fetch_20250910"):
            out.append(tool)
    return out


def convert_tool_choice(tool_choice):
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        return {"none": "none", "auto": "auto",
                "required": "required", "tool_calls": "required"}.get(tool_choice, tool_choice)
    if isinstance(tool_choice, dict):
        fn = tool_choice.get("function") or {}
        name = fn.get("name") if isinstance(fn, dict) else None
        if name:
            return {"type": "tool", "toolName": name}
    return None


def build_native_body(params, cfg):
    """Wrap converted params in the top-level command-code envelope."""
    if params.get("max_tokens") is not None:
        params["max_tokens"] = min(int(params["max_tokens"]), MAX_OUTPUT_TOKENS)
    return {
        "config": {
            "workingDir": cfg["working_dir"] or "/tmp",
            "date": time.strftime("%Y-%m-%d"),
            "environment": cfg["environment"] or "terminal",
            "structure": [],
            "isGitRepo": False,
            "currentBranch": "",
            "mainBranch": "",
            "gitStatus": "",
            "recentCommits": [],
        },
        "memory": cfg["memory"],
        "taste": cfg["taste"],
        "skills": cfg["skills"] or None,
        "permissionMode": cfg["permission_mode"] or "standard",
        "params": params,
    }


def convert_request(req, cfg):
    """Translate an OpenAI Chat Completions body into the native body."""
    messages, system = convert_messages(req.get("messages"))
    params = {
        "model": req.get("model") or cfg["default_model"],
        "messages": messages,
        "temperature": req.get("temperature") if req.get("temperature") is not None else 0.3,
        "stream": True,  # the native endpoint refuses anything else
    }
    if system:
        params["system"] = system
    tools = convert_tools(req.get("tools"))
    if tools:
        params["tools"] = tools
    tool_choice = convert_tool_choice(req.get("tool_choice"))
    if tool_choice is not None:
        params["toolChoice"] = tool_choice
    max_tokens = req.get("max_tokens") or req.get("max_completion_tokens")
    if max_tokens:
        params["max_tokens"] = max_tokens
    if req.get("top_p") is not None:
        params["top_p"] = req["top_p"]
    if req.get("reasoning_effort") is not None:
        params["reasoning_effort"] = req["reasoning_effort"]
    stop = req.get("stop")
    if stop:
        params["stop"] = stop if isinstance(stop, list) else [stop]
    return build_native_body(params, cfg)


# ---------------------------------------------------------------------------
# Native event stream -> OpenAI chunks
# ---------------------------------------------------------------------------


def iter_events(resp):
    """Yield one native event dict per line. The stream is newline-delimited
    JSON; `data:`/`event:`/comment framing is tolerated but not used."""
    for line in resp.iter_lines(decode_unicode=False):
        if not line:
            continue
        try:
            text = line.decode("utf-8").strip()
        except UnicodeDecodeError:
            continue
        if not text or text.startswith(":") or text.startswith("event:"):
            continue
        if text.startswith("data:"):
            text = text[5:].strip()
        if not text or text == "[DONE]":
            continue
        try:
            obj = json.loads(text)
        except ValueError:
            continue
        if isinstance(obj, dict):
            yield obj


def error_message(event):
    err = event.get("error")
    if isinstance(err, dict):
        return err.get("message") or json.dumps(err, ensure_ascii=False)
    if isinstance(err, str):
        return err
    return event.get("message") or "upstream stream error"


def map_finish_reason(reason):
    return {
        "tool-calls": "tool_calls",
        "tool-calls-paused": "tool_calls",
        "tool_calls": "tool_calls",
        "max-tokens": "length",
        "max-tokens-paused": "length",
        "max_output_tokens": "length",
        "length": "length",
    }.get(reason or "", "stop")


def openai_usage(total):
    prompt, completion = total.get("inputTokens") or 0, total.get("outputTokens") or 0
    usage = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total.get("totalTokens") or (prompt + completion),
    }
    cached = (total.get("inputTokenDetails") or {}).get("cacheReadTokens")
    if cached:
        usage["prompt_tokens_details"] = {"cached_tokens": cached}
    reasoning = (total.get("outputTokenDetails") or {}).get("reasoningTokens")
    if reasoning:
        usage["completion_tokens_details"] = {"reasoning_tokens": reasoning}
    return usage


class Translator:
    """Consumes native events and emits OpenAI chunks.

    Streaming callers get one chunk per event; buffered callers get nothing
    until `completion()`. Both paths share the accumulated state.

    Events seen in practice: start / start-step / provider-metadata (ignored),
    text-start|delta|end, reasoning-start|delta|end, tool-input-start|delta|end,
    tool-call (the complete form), finish-step (per-step usage, ignored),
    finish, error. Text and tool deltas are streamed as they arrive, so the
    trailing complete `tool-call` only fills in anything the deltas missed.
    """

    def __init__(self, model, streaming):
        self.model = model
        self.streaming = streaming
        self.id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        self.created = int(time.time())
        self.text = []
        self.reasoning = []
        self.tool_calls = []   # [{id, name, args}]
        self._by_id = {}       # upstream tool id -> index in tool_calls
        self.finish_reason = None
        self.usage = None
        self.finished = False
        self.error = None
        self._role_sent = False

    # -- chunk builders ----------------------------------------------------

    def _chunk(self, delta, finish_reason=None, usage=None):
        chunk = {
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        if usage is not None:
            chunk["usage"] = usage
        return chunk

    def _delta(self, delta):
        """Emit a delta chunk, attaching the assistant role to the first one."""
        if not self._role_sent:
            delta = {"role": "assistant", **delta}
            self._role_sent = True
        return self._chunk(delta)

    # -- event handling ----------------------------------------------------

    def feed(self, event):
        """Translate one native event into a list of OpenAI chunks to emit."""
        etype = event.get("type")

        if etype is None:
            # A bare error envelope can arrive in place of a stream.
            if event.get("error") or event.get("success") is False:
                self.error = error_message(event)
                self.finished = True
            return []

        if etype == "error":
            self.error = error_message(event)
            self.finished = True
            return []

        if etype == "text-delta":
            text = event.get("text") or ""
            self.text.append(text)
            return [self._delta({"content": text})] if self.streaming else []

        if etype == "reasoning-delta":
            text = event.get("text") or ""
            self.reasoning.append(text)
            return [self._delta({"reasoning_content": text})] if self.streaming else []

        if etype == "tool-input-start":
            index = len(self.tool_calls)
            call = {"id": event.get("id") or f"call_{index}",
                    "name": event.get("toolName") or "", "args": ""}
            self.tool_calls.append(call)
            self._by_id[event.get("id")] = index
            if not self.streaming:
                return []
            return [self._delta({"tool_calls": [{
                "index": index, "id": call["id"], "type": "function",
                "function": {"name": call["name"], "arguments": ""},
            }]})]

        if etype == "tool-input-delta":
            index = self._by_id.get(event.get("id"))
            delta = event.get("delta") or ""
            if index is not None:
                self.tool_calls[index]["args"] += delta
            if not self.streaming:
                return []
            return [self._chunk({"tool_calls": [{
                "index": index if index is not None else 0,
                "function": {"arguments": delta},
            }]})]

        if etype == "tool-call":
            return self._complete_tool_call(event)

        if etype == "finish":
            self.finish_reason = map_finish_reason(event.get("finishReason"))
            self.usage = openai_usage(event.get("totalUsage") or {})
            self.finished = True
            if not self.streaming:
                return []
            return [self._chunk({}, finish_reason=self.finish_reason, usage=self.usage)]

        return []  # start / start-step / text-start / text-end / tool-input-end / ...

    def _complete_tool_call(self, event):
        """The trailing complete `tool-call` event. Anything already streamed
        from tool-input-* wins; this only fills in gaps."""
        call_id = event.get("toolCallId") or event.get("id") or ""
        name = event.get("toolName") or ""
        raw = event.get("input", event.get("args", event.get("arguments")))
        args = tool_args(raw)
        index = self._by_id.get(call_id)
        if index is not None:
            call = self.tool_calls[index]
            call["name"] = call["name"] or name
            call["args"] = call["args"] or args
            return []
        index = len(self.tool_calls)
        self.tool_calls.append({"id": call_id or f"call_{index}", "name": name, "args": args})
        if call_id:
            self._by_id[call_id] = index
        if not self.streaming:
            return []
        return [self._delta({"tool_calls": [{
            "index": index, "id": self.tool_calls[index]["id"], "type": "function",
            "function": {"name": name, "arguments": args},
        }]})]

    def finalize(self):
        """Terminal chunk for a stream that ended without a `finish` event, so
        the client always sees a finish_reason."""
        if self.finished:
            return None
        self.finished = True
        self.finish_reason = self.finish_reason or "stop"
        if not self.streaming:
            return None
        return self._chunk({}, finish_reason=self.finish_reason, usage=self.usage)

    # -- buffered output ---------------------------------------------------

    def completion(self):
        message = {"role": "assistant", "content": "".join(self.text) or None}
        if self.reasoning:
            message["reasoning_content"] = "".join(self.reasoning)
        if self.tool_calls:
            message["tool_calls"] = [{
                "id": call["id"], "type": "function",
                "function": {"name": call["name"], "arguments": call["args"] or "{}"},
            } for call in self.tool_calls]
        return {
            "id": self.id,
            "object": "chat.completion",
            "created": self.created,
            "model": self.model,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": self.finish_reason or "stop",
            }],
            "usage": self.usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }


def upstream_error(resp):
    """Pull the human-readable message out of a failed upstream response."""
    try:
        body = resp.json()
    except ValueError:
        text = (resp.text or "").strip()
        return text[:500] or f"HTTP {resp.status_code}"
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict) and err.get("message"):
            return err["message"]
        if isinstance(err, str):
            return err
    return json.dumps(body, ensure_ascii=False)[:500]


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------


class BridgeHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "command-code-bridge/2.0"
    cfg = None  # set in main()

    # -- plumbing ----------------------------------------------------------

    def log_message(self, fmt, *args):
        debug(self.cfg, f"[http] {self.address_string()} {fmt % args}")

    def _read_body(self):
        length = self.headers.get("Content-Length")
        if length:
            return self.rfile.read(int(length))
        if (self.headers.get("Transfer-Encoding") or "").lower() == "chunked":
            return self._read_chunked()
        return b""

    def _read_chunked(self):
        chunks = []
        while True:
            line = self.rfile.readline().strip()
            if not line:
                break
            size = int(line.split(b";")[0], 16)
            if size == 0:
                self.rfile.readline()
                break
            chunks.append(self.rfile.read(size))
            self.rfile.read(2)  # trailing CRLF
        return b"".join(chunks)

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers",
                         "Authorization, Content-Type, x-api-key")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def _send_json(self, status, obj):
        payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self._cors()
        self.end_headers()
        self.wfile.write(payload)

    def _send_error(self, status, message, etype="invalid_request_error"):
        self._send_json(status, {"error": {"message": message, "type": etype}})

    # -- SSE ---------------------------------------------------------------

    def _sse_open(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Transfer-Encoding", "chunked")
        self._cors()
        self.end_headers()

    def _sse_frame(self, obj):
        self._sse_raw("data: " + json.dumps(obj, ensure_ascii=False) + "\n\n")

    def _sse_raw(self, text):
        data = text.encode("utf-8")
        self.wfile.write(b"%X\r\n" % len(data) + data + b"\r\n")
        self.wfile.flush()

    def _sse_end(self):
        try:
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except OSError:
            pass

    # -- routes ------------------------------------------------------------

    def _authorized(self):
        """Optional proxy-level gate: `.env api_key` requires the client to
        present that exact bearer token (or x-api-key)."""
        want = self.cfg["api_key"]
        if not want:
            return True
        got = (self.headers.get("x-api-key") or "").strip()
        if not got:
            auth = self.headers.get("Authorization") or ""
            if auth.lower().startswith("bearer "):
                got = auth[7:].strip()
        return hmac.compare_digest(got.encode("utf-8"), want.encode("utf-8"))

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        path = urllib.parse.urlsplit(self.path).path.rstrip("/")
        if path in ("/health", "/healthz"):
            self._send_json(200, {
                "status": "ok",
                "upstream": self.cfg["base_url"],
                "models_url": self.cfg["models_url"],
                "keys": len(self.cfg["api_keys"]),
                "proxy": self.cfg["proxy"] or None,
            })
            return
        if path == "/v1/models":
            self._handle_models()
            return
        self._send_error(404, f"Unknown path: {self.path}", "not_found_error")

    def do_POST(self):
        # Drain the body first: on an early 401/404 the request would otherwise
        # be left unread and the next request on this keep-alive connection
        # would be parsed as its body.
        raw = self._read_body()
        path = urllib.parse.urlsplit(self.path).path.rstrip("/")
        if path != "/v1/chat/completions":
            self._send_error(404, f"Unknown path: {self.path}", "not_found_error")
            return
        self._handle_chat(raw)

    # -- handlers ----------------------------------------------------------

    def _handle_models(self):
        if not self._authorized():
            self._send_error(401, "Missing or invalid proxy API key", "authentication_error")
            return
        try:
            payload = get_models(self.cfg)
        except Exception as exc:
            self._send_error(502, f"Failed to fetch model list: {exc}", "upstream_error")
            return
        self._send_json(200, payload)

    def _handle_chat(self, raw):
        if not self._authorized():
            self._send_error(401, "Missing or invalid proxy API key", "authentication_error")
            return

        try:
            req = json.loads(raw.decode("utf-8") or "{}")
        except (UnicodeDecodeError, ValueError) as exc:
            self._send_error(400, f"Invalid JSON body: {exc}")
            return
        if not isinstance(req, dict):
            self._send_error(400, "Request body must be a JSON object")
            return
        if not isinstance(req.get("messages"), list) or not req["messages"]:
            self._send_error(400, "Missing required field: messages")
            return

        streaming = req.get("stream") is True
        model = req.get("model") or self.cfg["default_model"]
        try:
            native = convert_request(req, self.cfg)
        except Exception as exc:
            self._send_error(400, f"Request conversion failed: {exc}")
            return

        debug(self.cfg, f"[chat] model={model} stream={streaming} "
                        f"messages={len(native['params']['messages'])} "
                        f"tools={len(native['params'].get('tools', []))}")

        client_key = client_key_from_headers(self.headers)
        last_error = None

        for attempt in range(MAX_ATTEMPTS):
            if attempt:
                time.sleep(0.5 * (2 ** (attempt - 1)))
            key = pick_upstream_key(self.cfg, client_key)
            debug(self.cfg, f"[chat] attempt {attempt + 1}/{MAX_ATTEMPTS} key={mask(key) or '(none)'}")
            try:
                resp = SESSION.post(
                    self.cfg["base_url"],
                    json=native,
                    headers=upstream_headers(key),
                    stream=True,
                    timeout=(self.cfg["connect_timeout"], self.cfg["read_timeout"]),
                    proxies=self.cfg["proxies"],
                )
            except requests.RequestException as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                debug(self.cfg, f"[chat] transport error: {last_error}")
                continue

            if resp.status_code == 200:
                if streaming:
                    self._stream(resp, model)
                else:
                    self._buffered(resp, model)
                return

            last_error = upstream_error(resp)
            resp.close()
            debug(self.cfg, f"[chat] upstream HTTP {resp.status_code}: {last_error}")
            if resp.status_code in RETRY_STATUS and attempt < MAX_ATTEMPTS - 1:
                continue
            self._send_error(resp.status_code, last_error, "upstream_error")
            return

        self._send_error(502, f"Upstream request failed after {MAX_ATTEMPTS} attempts: "
                              f"{last_error}", "upstream_error")

    def _stream(self, resp, model):
        translator = Translator(model, streaming=True)
        self._sse_open()
        try:
            for event in iter_events(resp):
                for chunk in translator.feed(event):
                    self._sse_frame(chunk)
                if translator.finished:
                    break
            final = translator.finalize()
            if final:
                self._sse_frame(final)
            if translator.error:
                self._sse_frame({"error": {"message": translator.error,
                                           "type": "upstream_error"}})
            self._sse_raw("data: [DONE]\n\n")
        except OSError:
            # Client hung up mid-stream; nothing left to say to it.
            debug(self.cfg, "[chat] client disconnected during stream")
        finally:
            resp.close()
            self._sse_end()

    def _buffered(self, resp, model):
        translator = Translator(model, streaming=False)
        try:
            for event in iter_events(resp):
                translator.feed(event)
                if translator.finished:
                    break
        finally:
            resp.close()
        if translator.error:
            self._send_error(502, translator.error, "upstream_error")
            return
        self._send_json(200, translator.completion())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    cfg = load_config()
    BridgeHandler.cfg = cfg

    if cfg["api_keys"]:
        print(f"[bridge] upstream keys: {len(cfg['api_keys'])} from {cfg['key_source']}"
              + (" (pinned)" if cfg["key_pinned"] else " (round-robin)"))
        for key in cfg["api_keys"]:
            print(f"[bridge]   {mask(key)}")
    else:
        print("[bridge] upstream keys: none — only client-supplied 'user_*' keys will work")
    print(f"[bridge] generate endpoint: {cfg['base_url']}")
    print(f"[bridge] models endpoint:   {cfg['models_url']}")
    print(f"[bridge] default model:     {cfg['default_model']}")
    if cfg["proxy"]:
        print(f"[bridge] outbound proxy:    {cfg['proxy']}")
    else:
        print("[bridge] outbound proxy:    none (uses HTTP_PROXY/HTTPS_PROXY if set)")
    if cfg["api_key"]:
        print("[bridge] client auth:       required (Authorization: Bearer <api_key>)")

    server = ThreadingHTTPServer((cfg["host"], cfg["port"]), BridgeHandler)
    server.daemon_threads = True
    print(f"[bridge] listening on http://{cfg['host']}:{cfg['port']}/v1  "
          f"(Ctrl+C to stop)\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[bridge] shutting down")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
