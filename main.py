#!/usr/bin/env python3
"""command-code bridge — an OpenAI-compatible /v1/chat/completions endpoint on
top of the command-code native /alpha/generate API.

    client (OpenAI JSON)  ──▶  this proxy  ──▶  /alpha/generate   (native NDJSON)
    GET /v1/models        ──▶  https://api.commandcode.ai/provider/v1/models

A Python port of c0mmandc0de2api (TypeScript), plus outbound HTTP proxy support
and a live model catalog. The native protocol has several quirks that shape
everything below; all of them were verified against the live endpoint:

  * Messages follow a role-discriminated schema — NOT the OpenAI shape. An
    assistant turn is `content: [text|reasoning|tool-call]` blocks, a tool
    result is a `role:"tool"` message with `content: [tool-result]` blocks, and
    a user turn must be a plain string (a tool-result block in a user message is
    rejected outright). The upstream normalizes all of this back into the OpenAI
    shape internally, so the round-trip preserves real tool structure.
  * The system prompt travels in its own `params.system` field.
  * Responses arrive as newline-delimited JSON events (not `data:` SSE frames),
    always with `stream: true` — a non-streaming request is refused upstream, so
    non-streaming callers get a buffered single-object response instead.

Single file. `requests` is used when available and the stdlib urllib shim below
takes over when it is not.   Run:  python main.py
"""

import hmac
import json
import os
import random
import re
import socket
import string
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:  # preferred: connection pooling and cleaner streaming
    import requests
except ImportError:  # noqa: S110 — the stdlib shim below covers everything we use
    requests = None

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
    # OAuth (used only when no key can be found anywhere)
    "auth_timeout": "15000",  # ms to wait for the browser callback, as in the CLI
    # native request envelope
    "working_dir": "",  # empty -> a random fake CLI-looking path (see below)
    "environment": "",  # empty -> a fake Node CLI fingerprint
    "memory": "",
    "taste": "",
    "skills": "",
    "permission_mode": "standard",
}

# The CLI's own headers. The version header selects the event vocabulary —
# 0.38.2 is the shape handled by Translator. The User-Agent is required, not
# cosmetic: Cloudflare answers the default `Python-urllib/3.x` with a 403
# "Error 1010: Access denied" (browser-signature block), so the stdlib transport
# must present something explicit. Any non-default UA passes.
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36")
CLI_VERSION = "0.38.2"
UPSTREAM_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "text/event-stream",
    "User-Agent": USER_AGENT,
    "x-command-code-version": CLI_VERSION,
    "x-cli-environment": "production",
    "x-taste-learning": "true",
    "x-co-flag": "false",
}

MAX_OUTPUT_TOKENS = 200_000
MAX_ATTEMPTS = 3
RETRY_DELAY = 0.2  # seconds; doubles per attempt, as in the CLI
RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
HARD_LIMIT_COOLDOWN = 300.0  # seconds to stop calling upstream once every key is limited


# ---------------------------------------------------------------------------
# Transport
#
# `requests` is preferred — its pooled connections and lazy byte iteration suit
# long SSE streams — but the proxy has to run on a bare interpreter too, so an
# ImportError falls back to a stdlib urllib shim exposing the small surface used
# here: a Session with get/post, a Response with status_code / iter_lines /
# json / text / close, and a RequestException to catch. Errors raised by both
# paths derive from RequestException, so callers never care which is active.
# ---------------------------------------------------------------------------


def iter_raw_lines(raw):
    """Yield one decoded line at a time, dropping the line terminator. A decode
    failure is skipped rather than aborting the stream, matching the lenient
    handling the requests path applies to a stray byte.

    http.client fully decodes chunked transfers before readline sees them, so
    this reader needs no transfer-encoding awareness."""
    reader = raw.readline
    while True:
        line = reader()
        if not line:
            return
        try:
            yield line.decode("utf-8").rstrip("\r\n")
        except UnicodeDecodeError:
            continue


if requests is not None:
    RequestException = requests.RequestException

    class Session:
        """requests.Session subclass exposing iter_lines as bytes, so the event
        reader is byte-oriented on both transports."""

        def __init__(self):
            self._session = requests.Session()

        def _with_bytes(self, resp):
            if not hasattr(resp, "_byte_iter_lines"):
                def iter_lines(**kwargs):
                    kwargs["decode_unicode"] = False
                    return requests.Response.iter_lines(resp, **kwargs)
                resp._byte_iter_lines = iter_lines
            return resp

        def get(self, url, **kwargs):
            return self._with_bytes(self._session.get(url, **kwargs))

        def post(self, url, **kwargs):
            return self._with_bytes(self._session.post(url, **kwargs))

else:
    # `post(json=...)` binds its parameter over the module, so keep our own
    # handle on the json module.
    _json = json

    class RequestException(Exception):
        pass

    class _Headers:
        """requests.headers.entries() used to forward upstream headers."""

        def __init__(self, message):
            self._message = message

        def get(self, name, default=None):
            return self._message.get(name, default)

        def entries(self):
            return list(self._message.items())

    class Response:
        def __init__(self, status_code, headers, raw, body, exception_cls):
            self.status_code = status_code
            self.headers = headers
            self._raw = raw
            self._body = body
            self._exc = exception_cls

        def iter_lines(self, decode_unicode=False):
            if self._raw is None:
                return
            lines = iter_raw_lines(self._raw)
            if decode_unicode:
                return lines
            return (line.encode("utf-8") for line in lines)

        def json(self):
            if self._body is None:
                self._body = self._raw.read()
                try:
                    self._raw.close()
                except Exception:
                    pass
            return json.loads(self._body.decode("utf-8"))

        @property
        def text(self):
            if self._body is None:
                self._body = self._raw.read()
                try:
                    self._raw.close()
                except Exception:
                    pass
            return self._body.decode("utf-8", "replace")

        def raise_for_status(self):
            if self.status_code >= 400:
                raise self._exc(f"HTTP {self.status_code}")

        def close(self):
            try:
                if self._raw is not None:
                    self._raw.close()
            except Exception:
                pass

    class Session:
        def get(self, url, headers=None, timeout=None, proxies=None):
            return self._request("GET", url, headers=headers, timeout=timeout,
                                 proxies=proxies)

        def post(self, url, json=None, headers=None, stream=False,
                 timeout=None, proxies=None):
            body = None
            headers = dict(headers or {})
            if json is not None:
                body = _json.dumps(json).encode("utf-8")
                headers.setdefault("Content-Type", "application/json")
            return self._request("POST", url,
                                 headers=headers, body=body, timeout=timeout,
                                 proxies=proxies, stream=stream)

        def _request(self, method, url, headers=None, body=None,
                     timeout=None, proxies=None, stream=True):
            opener = urllib.request.build_opener(
                urllib.request.ProxyHandler(proxies or {})
            ) if proxies else urllib.request.build_opener()
            request = urllib.request.Request(url, data=body, method=method,
                                             headers=headers or {})
            try:
                raw = opener.open(request, timeout=_read_timeout(timeout))
            except urllib.error.HTTPError as exc:
                # An HTTPError is still a readable response — the upstream's
                # error body is what callers want. http.client has already
                # decoded any chunked transfer, so the body is read directly.
                headers = _Headers(exc.headers)
                if stream:
                    return Response(exc.code, headers, exc, None, RequestException)
                return Response(exc.code, headers, None, exc.read(), RequestException)
            except (urllib.error.URLError, socket.timeout, OSError) as exc:
                raise RequestException(str(exc)) from exc
            headers = _Headers(raw.headers)
            if stream:
                return Response(raw.status, headers, raw, None, RequestException)
            return Response(raw.status, headers, None, raw.read(), RequestException)


def _read_timeout(timeout):
    """urllib wants a single timeout; use the (connect, read) tuple's read half
    when it was given, since what stalls a stream is the read."""
    if isinstance(timeout, (tuple, list)) and len(timeout) == 2:
        return timeout[1]
    return timeout


SESSION = Session()

# ---------------------------------------------------------------------------
# CLI disguise
#
# The upstream fingerprints clients. A fixed deployment path and a bare
# "terminal" environment string are not what a real CLI session looks like, so
# the working directory is randomized once per process (stable for the process
# lifetime — a real user keeps working in one directory — and different after a
# restart, so no cross-restart fingerprint forms).
# ---------------------------------------------------------------------------


def _random_lower(n):
    return "".join(random.choice(string.ascii_lowercase) for _ in range(n))


def generate_fake_working_dir():
    prefix = random.choice(("/Users/", "/home/"))
    subdir = random.choice(("projects", "dev", "code", "work", "src"))
    return (f"{prefix}{_random_lower(random.randint(5, 8))}/"
            f"{subdir}/{_random_lower(random.randint(4, 8))}")


def project_slug_from_path(path):
    """Mirrors the CLI's x-project-slug derivation: lowercase, drop the drive
    letter, non-alphanumerics to '-', trim dashes. The drive letter must go
    first — substituting first would turn "C:" into "c-" and keep the "c"."""
    slug = re.sub(r"^[a-z]:", "", path.lower())
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    return slug.strip("-") or "project"


FAKE_WORKING_DIR = generate_fake_working_dir()
FAKE_ENVIRONMENT = "linux-x64, Node.js v20.11.0"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


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


def parse_proxy(raw):
    """A single proxy URL applies to both schemes. Empty returns None so that
    requests falls back to the standard HTTP_PROXY / HTTPS_PROXY variables."""
    raw = (raw or "").strip()
    return {"http": raw, "https": raw} if raw else None


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
    cfg["auth_timeout"] = max(0, int(cfg["auth_timeout"])) / 1000.0
    cfg["connect_timeout"] = float(cfg["connect_timeout"])
    cfg["read_timeout"] = float(cfg["read_timeout"])
    cfg["proxies"] = parse_proxy(cfg["proxy"])
    cfg["models_url"] = cfg["models_url"] or derive_models_url(cfg["base_url"])
    cfg["debug"] = str(cfg["debug"]).lower() in ("1", "true", "yes", "on") or \
        str(os.environ.get("DEBUG", "")).lower() in ("1", "true", "yes", "on")
    cfg["working_dir"] = cfg["working_dir"] or FAKE_WORKING_DIR
    cfg["environment"] = cfg["environment"] or FAKE_ENVIRONMENT
    cfg["project_slug"] = project_slug_from_path(cfg["working_dir"])
    cfg["file_env"] = file_env
    refresh_keys(cfg)
    return cfg


def debug(cfg, *parts):
    if cfg["debug"]:
        sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] "
                         + " ".join(str(p) for p in parts) + "\n")
        sys.stderr.flush()


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


def mask(key):
    return key if len(key) <= 12 else f"{key[:8]}…{key[-4:]}"


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


def refresh_keys(cfg):
    keys, pinned, source = resolve_server_keys(cfg.get("file_env", {}))
    cfg["api_keys"], cfg["key_pinned"], cfg["key_source"] = keys, pinned, source
    cfg["key_pool"] = KeyPool(keys)
    return keys


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


def sanitize_error_text(cfg, text):
    """Upstream errors sometimes echo the request back. Never let a key reach
    the client in the clear."""
    for key in cfg["api_keys"]:
        if len(key) > 8 and key in text:
            text = text.replace(key, mask(key))
    return text


# ---------------------------------------------------------------------------
# OAuth browser login
#
# Ported from the CLI: a throwaway local server receives the key the Studio
# website POSTs back after the user authenticates, with a state token checked
# against CSRF. Falls back to a terminal paste if the browser round-trip fails.
# ---------------------------------------------------------------------------

STUDIO_BASE_URL = "https://commandcode.ai"
AUTH_PORT = 5959
AUTH_PORT_RANGE = 10


class AuthTimeout(Exception):
    pass


class CallbackHolder:
    def __init__(self):
        self.event = threading.Event()
        self.value = None
        self.error = None

    def resolve(self, value):
        self.value = value
        self.event.set()

    def reject(self, error):
        self.error = error
        self.event.set()

    def wait(self, timeout):
        if not self.event.wait(timeout):
            raise AuthTimeout()
        if self.error:
            raise self.error
        return self.value


class AuthCallbackHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "command-code-bridge-auth/1.0"
    holder = None

    def log_message(self, *args):
        pass

    def _cors(self):
        origin = self.headers.get("Origin") or ""
        allowed = ("http://localhost:3000", "https://staging.commandcode.ai",
                   "https://commandcode.ai")
        self.send_header("Access-Control-Allow-Origin",
                         origin if origin in allowed else allowed[0])
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        requested = self.headers.get("Access-Control-Request-Headers")
        self.send_header("Access-Control-Allow-Headers",
                         requested if requested else "Content-Type")
        # Chrome Private Network Access preflight
        self.send_header("Access-Control-Allow-Private-Network", "true")
        self.send_header("Content-Type", "application/json")

    def _reply(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self._cors()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))[:10_000]
        if urllib.parse.urlsplit(self.path).path != "/callback":
            self._reply(404, {"success": False, "error": "Not found"})
            return
        try:
            parsed = json.loads(raw.decode("utf-8") or "{}")
        except (UnicodeDecodeError, ValueError):
            self._reply(400, {"success": False, "error": "Invalid JSON"})
            return
        if not isinstance(parsed, dict):
            self._reply(400, {"success": False, "error": "Invalid JSON"})
            return

        if parsed.get("error"):
            description = parsed.get("error_description") or str(parsed["error"])
            self._reply(200, {"success": True})
            self.holder.reject(ValueError(description))
            return

        fields = {name: (parsed.get(name) if isinstance(parsed.get(name), str) else "")
                  for name in ("apiKey", "state", "userId", "userName", "keyName")}
        if not all(fields.values()):
            self._reply(400, {"success": False, "error": "Missing required fields"})
            return

        # Reply before resolving: the login flow closes the server as soon as
        # the callback lands, which would cut off an unwritten response.
        self._reply(200, {"success": True})
        self.holder.resolve(fields)


def _bind_auth_server(start_port=AUTH_PORT, port_range=AUTH_PORT_RANGE):
    """Prefer the CLI's usual port, then the next few, then any free one."""
    last = None
    for offset in range(port_range + 1):
        port = start_port + offset if offset < port_range else 0
        try:
            return ThreadingHTTPServer(("127.0.0.1", port), AuthCallbackHandler)
        except OSError as exc:
            last = exc
    raise OSError(f"could not bind an auth callback port: {last}")


def open_browser(url):
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", url])
        elif sys.platform.startswith("win"):
            os.startfile(url)  # noqa: S606 — Windows shell open
        else:
            subprocess.Popen(["xdg-open", url])
    except Exception:
        print(f"   无法自动打开浏览器，请手动访问：{url}")


def sanitize_api_key(raw):
    """Strip bracketed-paste markers and control characters left by a terminal
    paste."""
    text = str(raw)
    for marker in ("\x1b[200~", "\x1b[201~", "[200~", "[201~"):
        text = text.replace(marker, "")
    return "".join(ch for ch in text if ord(ch) > 31 and ord(ch) != 127).strip()


def save_api_key(api_key):
    directory = os.path.expanduser("~/.commandcode")
    try:
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, "auth.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"apiKey": api_key}, fh, indent=2)
        print(f"   API Key 已保存到 {path}")
    except OSError as exc:
        print(f"   [warn] 无法保存 API Key 到 {directory}: {exc}")


def prompt_for_api_key(message):
    """Interactive paste. A pasted auth.json blob is unwrapped to its key."""
    try:
        answer = input(f"{message}\n> ")
    except (EOFError, KeyboardInterrupt):
        return ""
    try:
        parsed = json.loads(answer.strip())
        if isinstance(parsed, dict):
            if isinstance(parsed.get("apiKey"), str):
                answer = parsed["apiKey"]
            elif isinstance(parsed.get("command-code"), dict):
                answer = parsed["command-code"].get("key", answer)
            elif isinstance(parsed.get("key"), str):
                answer = parsed["key"]
        elif isinstance(parsed, str):
            answer = parsed
    except ValueError:
        pass
    return sanitize_api_key(answer)


def login(cfg):
    """Full OAuth flow; returns the API key and saves it to auth.json."""
    print("\n   开始 CommandCode OAuth 登录...")

    holder = CallbackHolder()
    AuthCallbackHandler.holder = holder
    try:
        server = _bind_auth_server()
    except OSError:
        print("   无法启动本地认证服务器，请手动粘贴 API Key。")
        key = prompt_for_api_key("请粘贴你的 CommandCode API Key")
        if not key:
            raise RuntimeError("未提供 CommandCode API Key")
        save_api_key(key)
        return key

    state = uuid.uuid4().hex + uuid.uuid4().hex
    callback_url = f"http://localhost:{server.server_address[1]}/callback"
    auth_url = (f"{STUDIO_BASE_URL}/studio/auth/cli"
                f"?callback={urllib.parse.quote(callback_url, safe='')}"
                f"&state={urllib.parse.quote(state, safe='')}")

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print("   正在打开浏览器...")
    open_browser(auth_url)

    timed_out = False
    try:
        callback = holder.wait(cfg["auth_timeout"])
    except AuthTimeout:
        callback, timed_out = None, True
    finally:
        server.shutdown()
        server.server_close()

    if timed_out:
        print("   自动回传超时，请手动粘贴 API Key。")
        key = prompt_for_api_key("自动回传失败。请在浏览器中复制 API Key 并粘贴到此处")
        if not key:
            raise RuntimeError("未提供 CommandCode API Key")
        save_api_key(key)
        return key

    if callback["state"] != state:
        raise RuntimeError("State token 不匹配，认证可能被篡改")

    print(f"   认证成功！用户: {callback['userName']} ({callback['keyName']})")
    save_api_key(callback["apiKey"])
    return callback["apiKey"]


def ensure_api_keys(cfg):
    """Load keys, falling back to the OAuth flow when there are none."""
    keys = refresh_keys(cfg)
    if keys:
        return keys
    try:
        key = login(cfg)
    except Exception as exc:
        print(f"❌ OAuth 登录失败: {exc}", file=sys.stderr)
        print("   请手动配置 API Key：", file=sys.stderr)
        print("   方式1: 设置环境变量 COMMANDCODE_API_KEY=user_...", file=sys.stderr)
        print("   方式2: 设置环境变量 COMMANDCODE_API_KEYS=key1,key2,...", file=sys.stderr)
        print('   方式3: 创建 ~/.commandcode/auth.json 包含 {"apiKey":"user_..."}',
              file=sys.stderr)
        raise SystemExit(1)
    if not refresh_keys(cfg):
        cfg["api_keys"] = [key]
        cfg["key_pool"] = KeyPool([key])
    return cfg["api_keys"]


# ---------------------------------------------------------------------------
# Hard-limit cooldown
#
# Once every key in the pool has come back hard-limited, stop hammering the
# upstream and answer 429 locally until the window passes.
# ---------------------------------------------------------------------------

_hard_limit_until = 0.0
_hard_limit_lock = threading.Lock()


def in_hard_limit_cooldown():
    return time.time() < _hard_limit_until


def enter_hard_limit_cooldown():
    global _hard_limit_until
    with _hard_limit_lock:
        _hard_limit_until = time.time() + HARD_LIMIT_COOLDOWN


def cooldown_remaining():
    return max(0.0, _hard_limit_until - time.time())


def is_rate_limit_or_quota(text):
    """Whether an upstream error should send the retry to a different key."""
    return bool(
        re.search(r"\b429\b", text)
        or re.search(r"RATE.?LIMITED", text, re.I)
        or re.search(r"MODEL.?NOT.?IN.?PLAN", text, re.I)
        or re.search(r"usage\s*limit", text, re.I)
        or re.search(r"quota", text, re.I)
        or (re.search(r"\b403\b", text) and re.search(r"billing|plan|quota", text, re.I))
    )


# ---------------------------------------------------------------------------
# GET /v1/models — live from the official Provider API
# ---------------------------------------------------------------------------

_MODELS_CACHE = {"at": 0.0, "payload": None}
_MODELS_LOCK = threading.Lock()


def fetch_models(cfg):
    headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
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
    """Flatten an OpenAI content-parts array into plain text. A user turn must
    be a string upstream, so images and files become inline placeholders rather
    than being dropped silently."""
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


def parse_tool_arguments(raw):
    """Tool arguments travel as a JSON string; the native shape wants the parsed
    object, so an unparseable string is passed through as-is."""
    if isinstance(raw, (dict, list)):
        return raw
    if not isinstance(raw, str):
        return {}
    try:
        return json.loads(raw)
    except ValueError:
        return raw


def convert_messages(messages):
    """Split OpenAI messages into (native messages, system prompt).

    The native schema is role-discriminated, so each turn is rebuilt in the
    shape that role accepts:

        user       -> string content
        assistant  -> [text | reasoning | tool-call] blocks
        tool       -> [tool-result] blocks, carrying tool_call_id

    Tool structure is preserved end to end — the upstream normalizes these
    blocks into the standard OpenAI `tool_calls` / `tool_call_id` fields itself,
    so an agentic client gets a real tool round-trip rather than flattened text.
    """
    native, system_parts = [], []
    # tool_call_id -> tool name, so a tool result can name the call it answers
    tool_names = {}
    for msg in messages or []:
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            for call in msg.get("tool_calls") or []:
                if isinstance(call, dict) and call.get("id"):
                    fn = call.get("function") or {}
                    tool_names[call["id"]] = fn.get("name", "") if isinstance(fn, dict) else ""

    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        role, content = msg.get("role"), msg.get("content")

        if role in ("system", "developer"):
            system_parts.append(content if isinstance(content, str) else parts_to_text(content))
            continue

        if role == "tool":
            text = content if isinstance(content, str) else parts_to_text(content)
            call_id = msg.get("tool_call_id") or ""
            native.append({"role": "tool", "content": [{
                "type": "tool-result",
                "toolCallId": call_id,
                "toolName": tool_names.get(call_id, ""),
                "output": {"type": "text", "value": text},
            }]})
            continue

        if role == "assistant":
            blocks = []
            if isinstance(content, str) and content:
                blocks.append({"type": "text", "text": content})
            elif isinstance(content, list):
                text = parts_to_text(content)
                if text:
                    blocks.append({"type": "text", "text": text})
            if msg.get("reasoning_content"):
                blocks.append({"type": "reasoning", "text": msg["reasoning_content"]})
            for call in msg.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                fn = call.get("function") or {}
                blocks.append({
                    "type": "tool-call",
                    "toolCallId": call.get("id") or "",
                    "toolName": fn.get("name", "") if isinstance(fn, dict) else "",
                    "input": parse_tool_arguments(fn.get("arguments") if isinstance(fn, dict) else None),
                })
            native.append({"role": "assistant",
                           "content": blocks or [{"type": "text", "text": ""}]})
            continue

        if role == "user":
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text = parts_to_text(content)
            else:
                text = ""
            native.append({"role": "user", "content": text})

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
            "workingDir": cfg["working_dir"],
            "date": time.strftime("%Y-%m-%d"),
            "environment": cfg["environment"],
            "structure": [],
            "isGitRepo": False,
            "currentBranch": "",
            "mainBranch": "",
            "gitStatus": "",
            "recentCommits": [],
        },
        "memory": cfg["memory"] or None,
        "taste": cfg["taste"] or None,
        "skills": cfg["skills"] or None,
        "permissionMode": cfg["permission_mode"] or "standard",
        "params": params,
        # Reused across retries so a resumed generation keeps its context.
        "threadId": str(uuid.uuid4()),
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
    until `completion()`. Both paths share the accumulated state, which is what
    lets a resumed stream continue the same assistant message.

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
        args = raw if isinstance(raw, str) else json.dumps(raw if raw is not None else {},
                                                          ensure_ascii=False)
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

        # Every key is hard-limited: answer locally instead of hammering upstream.
        if in_hard_limit_cooldown():
            retry_after = int(cooldown_remaining()) + 1
            self.send_response(429)
            self.send_header("Retry-After", str(retry_after))
            self.send_header("Content-Type", "application/json; charset=utf-8")
            payload = json.dumps({"error": {
                "message": f"上游限流冷却中，{retry_after}s 后重试",
                "type": "rate_limit_error",
            }}, ensure_ascii=False).encode("utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self._cors()
            self.end_headers()
            self.wfile.write(payload)
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
                        f"tools={len(native['params'].get('tools', []))} "
                        f"threadId={native['threadId']}")

        client_key = client_key_from_headers(self.headers)
        if streaming:
            self._stream(native, model, client_key)
        else:
            self._buffered(native, model, client_key)

    # -- upstream plumbing -------------------------------------------------

    def _post(self, native, key):
        headers = dict(UPSTREAM_HEADERS)
        headers["x-project-slug"] = self.cfg["project_slug"]
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return SESSION.post(
            self.cfg["base_url"],
            json=native,
            headers=headers,
            stream=True,
            timeout=(self.cfg["connect_timeout"], self.cfg["read_timeout"]),
            proxies=self.cfg["proxies"],
        )

    def _handle_upstream_failure(self, cfg, status, detail, attempt, can_retry):
        """Decide whether to rotate keys and retry. Returns True to continue."""
        if is_rate_limit_or_quota(detail) and attempt >= len(cfg["key_pool"]) and \
                re.search(r"Your limit resets at", detail):
            enter_hard_limit_cooldown()
            print(f"[bridge] 所有 {len(cfg['key_pool'])} 个 key 均触发硬限流，"
                  f"进入 {int(HARD_LIMIT_COOLDOWN)}s 冷却", file=sys.stderr)
            return False
        if status in RETRY_STATUS and can_retry:
            debug(cfg, f"[chat] HTTP {status}，换 key 重试")
            return True
        return False

    # -- streaming ---------------------------------------------------------

    def _stream(self, native, model, client_key):
        cfg = self.cfg
        translator = Translator(model, streaming=True)
        opened = False
        attempt = 0
        last_error = None

        try:
            while attempt < MAX_ATTEMPTS:
                attempt += 1
                key = pick_upstream_key(cfg, client_key)
                debug(cfg, f"[chat] attempt {attempt}/{MAX_ATTEMPTS} key={mask(key) or '(none)'}")
                try:
                    resp = self._post(native, key)
                except RequestException as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                    debug(cfg, f"[chat] transport error: {last_error}")
                    if not opened and attempt >= MAX_ATTEMPTS:
                        self._send_error(504 if "Timeout" in last_error else 502,
                                         f"上游请求失败（{MAX_ATTEMPTS} 次尝试）：{last_error}",
                                         "upstream_error")
                        return
                    time.sleep(RETRY_DELAY * (2 ** (attempt - 1)))
                    continue

                if resp.status_code != 200:
                    status = resp.status_code
                    detail = sanitize_error_text(cfg, upstream_error(resp))
                    resp.close()
                    last_error = detail
                    debug(cfg, f"[chat] upstream HTTP {status}: {detail}")
                    if self._handle_upstream_failure(cfg, status, detail, attempt,
                                                     attempt < MAX_ATTEMPTS):
                        time.sleep(RETRY_DELAY * (2 ** (attempt - 1)))
                        continue
                    if not opened:
                        self._send_error(status, detail, "upstream_error")
                        return
                    # Headers already sent — an SSE frame is the only channel left.
                    translator.error = detail
                    break

                if not opened:
                    self._sse_open()
                    opened = True

                try:
                    # A broken upstream raises a RequestException here; a broken
                    # *client* raises OSError from _sse_frame and must not be
                    # mistaken for one, so only request errors are caught. The
                    # client case unwinds to the handler below.
                    for event in iter_events(resp):
                        chunks = translator.feed(event)
                        finished = translator.finished
                        for chunk in chunks:
                            self._sse_frame(chunk)
                        if finished:
                            break
                except RequestException as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                    debug(cfg, "[chat] 流中断，尝试用同一 threadId 续写")
                finally:
                    resp.close()

                if translator.finished:
                    break
                if attempt >= MAX_ATTEMPTS:
                    break
                time.sleep(RETRY_DELAY * (2 ** (attempt - 1)))

            if not opened:
                self._send_error(502, f"上游请求失败：{last_error}", "upstream_error")
                return

            final = translator.finalize()
            if final:
                self._sse_frame(final)
            if translator.error:
                self._sse_frame({"error": {"message": sanitize_error_text(cfg, translator.error),
                                           "type": "upstream_error"}})
            self._sse_raw("data: [DONE]\n\n")
        except OSError:
            # The client hung up. Every upstream response is already closed by
            # the loop's finally, so there is nothing left to release.
            debug(cfg, "[chat] 客户端已断开")
        finally:
            if opened:
                self._sse_end()

    # -- buffered ----------------------------------------------------------

    def _buffered(self, native, model, client_key):
        cfg = self.cfg
        translator = Translator(model, streaming=False)
        attempt = 0
        last_error = None
        status = None

        while attempt < MAX_ATTEMPTS:
            attempt += 1
            key = pick_upstream_key(cfg, client_key)
            try:
                resp = self._post(native, key)
            except RequestException as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                debug(cfg, f"[chat] transport error: {last_error}")
                if attempt >= MAX_ATTEMPTS:
                    self._send_error(504 if "Timeout" in last_error else 502,
                                     f"上游请求失败（{MAX_ATTEMPTS} 次尝试）：{last_error}",
                                     "upstream_error")
                    return
                time.sleep(RETRY_DELAY * (2 ** (attempt - 1)))
                continue

            status = resp.status_code
            if status != 200:
                last_error = sanitize_error_text(cfg, upstream_error(resp))
                resp.close()
                debug(cfg, f"[chat] upstream HTTP {status}: {last_error}")
                if self._handle_upstream_failure(cfg, status, last_error, attempt,
                                                 attempt < MAX_ATTEMPTS):
                    time.sleep(RETRY_DELAY * (2 ** (attempt - 1)))
                    continue
                self._send_error(status, last_error, "upstream_error")
                return

            try:
                for event in iter_events(resp):
                    translator.feed(event)
                    if translator.finished:
                        break
            except RequestException as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                debug(cfg, "[chat] 流中断，尝试用同一 threadId 续写")
            finally:
                resp.close()

            if translator.finished:
                break
            if attempt >= MAX_ATTEMPTS:
                break
            time.sleep(RETRY_DELAY * (2 ** (attempt - 1)))

        if translator.error:
            self._send_error(502, sanitize_error_text(cfg, translator.error), "upstream_error")
            return
        self._send_json(200, translator.completion())


class BridgeServer(ThreadingHTTPServer):
    """A keep-alive handler blocks reading the next request line after each
    response, so a client that goes away mid-connection surfaces as a
    ConnectionResetError from socketserver — which would print a full traceback
    for every aborted connection. Those are expected; everything else still
    reports normally."""

    daemon_threads = True

    def handle_error(self, request, client_address):
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionResetError, ConnectionAbortedError,
                            BrokenPipeError, TimeoutError)):
            if BridgeHandler.cfg:
                debug(BridgeHandler.cfg, f"[http] {client_address[0]} 连接已断开")
            return
        super().handle_error(request, client_address)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    cfg = load_config()
    BridgeHandler.cfg = cfg

    keys = ensure_api_keys(cfg)

    if keys:
        print(f"[bridge] upstream keys: {len(keys)} from {cfg['key_source']}"
              + (" (pinned)" if cfg["key_pinned"] else " (round-robin)"))
        for key in keys:
            print(f"[bridge]   {mask(key)}")
    else:
        print("[bridge] upstream keys: none — only client-supplied 'user_*' keys will work")
    print(f"[bridge] generate endpoint: {cfg['base_url']}")
    print(f"[bridge] http transport:    "
          + ("requests" if requests is not None else "urllib (stdlib fallback)"))
    print(f"[bridge] models endpoint:   {cfg['models_url']}")
    print(f"[bridge] default model:     {cfg['default_model']}")
    print(f"[bridge] cli disguise:      workingDir={cfg['working_dir']} "
          f"slug={cfg['project_slug']}")
    if cfg["proxy"]:
        print(f"[bridge] outbound proxy:    {cfg['proxy']}")
    else:
        print("[bridge] outbound proxy:    none (uses HTTP_PROXY/HTTPS_PROXY if set)")
    if cfg["api_key"]:
        print("[bridge] client auth:       required (Authorization: Bearer <api_key>)")

    server = BridgeServer((cfg["host"], cfg["port"]), BridgeHandler)
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
