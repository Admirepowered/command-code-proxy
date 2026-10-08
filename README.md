# command-code OpenAI bridge

A Python port of [c0mmandc0de2api](https://github.com/) (TypeScript): a local proxy that
exposes an **OpenAI-compatible Chat Completions API** and forwards requests to the
command-code native `/alpha/generate` endpoint, translating requests and responses both
ways. `GET /v1/models` is served live from the official **Command Code Provider API**.

```
your app ── OpenAI /v1/chat/completions ──┐
                                          ├─▶ main.py ── /alpha/generate ──────▶ api.commandcode.ai
             OpenAI /v1/models     ───────┘           └─ /provider/v1/models ──▶ api.commandcode.ai
```

Everything lives in a single `main.py` — no model catalog to keep in sync, no split modules.
Beyond the port it adds **outbound HTTP proxy support** (the CLI has none, and on some
networks a direct connection to `api.commandcode.ai` fails at the TLS handshake).

## Requirements

- Python 3.10+ (tested on 3.14)
- `requests` — **optional.** If it is missing, the proxy falls back to a stdlib
  `urllib` transport automatically; `pip install requests` only buys connection pooling.

## Configuration

Edit `.env` (same directory as `main.py`). Any key can also be supplied as the
upper-cased environment variable — `PORT=9000 python main.py`.

| key | default | description |
|---|---|---|
| `base_url` | `https://api.commandcode.ai/alpha/generate` | generation endpoint |
| `models_url` | derived | model list source; empty derives `<origin>/provider/v1/models` |
| `auth_token` | — | upstream key, **pinned** (see below) |
| `proxy` | — | outbound HTTP(S) proxy, e.g. `http://127.0.0.1:7890` |
| `connect_timeout` | `15` | connect timeout, seconds |
| `read_timeout` | `600` | streaming read timeout, seconds |
| `auth_timeout` | `15000` | ms to wait for the OAuth browser callback (CLI default) |
| `host` / `port` | `0.0.0.0` / `8080` | listen address |
| `api_key` | — | optional: require this bearer token from clients |
| `debug` | — | `true` logs upstream calls and the key in use (masked) |
| `default_model` | `deepseek/deepseek-v4-flash` | used when the client omits `model` |
| `models` | — | optional comma-separated filter/order for `/v1/models` |
| `models_ttl` | `300` | seconds the upstream model list is cached |
| `working_dir` | *random* | empty uses a random fake CLI path — see **CLI disguise** |
| `environment` | *random* | empty uses a fake Node CLI fingerprint |
| `memory`, `taste`, `skills`, `permission_mode` | — | fields of the native request envelope |

### API key lookup

The upstream key is resolved once at startup, stopping at the first source that yields one:

| # | source | notes |
|---|---|---|
| 1 | `.env` `auth_token` | **Pinned** — used for every request; client keys ignored |
| 2 | `COMMANDCODE_API_KEY` | env var, or written into `.env` |
| 3 | `COMMANDCODE_API_KEYS` | comma/newline-separated **pool**, used round-robin |
| 4 | `~/.commandcode/auth.json` | |
| 5 | `~/.pi/agent/auth.json` | pi-compatible |
| 6 | `~/.omp/agent/auth.json` | OMP-compatible |
| 7 | **OAuth browser login** | only when nothing above matched — see below |
| 8 | client-supplied key | only when nothing above matched — see below |

`auth.json` accepts `{"apiKey": "user_..."}`, `{"commandcode": "user_..."}`, or
`{"command-code": {"type": "api", "key": "user_..."}}`.

**Client-key passthrough** lets each caller bring its own upstream key. It reads
`x-api-key` (Anthropic-style clients) or `Authorization: Bearer` (OpenAI clients) and
forwards the key upstream — but **only if it starts with `user_`**. That guard matters:
Claude Code and the OpenAI SDKs refuse to start without *some* token set, so a placeholder
like `sk-none` would otherwise be forwarded and turn a perfectly good server key into a 401.
When `.env auth_token` is set it wins over everything.

### OAuth browser login

When no key is found anywhere, the proxy runs the same login the CLI does: it starts a
throwaway HTTP server on `127.0.0.1:5959` (the next 9 ports if taken, then any free one),
opens `commandcode.ai/studio/auth/cli` in the browser, and waits `auth_timeout` ms for the
Studio site to POST the key back to `/callback`. A `state` token is checked against CSRF.
The key is saved to `~/.commandcode/auth.json`. If the callback times out or the local
server cannot bind, it falls back to a terminal paste (a pasted `auth.json` blob is
unwrapped to its key, and bracketed-paste escape codes are stripped).

The CLI's 15-second default is short for a browser round-trip — raise `auth_timeout` if you
find yourself landing in the paste fallback.

### CLI disguise

The upstream fingerprints its clients, so a fixed deployment path and a bare `"terminal"`
environment string are not what a real CLI session looks like. Following the CLI, each
process:

- generates a **random working directory** once at startup (`/Users/kqbmx/dev/wplt`-style).
  Stable for the process lifetime — a real user keeps working in one directory — and
  different after a restart, so no cross-restart fingerprint forms;
- derives `x-project-slug` from it the way the CLI does
  (`/Users/alice/Code` → `users-alice-code`);
- reports `environment` as `linux-x64, Node.js v20.11.0`;
- sends `x-taste-learning` and `x-co-flag` alongside the version headers.

**Setting `working_dir` or `environment` in `.env` overrides the disguise.** Leave them
empty to keep it.

### Retries, key rotation and the hard-limit cooldown

A chat request is attempted up to 3 times with backoff, taking the next key from the pool
each attempt. Transport errors and 408/409/425/429/5xx are retried; 4xx like
`MODEL_NOT_IN_PLAN` rotate the key too, since another key may be on a different plan.

If a stream breaks mid-response, the retry re-POSTs with the **same `threadId`**, so the
upstream resumes the generation and the client keeps receiving deltas into the same
assistant message.

Once *every* key in the pool has come back hard-limited (a 429 carrying
`Your limit resets at`), the proxy enters a **5-minute cooldown** and answers 429 locally
with a `Retry-After` header instead of hammering the upstream.

Upstream error text is passed back only after any key appearing in it is masked.

### Model list

`GET /v1/models` is fetched live from the official Provider API
(`https://api.commandcode.ai/provider/v1/models`) and cached for `models_ttl` seconds, so
new upstream models appear without a code change. It passes through `name`,
`context_length` and `supported_endpoints` alongside the usual OpenAI fields. A failed
refresh serves the previous copy; only a cold cache with a failing upstream returns 502.

Optionally, `.env models="a,b,c"` narrows the list to those ids and orders it that way.
The upstream catalog is not plan-filtered, so this is the way to hide models your plan
cannot call (on a Go plan, e.g., `gpt-5.4` returns 403 `MODEL_NOT_IN_PLAN`).

## Run

```bash
python main.py
```

Point your client at `http://localhost:3000/v1` (any API key, unless `api_key` is set).

## Endpoints

| endpoint | method | notes |
|---|---|---|
| `/v1/chat/completions` | POST | streaming and non-streaming |
| `/v1/models` | GET | live from the Provider API |
| `/health` | GET | shows the resolved endpoints, key count and proxy |

## Transports

`requests` is used when importable; otherwise a stdlib `urllib` shim (same `Session` /
`Response` surface, its own `RequestException`) takes over, so the proxy runs on a bare
interpreter with nothing installed. The startup banner reports which one is active.

Both transports send an explicit `User-Agent`. That is load-bearing: Cloudflare answers the
default `Python-urllib/3.x` agent with a **403 `Error 1010: Access denied`** (browser-signature
block), which looks exactly like an auth failure but is really the CDN refusing the client.
`requests`' default agent happens to pass, which is why the gap only shows up on the stdlib
path.

## Quick check

```bash
# streaming
curl -N http://localhost:3000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek/deepseek-v4-flash","messages":[{"role":"user","content":"hello"}],"stream":true}'

# non-streaming
curl http://localhost:3000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek/deepseek-v4-flash","messages":[{"role":"user","content":"hello"}]}'

# tool call
curl -N http://localhost:3000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek/deepseek-v4-flash","stream":true,
       "messages":[{"role":"user","content":"Weather in Paris? Use the tool."}],
       "tools":[{"type":"function","function":{"name":"get_weather","description":"Get weather",
                 "parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}]}'
```

## What the native endpoint actually requires

Verified against the live API — these constraints drive the whole translation layer:

- **Messages are role-discriminated, not the OpenAI shape.** Each role accepts a different
  content type, and the upstream normalizes all of it into standard OpenAI `tool_calls` /
  `tool_call_id` internally:

  | role | accepted content |
  |---|---|
  | `system` | goes in `params.system` instead |
  | `user` | a plain string (a `tool-result` block here is rejected outright) |
  | `assistant` | `[text \| reasoning \| tool-call]` blocks |
  | `tool` | `[tool-result]` blocks; must follow an assistant `tool_calls` turn |

  Tool structure therefore survives the round trip intact — an agentic client gets a real
  tool conversation, not flattened text. (Sending the OpenAI shape directly is rejected:
  `expected one of "user"|"assistant" at "params.messages[2].role"`.)

- **`stream: true` is mandatory.** The endpoint refuses non-streaming calls, so
  non-streaming clients get the stream consumed and buffered into one response object.
- **The response is newline-delimited JSON**, not `data:`-framed SSE. Event vocabulary
  (selected by the `x-command-code-version` header, currently `0.38.2`):

  | event | handling |
  |---|---|
  | `start`, `start-step`, `provider-metadata`, `text-start`, `text-end`, `tool-input-end` | ignored |
  | `text-delta` | → `delta.content` |
  | `reasoning-delta` | → `delta.reasoning_content` |
  | `tool-input-start` / `tool-input-delta` | → `delta.tool_calls[]` streamed as it arrives |
  | `tool-call` | complete form; fills gaps the deltas left |
  | `finish-step` | per-step usage, ignored |
  | `finish` | `finish_reason` + `usage`, ends the stream |
  | `error` | surfaced as an OpenAI error |

- `finishReason: "tool-calls"` maps to OpenAI's `tool_calls`; `max-tokens` maps to `length`.
- `usage` carries `prompt_tokens_details.cached_tokens` and
  `completion_tokens_details.reasoning_tokens` when the upstream reports them.

## ⚠️ Risk disclosure

The `/alpha/generate` endpoint is intended for the command-code **CLI** only. commandcode.ai
actively detects proxying and warns that *"continued proxying of your subscription violates
the TOS and will result in account ban."* Using this bridge may get your account banned.

A legitimate alternative exists: the official **Command Code Provider API**
(`https://api.commandcode.ai/provider/v1`, OpenAI- and Anthropic-compatible, same key), which
requires a plan with API access (Provider/GOAT+; the Go plan returns 403
`upgrade_required`). If your plan supports it, point your apps there directly — this proxy
only reads `/provider/v1/models` from it, precisely because that endpoint works on every plan.
