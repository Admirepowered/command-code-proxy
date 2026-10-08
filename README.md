# command-code OpenAI bridge

A local proxy that exposes an **OpenAI-compatible Chat Completions API** and forwards
requests to the command-code native `/alpha/generate` endpoint, translating requests and
responses both ways. `GET /v1/models` is served live from the official
**Command Code Provider API**.

```
your app ── OpenAI /v1/chat/completions ──┐
                                          ├─▶ main.py ── /alpha/generate ──────▶ api.commandcode.ai
             OpenAI /v1/models     ───────┘           └─ /provider/v1/models ──▶ api.commandcode.ai
```

Everything lives in a single `main.py` — no model catalog to keep in sync, no split modules.

## Requirements

- Python 3.10+ (tested on 3.14)
- `requests`

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
| `host` / `port` | `0.0.0.0` / `8080` | listen address |
| `api_key` | — | optional: require this bearer token from clients |
| `debug` | — | `true` logs upstream calls and the key in use (masked) |
| `default_model` | `deepseek/deepseek-v4-flash` | used when the client omits `model` |
| `models` | — | optional comma-separated filter/order for `/v1/models` |
| `models_ttl` | `300` | seconds the upstream model list is cached |
| `working_dir`, `environment`, `memory`, `taste`, `skills`, `permission_mode` | — | fields of the native request envelope |

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
| 7 | client-supplied key | only when nothing above matched — see below |

`auth.json` accepts `{"apiKey": "user_..."}`, `{"commandcode": "user_..."}`, or
`{"command-code": {"type": "api", "key": "user_..."}}`.

**Client-key passthrough** lets each caller bring its own upstream key. It reads
`x-api-key` (Anthropic-style clients) or `Authorization: Bearer` (OpenAI clients) and
forwards the key upstream — but **only if it starts with `user_`**. That guard matters:
Claude Code and the OpenAI SDKs refuse to start without *some* token set, so a placeholder
like `sk-none` would otherwise be forwarded and turn a perfectly good server key into a 401.
When `.env auth_token` is set it wins over everything.

Under `COMMANDCODE_API_KEYS` the proxy rotates keys: on a transport error or a retryable
status (408/409/425/429/5xx) it retries up to 3 times with backoff, picking the next key in
the pool each attempt. Once a response has begun streaming, nothing is retried.

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

- **`messages[].content` must be a plain string.** Structured content blocks are rejected:
  `expected "text" at "params.messages[2].content[0].type"`. So tool results are flattened
  into `"[tool result for <id>]"` user turns and prior assistant tool calls become
  `"[called tool name(args)]"` markers.
- **The system prompt goes in `params.system`**, not as a message. The upstream then folds
  it back into `messages[0]` itself.
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
