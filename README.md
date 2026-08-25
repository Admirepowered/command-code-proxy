# command-code OpenAI bridge

A small local proxy that exposes **OpenAI-compatible** (Chat Completions, Responses) and
**Anthropic-compatible** (Messages) APIs and forwards requests to the command-code
`/alpha/generate` endpoint, translating requests and responses both ways.

Other apps (NextChat, Cherry Studio, Codex, Claude Code, scripts using the OpenAI or Anthropic
SDKs, etc.) point their `base_url` at this proxy and talk standard OpenAI/Anthropic; this server
does the translation.

```
your app ── OpenAI /v1/chat/completions  ─┐
            OpenAI /v1/responses         ─┤
                                          ├─> bridge (python main.py) ── native /alpha/generate ──> api.commandcode.ai
            Anthropic /v1/messages       ─┘
```

## Requirements

- Python 3.10+ (tested on 3.14)
- `requests` (`pip install requests` — usually already present)

## Configuration

Edit `.env` (same directory as `main.py`):

| key               | description                                              |
|-------------------|----------------------------------------------------------|
| `base_url`        | command-code endpoint (defaults to `/alpha/generate`)    |
| `auth_token`      | your command-code CLI key (`user_...`)                   |
| `host`            | listen address (default `0.0.0.0`)                       |
| `port`            | listen port (default `8080`)                             |
| `default_model`   | model used when the client omits `model`                 |
| `models`          | comma-separated catalog returned by `GET /v1/models`     || `working_dir`     | native request `workingDir` (default `/tmp`)             |
| `environment`     | native request `environment` (default `terminal`)        |
| `memory` / `taste`| native request memory / taste strings (default empty)    |
| `skills`          | native request skills (default empty → null)             |
| `permission_mode` | native request `permissionMode` (default `standard`)     |

### Model catalog (`models.py`)

The full command-code model registry (59 models: Anthropic, OpenAI, DeepSeek,
Kimi, GLM, MiniMax, Qwen, Gemini, Grok, …) lives in `models.py`. It serves two
purposes:

- **`GET /v1/models`** — the `.env` `models` list selects and orders the entries
  shown to clients (matching works on canonical ids *or* aliases); leave it
  empty to serve the whole visible registry. Two free promo models are hidden,
  mirroring the CLI picker, but stay callable when requested explicitly.
- **Id resolution** — every request's `model` field passes through
  `models.resolve()`, which maps legacy and gateway-specific ids onto the
  canonical id sent upstream (e.g. `claude-opus-4-6` → `claude-opus-4-7`,
  `zai/glm-5.2` → `zai-org/GLM-5.2`, `openai/gpt-5.6-luna` → `gpt-5.6-luna`).
  Unknown ids pass through unchanged, so brand-new upstream models work
  without a catalog update.


## Run

```bash
python main.py
```

Endpoints:

- `POST /v1/chat/completions` — OpenAI-compatible chat (streaming and non-streaming)
- `POST /v1/responses` — OpenAI Responses API (streaming and non-streaming; what Codex uses)
- `POST /v1/messages` — Anthropic Messages API (streaming and non-streaming; what Claude Code uses)
- `GET  /v1/models` — model list

Point your client at `http://localhost:8080/v1` (e.g. base URL `http://localhost:8080/v1`,
any API key).

## Quick check

```bash
# streaming
curl -N http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek/deepseek-v4-flash","messages":[{"role":"user","content":"hello"}],"stream":true}'

# non-streaming
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek/deepseek-v4-flash","messages":[{"role":"user","content":"hello"}]}'

# Responses API (Codex)
curl -N http://localhost:8080/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek/deepseek-v4-flash","instructions":"You are a terse assistant.","input":"hello","stream":true}'
```

## Responses API (`/v1/responses`)

Requests are translated from the Responses format to the same native call:
- `instructions` (or `input` items with `type:"message"` / `role:"system"`) → native `params.system`
- `input` items (`input_text`/`input_image`/`output_text`/`function_call`/`function_call_output`) →
  native `messages` (tool outputs are flattened into user messages)
- `tools` (`{type:"function", name, description, parameters}`) → native Anthropic shape;
  `web_search`/`web_search_preview` → built-in `web_search_20250305`, `web_fetch` → `web_fetch_20250910`
- `tool_choice`, `max_output_tokens`, `temperature`, `top_p` → mapped equivalents

Streaming responses are re-emitted as the standard Responses SSE event sequence
(`response.created`, `response.output_item.added`, `response.output_text.delta`,
`response.function_call_arguments.delta`, … `response.completed`). Reasoning tokens are
exposed via `response.reasoning_summary_text.delta` events. Non-streaming clients get a
single complete `Response` object.

## Anthropic Messages API (`/v1/messages`)

Requests are translated from the Anthropic format to the same native call:
- `system` (string or text blocks) → native `params.system`
- `messages` content blocks: `text` → plain text; `tool_result` → `"[tool result for <id>]"`
  user text (same convention as the OpenAI path); `tool_use` in history → `"[called tool ...]"`
  marker; `image`/`document` → inline placeholders
- `tools` — Anthropic `{name, description, input_schema}` already matches the native shape and
  passes through; built-in `web_search_20250305`/`web_fetch_20250910` pass through untouched
- `tool_choice` (`auto`/`any`/`none`/`{type:"tool", name}`) → mapped equivalents
- `max_tokens`, `temperature`, `top_p`, `stop_sequences` → mapped equivalents

Streaming responses use the standard Anthropic SSE sequence: `message_start` →
`content_block_start` / `content_block_delta` (`thinking_delta`, `text_delta`,
`input_json_delta`) / `content_block_stop` → `message_delta` (stop_reason + usage) →
`message_stop`. Reasoning is surfaced as a `thinking` block. Non-streaming clients get a single
complete message object with parsed `tool_use.input`. Errors use the Anthropic error envelope.

## Translation notes

- **System prompt** — OpenAI `system`/`developer` messages are joined into the native
  `params.system` field.
- **Tools** — OpenAI `{type:"function", function:{...}}` schemas are converted to the native
  `{name, description, input_schema}` shape. Tool-call events are streamed back as OpenAI
  `delta.tool_calls`. Note: the upstream emits tool calls but cannot execute them or accept
  tool results within a single call, so `role:"tool"` results in history are flattened into
  user messages.
- **Streaming** — upstream always receives `stream: true` (the CLI endpoint rejects
  non-streaming requests). Non-streaming clients get a buffered single JSON response.
- **Reasoning** — reasoning deltas are exposed as `delta.reasoning_content`
  (DeepSeek convention).
- **Errors** — upstream errors are re-enveloped in the OpenAI error shape.

## ⚠️ Risk disclosure

The `/alpha/generate` endpoint is intended for the command-code **CLI** only. commandcode.ai
actively detects proxying and warns that *"continued proxying of your subscription violates the
TOS and will result in account ban."* Using this bridge may get your account banned.

A legitimate alternative exists: the official **Command Code Provider API**
(`https://api.commandcode.ai/provider/v1`, OpenAI- and Anthropic-compatible, same key), which
requires a plan with API access (GOAT/Provider+; the Go plan returns 403 `upgrade_required`).
If your plan supports it, point your apps there directly and skip this bridge entirely.
