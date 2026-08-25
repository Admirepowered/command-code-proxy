#!/usr/bin/env python3
"""OpenAI-compatible bridge to the command-code /alpha/generate endpoint.

Exposes an OpenAI Chat Completions API (POST /v1/chat/completions, GET /v1/models)
and translates each request into the native command-code body format, forwards it,
and translates the native SSE response back into OpenAI SSE.

Run:  python main.py
"""

import json
import os
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import requests

import models as catalog

# ---------------------------------------------------------------------------
# Config (.env)
# ---------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(HERE, ".env")

DEFAULTS = {
    "base_url": "https://api.commandcode.ai/alpha/generate",
    "auth_token": "",
    "host": "0.0.0.0",
    "port": "8080",
    # model settings
    "default_model": "deepseek/deepseek-v4-flash",
    "models": "deepseek/deepseek-v4-flash,deepseek/deepseek-v4-pro,"
              "claude-sonnet-5,claude-fable-5,gpt-5.4,gpt-5.3-codex,moonshotai/Kimi-K3",
    # native request envelope (optional overrides)
    "working_dir": "/tmp",
    "environment": "terminal",
    "memory": "",
    "taste": "",
    "skills": "",
    "permission_mode": "standard",
}

# Headers the command-code CLI sends; forwarded upstream unchanged.
UPSTREAM_HEADERS = {
    "Content-Type": "application/json",
    "x-command-code-version": "0.24.1",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36",
}


def load_env(path):
    """Minimal KEY=VALUE .env parser (no external dependency)."""
    env = {}
    if not os.path.exists(path):
        return env
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def get_config():
    env = load_env(ENV_PATH)
    cfg = {}
    for key, default in DEFAULTS.items():
        cfg[key] = env.get(key, default) or default
    if not cfg["auth_token"]:
        print("[warn] auth_token missing from .env; upstream calls will be unauthorized",
              file=sys.stderr)
    return cfg


# ---------------------------------------------------------------------------
# OpenAI -> native request conversion
# ---------------------------------------------------------------------------


def parts_to_text(parts):
    """Flatten an OpenAI content-parts array into plain text.

    Handles both Chat Completions parts (text/image_url) and Responses-API
    parts (input_text/output_text/refusal/input_image/input_file). The native
    endpoint is text-only, so non-text parts are represented inline rather than
    dropped silently.
    """
    out = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        if ptype in ("text", "input_text", "output_text"):
            out.append(part.get("text", ""))
        elif ptype == "refusal":
            out.append(part.get("refusal", ""))
        elif ptype in ("image_url", "input_image", "input_file"):
            out.append(f"[{ptype.removeprefix('input_')}]")
        else:
            if "text" in part:
                out.append(str(part.get("text", "")))
    return "\n".join(out)


def convert_messages(messages):
    """Split OpenAI messages into (native_messages, system_text).

    The native API only accepts `user`/`assistant` roles and keeps the system
    prompt in a separate `params.system` field, so:
      - system/developer messages -> concatenated into the system prompt
      - tool messages             -> flattened into a user message
      - assistant tool_calls      -> stripped (kept as plain text if any)
    """
    native = []
    system_parts = []
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if role in ("system", "developer"):
            if isinstance(content, str):
                system_parts.append(content)
            else:
                system_parts.append(parts_to_text(content))
            continue
        if role == "tool":
            text = content if isinstance(content, str) else parts_to_text(content)
            tool_call_id = msg.get("tool_call_id") or "?"
            native.append({
                "role": "user",
                "content": f"[tool result for {tool_call_id}]\n{text}",
            })
            continue
        if role in ("user", "assistant"):
            if isinstance(content, str):
                native.append({"role": role, "content": content})
            elif isinstance(content, list):
                native.append({"role": role, "content": parts_to_text(content)})
            elif content is None:
                # e.g. an assistant message that only carried tool_calls
                native.append({"role": role, "content": ""})
    system = "\n\n".join(p for p in system_parts if p)
    return native, system


def convert_tools(tools):
    """Convert OpenAI tool schemas to the native {name, description, input_schema} shape.

    Built-in web_search/web_fetch tools are passed through untouched.
    """
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
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            })
        elif ttype in ("web_search_20250305", "web_fetch_20250910"):
            out.append(tool)
    return out


def convert_tool_choice(tool_choice):
    """Map OpenAI tool_choice to the native AI-SDK toolChoice value."""
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        return {"none": "none", "auto": "auto", "required": "required",
                "tool_calls": "required"}.get(tool_choice, tool_choice)
    if isinstance(tool_choice, dict):
        fn = tool_choice.get("function") or {}
        name = fn.get("name") if isinstance(fn, dict) else None
        if name:
            return {"type": "tool", "toolName": name}
    return None


def build_native_body(params, cfg):
    """Wrap converted params in the top-level command-code envelope."""
    body = {
        "config": {
            "workingDir": cfg.get("working_dir") or "/tmp",
            "date": time.strftime("%Y-%m-%d"),
            "environment": cfg.get("environment") or "terminal",
            "structure": [],
            "isGitRepo": False,
            "currentBranch": "",
            "mainBranch": "",
            "gitStatus": "",
            "recentCommits": [],
        },
        "memory": cfg.get("memory", ""),
        "taste": cfg.get("taste", ""),
        "skills": cfg.get("skills") or None,
        "permissionMode": cfg.get("permission_mode") or "standard",
        "params": params,
    }
    return body


def convert_openai_request(openai_body, cfg):
    """Translate an OpenAI /v1/chat/completions body into the native body.

    `stream` is always forced to True upstream: the CLI endpoint rejects
    non-streaming requests (anti-proxy check).
    """
    messages, system = convert_messages(openai_body.get("messages", []))
    tools = convert_tools(openai_body.get("tools"))
    tool_choice = convert_tool_choice(openai_body.get("tool_choice"))

    params = {
        "model": catalog.resolve(openai_body.get("model"))
                 or cfg.get("default_model") or "deepseek/deepseek-v4-flash",
        "messages": messages,
        "system": system,
        "stream": True,
    }
    if tools:
        params["tools"] = tools
    if tool_choice is not None:
        params["toolChoice"] = tool_choice
    # Whitelisted passthrough of common sampling params.
    max_tokens = openai_body.get("max_tokens") or openai_body.get("max_completion_tokens")
    if max_tokens is not None:
        params["max_tokens"] = max_tokens
    for key in ("temperature", "top_p", "stop"):
        if openai_body.get(key) is not None:
            params[key] = openai_body[key]
    return build_native_body(params, cfg)


def responses_input_to_messages(input_items, system_parts):
    """Flatten OpenAI Responses-API `input` items into native user/assistant messages.

    Items that have no text equivalent (previous function_call / reasoning items)
    are represented inline, since the native API only accepts user/assistant text.
    """
    native = []
    for item in input_items or []:
        if isinstance(item, str):
            native.append({"role": "user", "content": item})
            continue
        if not isinstance(item, dict):
            continue
        itype = item.get("type")
        if itype == "function_call":
            args = item.get("arguments", "")
            name = item.get("name", "")
            native.append({"role": "assistant", "content": f"[called {name}({args})]"})
            continue
        if itype == "function_call_output":
            output = item.get("output", "")
            if isinstance(output, (dict, list)):
                output = json.dumps(output, ensure_ascii=False)
            native.append({
                "role": "user",
                "content": f"[tool result for {item.get('call_id', '?')}]\n{output}",
            })
            continue
        if itype in ("reasoning", "message_attempt"):
            continue  # previous reasoning/attempts, not needed by the model
        role = item.get("role")
        content = item.get("content")
        if role == "developer":
            system_parts.append(content if isinstance(content, str) else parts_to_text(content))
            continue
        if role in ("user", "assistant"):
            if isinstance(content, str):
                native.append({"role": role, "content": content})
            elif isinstance(content, list):
                native.append({"role": role, "content": parts_to_text(content)})
            elif content is None:
                native.append({"role": role, "content": ""})
    return native


def convert_responses_tools(tools):
    """Convert OpenAI Responses-API tool defs to the native shape.

    Responses tools are {type: function, name, description, parameters};
    built-ins (web_search / web_fetch) map to the native typed forms.
    """
    out = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        ttype = tool.get("type")
        if ttype == "function":
            out.append({
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "input_schema": tool.get("parameters") or {"type": "object", "properties": {}},
            })
        elif ttype in ("web_search", "web_search_preview"):
            out.append({"type": "web_search_20250305", "name": "web_search"})
        elif ttype == "web_fetch":
            out.append({"type": "web_fetch_20250910", "name": "web_fetch"})
        elif ttype in ("web_search_20250305", "web_fetch_20250910"):
            out.append(tool)
    return out


def convert_responses_tool_choice(tool_choice):
    """Map Responses-API tool_choice ({type:function, name}) to the native value."""
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        return {"none": "none", "auto": "auto", "required": "required"}.get(tool_choice, tool_choice)
    if isinstance(tool_choice, dict):
        name = tool_choice.get("name")
        if name:
            return {"type": "tool", "toolName": name}
    return None


def convert_responses_request(openai_body, cfg):
    """Translate an OpenAI Responses-API (/v1/responses) body into the native body."""
    input_items = openai_body.get("input")
    if isinstance(input_items, dict):
        input_items = [input_items]
    system_parts = []
    instructions = openai_body.get("instructions")
    if instructions:
        system_parts.append(instructions if isinstance(instructions, str)
                            else parts_to_text(instructions))
    messages = responses_input_to_messages(input_items, system_parts)
    system = "\n\n".join(p for p in system_parts if p)
    tools = convert_responses_tools(openai_body.get("tools"))
    tool_choice = convert_responses_tool_choice(openai_body.get("tool_choice"))

    params = {
        "model": catalog.resolve(openai_body.get("model"))
                 or cfg.get("default_model") or "deepseek/deepseek-v4-flash",
        "messages": messages,
        "system": system,
        "stream": True,
    }
    if tools:
        params["tools"] = tools
    if tool_choice is not None:
        params["toolChoice"] = tool_choice
    max_tokens = openai_body.get("max_output_tokens") or openai_body.get("max_tokens")
    if max_tokens is not None:
        params["max_tokens"] = max_tokens
    for key in ("temperature", "top_p", "stop"):
        if openai_body.get(key) is not None:
            params[key] = openai_body[key]
    return build_native_body(params, cfg)


# ---------------------------------------------------------------------------
# Anthropic Messages API (/v1/messages) request conversion
# ---------------------------------------------------------------------------


def anthropic_blocks_to_text(content):
    """Flatten Anthropic content (string or block list) into plain text."""
    if isinstance(content, str):
        return content
    out = []
    for block in content or []:
        if isinstance(block, str):
            out.append(block)
        elif isinstance(block, dict):
            btype = block.get("type")
            if btype == "text":
                out.append(block.get("text", ""))
            elif btype == "tool_result":
                out.append(anthropic_blocks_to_text(block.get("content")))
            elif btype in ("image", "document"):
                out.append(f"[{btype}]")
            elif "text" in block:
                out.append(str(block.get("text", "")))
    return "\n".join(t for t in out if t)


def convert_anthropic_messages(messages):
    """Convert Anthropic messages to native user/assistant-only messages.

    tool_result blocks are flattened into "[tool result for <id>]" text (same
    convention as the OpenAI path); tool_use blocks in assistant history are
    kept as a short "[called tool ...]" marker so the model sees its own prior
    actions.
    """
    native = []
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role not in ("user", "assistant"):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            native.append({"role": role, "content": content})
            continue
        parts = []
        for block in content or []:
            if isinstance(block, str):
                parts.append(block)
                continue
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                parts.append(block.get("text", ""))
            elif btype == "tool_result":
                result_text = anthropic_blocks_to_text(block.get("content"))
                parts.append(f"[tool result for {block.get('tool_use_id') or '?'}]\n{result_text}")
            elif btype == "tool_use":
                called_input = json.dumps(block.get("input") or {}, ensure_ascii=False)
                parts.append(f"[called tool {block.get('name', '?')}({called_input})]")
            elif btype in ("image", "document"):
                parts.append(f"[{btype}]")
        text = "\n".join(p for p in parts if p)
        if text:
            native.append({"role": role, "content": text})
    return native


def convert_anthropic_tools(tools):
    """Pass Anthropic tool schemas through — they already match the native
    {name, description, input_schema} shape. Built-in web_search/web_fetch
    share the same type names, so they pass through untouched too."""
    out = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        ttype = tool.get("type")
        if ttype in ("web_search_20250305", "web_fetch_20250910"):
            out.append(tool)
        elif tool.get("name"):
            out.append({
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "input_schema": tool.get("input_schema")
                                or {"type": "object", "properties": {}},
            })
    return out


def convert_anthropic_tool_choice(tool_choice):
    """Map Anthropic tool_choice ({type: auto|any|none|tool}) to the native
    AI-SDK toolChoice value."""
    if not isinstance(tool_choice, dict):
        return None
    ttype = tool_choice.get("type")
    if ttype == "any":
        return "required"
    if ttype in ("auto", "none"):
        return ttype
    if ttype == "tool" and tool_choice.get("name"):
        return {"type": "tool", "toolName": tool_choice["name"]}
    return None


def convert_anthropic_request(body, cfg):
    """Translate an Anthropic Messages-API (/v1/messages) body into the native body."""
    params = {
        "model": catalog.resolve(body.get("model"))
                 or cfg.get("default_model") or "deepseek/deepseek-v4-flash",
        "messages": convert_anthropic_messages(body.get("messages")),
        "system": anthropic_blocks_to_text(body.get("system")),
        "stream": True,
    }
    tools = convert_anthropic_tools(body.get("tools"))
    if tools:
        params["tools"] = tools
    tool_choice = convert_anthropic_tool_choice(body.get("tool_choice"))
    if tool_choice is not None:
        params["toolChoice"] = tool_choice
    if body.get("max_tokens") is not None:
        params["max_tokens"] = body["max_tokens"]
    for key in ("temperature", "top_p"):
        if body.get(key) is not None:
            params[key] = body[key]
    if body.get("stop_sequences"):
        params["stop"] = body["stop_sequences"]
    return build_native_body(params, cfg)


# ---------------------------------------------------------------------------
# Native response translation
# ---------------------------------------------------------------------------


def map_finish_reason(reason):
    """Native 'tool-calls' -> OpenAI 'tool_calls'."""
    return {
        "tool-calls": "tool_calls",
        "tool-calls-paused": "tool_calls",
    }.get(reason, reason or "stop")


def openai_usage(total_usage):
    """Map native totalUsage to the OpenAI usage shape."""
    return {
        "prompt_tokens": total_usage.get("inputTokens"),
        "completion_tokens": total_usage.get("outputTokens"),
        "total_tokens": total_usage.get("totalTokens"),
    }


class Translator:
    """Consumes native stream-part events, emits OpenAI chunk dicts.

    Both the streaming path (write each chunk as SSE) and the buffered path
    (collect everything, then build a single chat.completion) share this.
    """

    def __init__(self, model):
        self.model = model
        self.id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        self.created = int(time.time())
        self.reasoning = []
        self.text = []
        self.tool_calls = []   # {index, id, type, function:{name, arguments}}
        self.tool_index = -1
        self.finish_reason = None
        self.usage = None
        self._sent_role = False

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

    def _content_chunk(self, delta):
        """Emit a delta chunk, guaranteeing the assistant role on the first one."""
        if not self._sent_role:
            delta = {"role": "assistant", **delta}
            self._sent_role = True
        return self._chunk(delta)

    def on_event(self, obj):
        """Translate one native event; return a list of OpenAI chunks to emit."""
        etype = obj.get("type")
        if etype == "reasoning-delta":
            self.reasoning.append(obj.get("text", ""))
            return [self._content_chunk({"reasoning_content": obj.get("text", "")})]
        if etype == "text-start":
            return [self._content_chunk({"content": ""})]
        if etype == "text-delta":
            self.text.append(obj.get("text", ""))
            return [self._content_chunk({"content": obj.get("text", "")})]
        if etype == "tool-input-start":
            self.tool_index += 1
            tc = {
                "index": self.tool_index,
                "id": obj.get("id", f"call_{self.tool_index}"),
                "type": "function",
                "function": {"name": obj.get("toolName", ""), "arguments": ""},
            }
            self.tool_calls.append(tc)
            return [self._content_chunk({"tool_calls": [tc]})]
        if etype == "tool-input-delta":
            delta_text = obj.get("delta", "")
            if 0 <= self.tool_index < len(self.tool_calls):
                self.tool_calls[self.tool_index]["function"]["arguments"] += delta_text
            return [self._chunk({
                "tool_calls": [{"index": max(self.tool_index, 0),
                                "function": {"arguments": delta_text}}],
            })]
        if etype == "finish":
            self.finish_reason = map_finish_reason(obj.get("finishReason"))
            total_usage = obj.get("totalUsage") or {}
            self.usage = openai_usage(total_usage)
            return [self._chunk({}, finish_reason=self.finish_reason, usage=self.usage)]
        # start / start-step / finish-step / provider-metadata / tool-call -> nothing
        return []

    def final_message(self):
        message = {"role": "assistant", "content": "".join(self.text)}
        if self.reasoning:
            message["reasoning_content"] = "".join(self.reasoning)
        if self.tool_calls:
            message["tool_calls"] = [
                {"id": tc["id"], "type": "function",
                 "function": {"name": tc["function"]["name"],
                              "arguments": tc["function"]["arguments"]}}
                for tc in self.tool_calls
            ]
        return message

    def final_completion(self):
        return {
            "id": self.id,
            "object": "chat.completion",
            "created": self.created,
            "model": self.model,
            "choices": [{
                "index": 0,
                "message": self.final_message(),
                "finish_reason": self.finish_reason or "stop",
            }],
            "usage": self.usage or {
                "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            },
        }


def responses_usage(total_usage):
    """Map native totalUsage to the OpenAI Responses-API usage shape."""
    return {
        "input_tokens": total_usage.get("inputTokens"),
        "output_tokens": total_usage.get("outputTokens"),
        "total_tokens": total_usage.get("totalTokens"),
        "input_tokens_details": {"cached_tokens": total_usage.get("cachedInputTokens")},
        "output_tokens_details": {"reasoning_tokens": total_usage.get("reasoningTokens")},
    }


class ResponsesTranslator:
    """Consumes native stream-part events, emits OpenAI Responses-API SSE events.

    Emits the standard event sequence the OpenAI SDK / Codex expect:
    response.created -> response.in_progress -> (per output item:
    output_item.added -> content_part/arguments deltas -> output_item.done)
    -> response.completed. Reasoning is surfaced via reasoning_summary_text.
    """

    def __init__(self, model):
        self.model = model
        self.id = f"resp_{uuid.uuid4().hex[:24]}"
        self.created_at = int(time.time())
        self.reasoning = []
        self.text = []
        self.tool_calls = []   # {item_id, call_id, name, arguments, index}
        self.items = []        # completed output items, in generation order
        self.output_index = 0
        self.finish_reason = None
        self.usage = None
        self._reasoning_id = None
        self._reasoning_index = None
        self._msg_id = None
        self._msg_index = None
        self._current_tool = None
        self.completed = False   # True once a native 'finish' event is seen

    def meta_response(self, status="in_progress"):
        return {
            "id": self.id,
            "object": "response",
            "created_at": self.created_at,
            "status": status,
            "model": self.model,
            "output": [],
        }

    def on_event(self, obj):
        """Translate one native event; return a list of (event_type, data) pairs."""
        etype = obj.get("type")
        if etype == "reasoning-delta":
            self.reasoning.append(obj.get("text", ""))
            out = []
            if self._reasoning_id is None:
                self._reasoning_index = self.output_index
                self.output_index += 1
                self._reasoning_id = f"rsn_{uuid.uuid4().hex[:16]}"
                out.append(("response.output_item.added", {
                    "output_index": self._reasoning_index,
                    "item": {"id": self._reasoning_id, "type": "reasoning",
                             "status": "in_progress",
                             "summary": [{"type": "summary_text", "text": ""}],
                             "content": [{"type": "reasoning_text", "text": ""}]},
                }))
            out.append(("response.reasoning_summary_text.delta", {
                "item_id": self._reasoning_id, "output_index": self._reasoning_index,
                "summary_index": 0, "delta": obj.get("text", ""),
            }))
            return out
        if etype == "reasoning-end":
            if self._reasoning_id is None:
                return []
            text = "".join(self.reasoning)
            item = {"id": self._reasoning_id, "type": "reasoning", "status": "completed",
                    "summary": [{"type": "summary_text", "text": text}],
                    "content": [{"type": "reasoning_text", "text": text}]}
            self.items.append(item)
            out = [
                ("response.reasoning_summary_text.done", {
                    "item_id": self._reasoning_id, "output_index": self._reasoning_index,
                    "summary_index": 0, "summary": item["summary"],
                }),
                ("response.output_item.done", {
                    "output_index": self._reasoning_index, "item": item,
                }),
            ]
            self._reasoning_id = None
            return out
        if etype == "text-start":
            self._msg_index = self.output_index
            self.output_index += 1
            self._msg_id = f"msg_{uuid.uuid4().hex[:16]}"
            return [
                ("response.output_item.added", {
                    "output_index": self._msg_index,
                    "item": {"id": self._msg_id, "type": "message", "status": "in_progress",
                             "role": "assistant", "content": []},
                }),
                ("response.content_part.added", {
                    "item_id": self._msg_id, "output_index": self._msg_index,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": "", "annotations": []},
                }),
            ]
        if etype == "text-delta":
            self.text.append(obj.get("text", ""))
            return [("response.output_text.delta", {
                "item_id": self._msg_id, "output_index": self._msg_index,
                "content_index": 0, "delta": obj.get("text", ""),
            })]
        if etype == "text-end":
            if self._msg_id is None:
                return []
            full = "".join(self.text)
            item = {"id": self._msg_id, "type": "message", "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": full, "annotations": []}]}
            self.items.append(item)
            out = [
                ("response.output_text.done", {
                    "item_id": self._msg_id, "output_index": self._msg_index,
                    "content_index": 0, "text": full,
                }),
                ("response.content_part.done", {
                    "item_id": self._msg_id, "output_index": self._msg_index,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": full, "annotations": []},
                }),
                ("response.output_item.done", {
                    "output_index": self._msg_index, "item": item,
                }),
            ]
            self._msg_id = None
            return out
        if etype == "tool-input-start":
            call_id = obj.get("id", f"call_{uuid.uuid4().hex[:16]}")
            name = obj.get("toolName", "")
            idx = self.output_index
            self.output_index += 1
            self._current_tool = {"item_id": call_id, "call_id": call_id, "name": name,
                                  "arguments": "", "index": idx}
            self.tool_calls.append(self._current_tool)
            return [("response.output_item.added", {
                "output_index": idx,
                "item": {"id": call_id, "type": "function_call", "status": "in_progress",
                         "call_id": call_id, "name": name, "arguments": ""},
            })]
        if etype == "tool-input-delta":
            if self._current_tool is None:
                return []
            self._current_tool["arguments"] += obj.get("delta", "")
            return [("response.function_call_arguments.delta", {
                "item_id": self._current_tool["item_id"],
                "output_index": self._current_tool["index"],
                "delta": obj.get("delta", ""),
            })]
        if etype == "tool-input-end":
            if self._current_tool is None:
                return []
            tc = self._current_tool
            item = {"id": tc["item_id"], "type": "function_call", "status": "completed",
                    "call_id": tc["call_id"], "name": tc["name"],
                    "arguments": tc["arguments"]}
            self.items.append(item)
            out = [
                ("response.function_call_arguments.done", {
                    "item_id": tc["item_id"], "output_index": tc["index"],
                    "arguments": tc["arguments"],
                }),
                ("response.output_item.done", {
                    "output_index": tc["index"], "item": item,
                }),
            ]
            self._current_tool = None
            return out
        if etype == "finish":
            self.finish_reason = map_finish_reason(obj.get("finishReason"))
            self.usage = responses_usage(obj.get("totalUsage") or {})
            self.completed = True
            return [("response.completed", {
                "type": "response.completed",
                "response": self.final_response(status="completed"),
            })]
        return []

    def final_response(self, status="completed"):
        incomplete_details = None
        if self.finish_reason == "length":
            incomplete_details = {"reason": "max_output_tokens"}
        return {
            "id": self.id,
            "object": "response",
            "created_at": self.created_at,
            "status": status,
            "model": self.model,
            "output": self.items,
            "parallel_tool_calls": True,
            "tools": [],
            "tool_choice": "auto",
            "usage": self.usage or {
                "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
            "incomplete_details": incomplete_details,
            "error": None,
        }


class AnthropicTranslator:
    """Consumes native stream-part events, emits Anthropic Messages-API events.

    Event sequence: message_start -> per content block (content_block_start ->
    text_delta / input_json_delta -> content_block_stop) -> message_delta
    (with stop_reason + usage) -> message_stop. Reasoning is surfaced as a
    dedicated thinking block.
    """

    def __init__(self, model):
        self.model = model
        self.id = f"msg_{uuid.uuid4().hex[:24]}"
        self.created_at = int(time.time())
        self.reasoning = []
        self.text = []
        self.tool_calls = []   # {index, id, name, input}
        self.stop_reason = None
        self.usage = None
        self.blocks = []       # completed blocks for the buffered path
        self._block_index = -1     # index of the currently open block
        self._block_type = None
        self._tool = None
        self.completed = False

    # -- event construction helpers ----------------------------------------

    def _start_event(self):
        return ("message_start", {
            "type": "message_start",
            "message": {
                "id": self.id,
                "type": "message",
                "role": "assistant",
                "model": self.model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        })

    def _block_start(self, block):
        """Open a new content block; returns the content_block_start event."""
        self._block_index += 1
        self._block_type = block["type"]
        return ("content_block_start", {
            "type": "content_block_start",
            "index": self._block_index,
            "content_block": block,
        })

    def _block_stop(self):
        return ("content_block_stop", {
            "type": "content_block_stop",
            "index": self._block_index,
        })

    def anthropic_usage(self, total_usage):
        return {
            "input_tokens": total_usage.get("inputTokens") or 0,
            "output_tokens": total_usage.get("outputTokens") or 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": total_usage.get("cachedInputTokens") or 0,
        }

    def final_message(self, stop_reason=None):
        content = []
        if self.reasoning:
            content.append({"type": "thinking",
                            "thinking": "".join(self.reasoning)})
        content.append({"type": "text", "text": "".join(self.text)})
        for tc in self.tool_calls:
            content.append({"type": "tool_use", "id": tc["id"],
                            "name": tc["name"], "input": tc["input"]})
        return {
            "id": self.id,
            "type": "message",
            "role": "assistant",
            "model": self.model,
            "content": content,
            "stop_reason": stop_reason or self.stop_reason or "end_turn",
            "stop_sequence": None,
            "usage": self.usage or {
                "input_tokens": 0, "output_tokens": 0,
                "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
            },
        }

    # -- main entry ----------------------------------------------------------

    def on_event(self, obj):
        """Translate one native event; return a list of (event_type, data)."""
        etype = obj.get("type")
        out = []
        if etype == "reasoning-start":
            out.append(self._block_start({"type": "thinking", "thinking": ""}))
        elif etype == "reasoning-delta":
            if self._block_type != "thinking":
                out.append(self._block_start({"type": "thinking", "thinking": ""}))
            self.reasoning.append(obj.get("text", ""))
            out.append(("content_block_delta", {
                "type": "content_block_delta", "index": self._block_index,
                "delta": {"type": "thinking_delta", "thinking": obj.get("text", "")},
            }))
        elif etype == "reasoning-end":
            if self._block_type == "thinking":
                self.blocks.append({"type": "thinking", "thinking": "".join(self.reasoning)})
                out.append(self._block_stop())
                self._block_type = None
        elif etype == "text-start":
            out.append(self._block_start({"type": "text", "text": ""}))
        elif etype == "text-delta":
            if self._block_type != "text":
                out.append(self._block_start({"type": "text", "text": ""}))
            self.text.append(obj.get("text", ""))
            out.append(("content_block_delta", {
                "type": "content_block_delta", "index": self._block_index,
                "delta": {"type": "text_delta", "text": obj.get("text", "")},
            }))
        elif etype == "text-end":
            if self._block_type == "text":
                self.blocks.append({"type": "text", "text": "".join(self.text)})
                out.append(self._block_stop())
                self._block_type = None
        elif etype == "tool-input-start":
            call_id = obj.get("id", f"toolu_{uuid.uuid4().hex[:16]}")
            name = obj.get("toolName", "")
            self._tool = {"index": None, "id": call_id, "name": name, "input_raw": ""}
            out.append(self._block_start({
                "type": "tool_use", "id": call_id, "name": name, "input": {},
            }))
            self._tool["index"] = self._block_index
        elif etype == "tool-input-delta":
            if self._tool is not None:
                self._tool["input_raw"] += obj.get("delta", "")
                out.append(("content_block_delta", {
                    "type": "content_block_delta", "index": self._block_index,
                    "delta": {"type": "input_json_delta",
                              "partial_json": obj.get("delta", "")},
                }))
        elif etype == "tool-input-end":
            if self._tool is not None:
                try:
                    parsed = json.loads(self._tool["input_raw"]) if self._tool["input_raw"] else {}
                except ValueError:
                    parsed = {}
                self.tool_calls.append({"id": self._tool["id"], "name": self._tool["name"],
                                        "input": parsed})
                self.blocks.append({"type": "tool_use", "id": self._tool["id"],
                                    "name": self._tool["name"], "input": parsed})
                out.append(self._block_stop())
                self._tool = None
                self._block_type = None
        elif etype == "finish":
            reason = obj.get("finishReason")
            self.stop_reason = "tool_use" if map_finish_reason(reason) == "tool_calls" else "end_turn"
            if reason == "length":
                self.stop_reason = "max_tokens"
            self.usage = self.anthropic_usage(obj.get("totalUsage") or {})
            self.completed = True
            out.append(("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": self.stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": self.usage["output_tokens"]},
            }))
            out.append(("message_stop", {"type": "message_stop"}))
        return out


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

def build_models(cfg):
    """Build the /v1/models catalog from the full model registry.

    The `.env` `models` list selects and orders the entries (matching is done
    on both canonical ids and aliases); when it's empty the whole registry is
    served. Ids listed there but absent from the registry are appended so a
    brand-new upstream model stays usable without editing models.py.
    """
    if cfg.get("models"):
        seen = set()
        picked = []
        for raw in cfg["models"].split(","):
            mid = raw.strip()
            if not mid:
                continue
            canonical = catalog.resolve(mid)
            entry = catalog.get(mid)
            if entry and canonical not in seen:
                seen.add(canonical)
                picked.append({"id": entry["id"], "object": "model",
                               "created": 0, "owned_by": entry["vendor"]})
            elif not entry:
                # Not in the registry (yet) — keep it callable anyway.
                picked.append({"id": mid, "object": "model",
                               "created": 0, "owned_by": "command-code"})
        return picked
    return catalog.openai_catalog()


def iter_sse_lines(resp):
    """Yield upstream SSE lines decoded as UTF-8, regardless of Content-Type.

    requests' decode_unicode=True falls back to ISO-8859-1 when the response
    has no charset (typical for text/event-stream), which garbles non-ASCII
    text ("ä½ å¥½" mojibake). Read bytes and decode explicitly instead.
    """
    for raw in resp.iter_lines():
        if raw is None:
            continue
        yield raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw


def error_envelope(status, message, code=None, type_="invalid_request_error"):
    return {"error": {"message": message, "type": type_, "code": code, "status": status}}


def upstream_error_envelope(upstream_error):
    """Translate the native {'success': false, 'error': {...}} shape to OpenAI's."""
    err = upstream_error.get("error", {}) if isinstance(upstream_error, dict) else {}
    status = err.get("status") or 502
    return error_envelope(
        status,
        err.get("message") or "upstream request failed",
        code=err.get("code"),
        type_="upstream_error",
    ), status


class BridgeHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "CommandCodeBridge/1.0"

    # -- helpers -----------------------------------------------------------

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def _send_json(self, status, obj, extra_headers=None):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors_headers()
        for key, value in (extra_headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self):
        try:
            length = int(self.headers.get("Content-Length", 0) or 0)
        except ValueError:
            length = 0
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return None

    def _close_upstream(self, resp):
        try:
            resp.close()
        except Exception:
            pass

    def _upstream_post(self, native_body, cfg):
        """POST to the command-code endpoint; on request failure reply 502."""
        headers = dict(UPSTREAM_HEADERS)
        headers["Authorization"] = f"Bearer {cfg['auth_token']}"
        try:
            return requests.post(
                cfg["base_url"],
                json=native_body,
                headers=headers,
                stream=True,
                timeout=(10, 600),
            )
        except requests.RequestException as exc:
            self._send_json(502, error_envelope(
                502, f"upstream request failed: {exc}", code="upstream_unreachable",
                type_="upstream_error"))
            return None

    def _upstream_ok(self, resp):
        """Return True if the upstream response is usable; else reply with an
        OpenAI error envelope and return False."""
        if resp.status_code == 200:
            return True
        try:
            upstream_body = resp.json()
        except ValueError:
            upstream_body = None
        self._close_upstream(resp)
        if upstream_body:
            err, status = upstream_error_envelope(upstream_body)
        else:
            err, status = (
                error_envelope(resp.status_code,
                               f"upstream returned HTTP {resp.status_code}",
                               code="upstream_error", type_="upstream_error"),
                resp.status_code)
        self._send_json(status, err)
        return False

    # -- routes -------------------------------------------------------------

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors_headers()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        if self.path.split("?")[0].rstrip("/") == "/v1/models":
            self._send_json(200, {"object": "list", "data": self.server.models})
        else:
            self._send_json(404, error_envelope(404, "Not Found", code="not_found"))

    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/")
        if path == "/v1/chat/completions":
            self._handle_chat_completions()
        elif path == "/v1/responses":
            self._handle_responses()
        elif path == "/v1/messages":
            self._handle_messages()
        else:
            self._send_json(404, error_envelope(404, "Not Found", code="not_found"))

    # -- chat completions ----------------------------------------------------

    def _handle_chat_completions(self):
        cfg = self.server.cfg
        openai_body = self._read_json_body()
        if openai_body is None:
            self._send_json(400, error_envelope(400, "Invalid JSON body"))
            return

        native_body = convert_openai_request(openai_body, cfg)
        client_wants_stream = bool(openai_body.get("stream"))

        resp = self._upstream_post(native_body, cfg)
        if resp is None or not self._upstream_ok(resp):
            return

        translator = Translator(openai_body.get("model")
                                or cfg.get("default_model") or "deepseek/deepseek-v4-flash")

        if client_wants_stream:
            self._stream_response(resp, translator)
        else:
            self._buffer_response(resp, translator)

    def _handle_responses(self):
        cfg = self.server.cfg
        openai_body = self._read_json_body()
        if openai_body is None:
            self._send_json(400, error_envelope(400, "Invalid JSON body"))
            return

        native_body = convert_responses_request(openai_body, cfg)
        client_wants_stream = bool(openai_body.get("stream"))

        resp = self._upstream_post(native_body, cfg)
        if resp is None or not self._upstream_ok(resp):
            return

        translator = ResponsesTranslator(openai_body.get("model")
                                         or cfg.get("default_model") or "deepseek/deepseek-v4-flash")

        if client_wants_stream:
            self._stream_responses(resp, translator)
        else:
            self._buffer_responses(resp, translator)

    # -- Anthropic messages --------------------------------------------------

    def _anthropic_error(self, status, message, err_type="invalid_request_error"):
        """Anthropic-shaped error body."""
        self._send_json(status, {"type": "error",
                                 "error": {"type": err_type, "message": message}})

    def _upstream_ok_anthropic(self, resp):
        """Like _upstream_ok but replies with an Anthropic error envelope."""
        if resp.status_code == 200:
            return True
        try:
            upstream_body = resp.json()
        except ValueError:
            upstream_body = None
        self._close_upstream(resp)
        if upstream_body and isinstance(upstream_body, dict) and upstream_body.get("error"):
            err = upstream_body["error"]
            message = err.get("message") or "upstream request failed"
            status = err.get("status") or resp.status_code
        else:
            message = f"upstream returned HTTP {resp.status_code}"
            status = resp.status_code
        self._anthropic_error(status, message, err_type="api_error")
        return False

    def _handle_messages(self):
        cfg = self.server.cfg
        body = self._read_json_body()
        if body is None:
            self._anthropic_error(400, "Invalid JSON body")
            return

        native_body = convert_anthropic_request(body, cfg)
        client_wants_stream = bool(body.get("stream"))

        resp = self._upstream_post(native_body, cfg)
        if resp is None:
            return
        # _upstream_post already replied on connection failure; _upstream_ok
        # would reply in OpenAI shape, so use the Anthropic variant instead.
        if not self._upstream_ok_anthropic(resp):
            return

        translator = AnthropicTranslator(body.get("model")
                                         or cfg.get("default_model")
                                         or "deepseek/deepseek-v4-flash")

        if client_wants_stream:
            self._stream_anthropic(resp, translator)
        else:
            self._buffer_anthropic(resp, translator)

    def _write_anthropic_event(self, event_type, data):
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.wfile.write(f"event: {event_type}\n".encode("utf-8"))
        self.wfile.write(b"data: " + payload + b"\n\n")
        self.wfile.flush()

    def _stream_anthropic(self, resp, translator):
        self._sse_headers()
        try:
            self._write_anthropic_event(*translator._start_event())
            for line in iter_sse_lines(resp):
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                for event_type, data in translator.on_event(obj):
                    self._write_anthropic_event(event_type, data)
            # The upstream may end without a native 'finish' event; never leave
            # the stream without terminal events.
            if not translator.completed:
                self._write_anthropic_event("message_delta", {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": sum(len(t) for t in translator.text)},
                })
                self._write_anthropic_event("message_stop", {"type": "message_stop"})
            self.close_connection = True
        except (BrokenPipeError, ConnectionResetError):
            pass  # client went away; stop pulling upstream (don't waste tokens)
        except requests.RequestException as exc:
            try:
                self._write_anthropic_event("error", {
                    "type": "error",
                    "error": {"type": "api_error", "message": str(exc)},
                })
            except (BrokenPipeError, ConnectionResetError):
                pass
        except Exception as exc:  # noqa: BLE001 -- a broken stream must never
            try:                  # end with a bare disconnect
                self._write_anthropic_event("error", {
                    "type": "error",
                    "error": {"type": "api_error", "message": str(exc)},
                })
            except (BrokenPipeError, ConnectionResetError):
                pass
        finally:
            self._close_upstream(resp)

    def _buffer_anthropic(self, resp, translator):
        try:
            for line in iter_sse_lines(resp):
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                translator.on_event(obj)
            self._close_upstream(resp)
        except requests.RequestException as exc:
            self._close_upstream(resp)
            self._anthropic_error(502, f"upstream stream failed: {exc}",
                                  err_type="api_error")
            return
        self._send_json(200, translator.final_message())

    # -- streaming helpers --------------------------------------------------

    def _sse_headers(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

    def _write_sse(self, chunk):
        if chunk is None:
            self.wfile.write(b"data: [DONE]\n\n")
        else:
            payload = json.dumps(chunk, ensure_ascii=False).encode("utf-8")
            self.wfile.write(b"data: " + payload + b"\n\n")
        self.wfile.flush()

    def _write_sse_event(self, event_type, data):
        # Responses-API SSE clients (openai SDK / codex) dispatch on the
        # top-level `type` field inside the JSON data, NOT on the `event:`
        # header -- a payload without it is silently dropped and the client
        # reports "stream ended before response.completed". Guarantee every
        # event carries its type (the Anthropic payloads already include it).
        if "type" not in data:
            data = {"type": event_type, **data}
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.wfile.write(f"event: {event_type}\n".encode("utf-8"))
        self.wfile.write(b"data: " + payload + b"\n\n")
        self.wfile.flush()

    def _stream_response(self, resp, translator):
        self._sse_headers()
        try:
            for line in iter_sse_lines(resp):
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                for chunk in translator.on_event(obj):
                    self._write_sse(chunk)
            self._write_sse(None)  # data: [DONE]
            # Close the connection so clients (curl -N, HTTP/1.1) see EOF.
            self.close_connection = True
        except (BrokenPipeError, ConnectionResetError):
            pass  # client went away; stop pulling upstream (don't waste tokens)
        finally:
            self._close_upstream(resp)

    def _stream_responses(self, resp, translator):
        self._sse_headers()
        try:
            self._write_sse_event("response.created",
                                  {"type": "response.created",
                                   "response": translator.meta_response("in_progress")})
            self._write_sse_event("response.in_progress",
                                  {"type": "response.in_progress",
                                   "response": translator.meta_response("in_progress")})
            for line in iter_sse_lines(resp):
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                for event_type, data in translator.on_event(obj):
                    self._write_sse_event(event_type, data)
            # Upstream may end without a native 'finish' event; never leave the
            # client without a terminal event, or it reports a bare disconnect.
            if not translator.completed:
                self._write_sse_event("response.completed",
                                      {"type": "response.completed",
                                       "response": translator.final_response(status="completed")})
            # Close the connection so clients see EOF after response.completed.
            self.close_connection = True
        except (BrokenPipeError, ConnectionResetError):
            pass  # client went away; stop pulling upstream (don't waste tokens)
        except requests.RequestException as exc:
            try:
                self._write_sse_event("response.failed", {
                    "type": "response.failed",
                    "response": translator.final_response(status="failed"),
                    "error": {"code": "upstream_error", "message": str(exc)},
                })
            except (BrokenPipeError, ConnectionResetError):
                pass
        except Exception as exc:  # noqa: BLE001 -- a broken stream must never
            try:                  # end with a bare disconnect
                self._write_sse_event("response.failed", {
                    "type": "response.failed",
                    "response": translator.final_response(status="failed"),
                    "error": {"code": "translation_error", "message": str(exc)},
                })
            except (BrokenPipeError, ConnectionResetError):
                pass
        finally:
            self._close_upstream(resp)

    def _buffer_response(self, resp, translator):
        try:
            for line in iter_sse_lines(resp):
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                translator.on_event(obj)
            self._close_upstream(resp)
        except requests.RequestException as exc:
            self._close_upstream(resp)
            self._send_json(502, error_envelope(
                502, f"upstream stream failed: {exc}", code="upstream_error",
                type_="upstream_error"))
            return
        self._send_json(200, translator.final_completion())

    def _buffer_responses(self, resp, translator):
        try:
            for line in iter_sse_lines(resp):
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                translator.on_event(obj)
            self._close_upstream(resp)
        except requests.RequestException as exc:
            self._close_upstream(resp)
            self._send_json(502, error_envelope(
                502, f"upstream stream failed: {exc}", code="upstream_error",
                type_="upstream_error"))
            return
        self._send_json(200, translator.final_response(status="completed"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    cfg = get_config()
    server = ThreadingHTTPServer((cfg["host"], int(cfg["port"])), BridgeHandler)
    server.cfg = cfg
    server.models = build_models(cfg)
    print(f"[command-code bridge] listening on http://{cfg['host']}:{cfg['port']}")
    print(f"[command-code bridge] upstream: {cfg['base_url']}")
    print(f"[command-code bridge] default model: {cfg['default_model']}")
    print(f"[command-code bridge] models catalog ({len(server.models)}): "
          + ", ".join(m["id"] for m in server.models))
    print("[command-code bridge] endpoints: POST /v1/chat/completions, POST /v1/responses, "
          "POST /v1/messages (Anthropic), GET /v1/models")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[command-code bridge] shutting down")
        server.shutdown()


if __name__ == "__main__":
    main()
