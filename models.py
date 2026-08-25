"""command-code model catalog.

Source of truth for GET /v1/models and for mapping client-supplied model ids
(including legacy and gateway-specific variants) to the canonical id that the
/alpha/generate endpoint expects.

Data transcribed from the command-code web app's model registry (Aug 2026).
Each entry's `id` doubles as the request id sent upstream; `ALIASES` below maps
everything else (dated Anthropic ids, lowercase gateway slugs) onto it.

Fields:
    id                canonical id — shown in /v1/models AND sent upstream
    label             display name in the command-code UI
    vendor            owner shown as `owned_by` in /v1/models
    description       UI marketing line, verbatim
    context_window    input context window in tokens (None = unspecified)
    max_output_tokens cap on output tokens (None = unspecified)
    reasoning         whether the model exposes reasoning/thinking output
    reasoning_efforts supported reasoning effort levels (None = not selectable)
    input_modalities  accepted input types ("text", "image")
    spec              upstream API flavor the CLI uses ("chatComplete"/"responses");
                      informational only — this bridge sends the same native call
    free              zero-cost promo model
    hidden            excluded from /v1/models (mirrors the CLI picker) but
                      still callable when requested explicitly
    notice            upstream status banner shown in the UI
"""

# ---------------------------------------------------------------------------
# Catalog (order mirrors the web app's registry)
# ---------------------------------------------------------------------------

_ALL_EFFORTS = ["low", "medium", "high", "xhigh", "max"]
_HIGH_XHIGH = ["low", "medium", "high", "xhigh"]
_HIGH_MAX = ["high", "max"]
_TEXT_IMAGE = ["text", "image"]

_UPSTREAM_ISSUE_NOTICE = ("{name} is going through upstream inference issues at "
                          "{vendor}, not Command Code. Expect slower responses "
                          "and occasional rate limits.")


def _model(id, label, vendor, description, *, context_window=None,
           max_output_tokens=None, reasoning=False, efforts=None,
           modalities=_TEXT_IMAGE, spec="chatComplete", free=False,
           hidden=False, notice=None):
    return {
        "id": id,
        "label": label,
        "vendor": vendor,
        "description": description,
        "context_window": context_window,
        "max_output_tokens": max_output_tokens,
        "reasoning": reasoning,
        "reasoning_efforts": list(efforts) if efforts else None,
        "input_modalities": list(modalities),
        "spec": spec,
        "free": free,
        "hidden": hidden,
        "notice": notice,
    }


MODELS = [
    # -- Anthropic ------------------------------------------------------------
    _model("claude-sonnet-5", "Claude Sonnet 5", "Anthropic",
           "best combo of speed & intelligence (recommended)",
           context_window=1_000_000, reasoning=True, efforts=_ALL_EFFORTS),
    _model("claude-sonnet-4-6", "Claude Sonnet 4.6", "Anthropic",
           "prev Sonnet, still fast & capable",
           context_window=1_000_000, efforts=_ALL_EFFORTS),
    _model("claude-fable-5", "Claude Fable 5", "Anthropic",
           "most capable for demanding reasoning & long-horizon agents",
           context_window=1_000_000, reasoning=True, efforts=_ALL_EFFORTS),
    _model("claude-opus-5", "Claude Opus 5", "Anthropic",
           "most intelligent Opus for agents and coding",
           context_window=1_000_000, reasoning=True, efforts=_ALL_EFFORTS),
    _model("claude-opus-4-8", "Claude Opus 4.8", "Anthropic",
           "prev flagship, still strong for agents and coding",
           context_window=1_000_000, reasoning=True, efforts=_ALL_EFFORTS),
    _model("claude-opus-4-7", "Claude Opus 4.7", "Anthropic",
           "older Opus, still strong for agents and coding",
           context_window=1_000_000, reasoning=True, efforts=_ALL_EFFORTS),
    _model("claude-haiku-4-5-20251001", "Claude Haiku 4.5", "Anthropic",
           "fastest & most compact, great for quick tasks",
           context_window=200_000),

    # -- OpenAI (Responses-API models) ----------------------------------------
    _model("gpt-5.6-sol", "GPT-5.6 Sol", "OpenAI",
           "frontier model for complex professional work",
           context_window=1_050_000, reasoning=True, efforts=_ALL_EFFORTS,
           spec="responses"),
    _model("gpt-5.6-terra", "GPT-5.6 Terra", "OpenAI",
           "balances intelligence and cost",
           context_window=1_050_000, reasoning=True, efforts=_ALL_EFFORTS,
           spec="responses"),
    _model("gpt-5.6-luna", "GPT-5.6 Luna", "OpenAI",
           "optimized for cost-sensitive workloads",
           context_window=1_050_000, reasoning=True, efforts=_ALL_EFFORTS,
           spec="responses"),
    _model("gpt-5.5", "GPT-5.5", "OpenAI",
           "latest frontier model for general complex work",
           context_window=400_000, reasoning=True, efforts=_HIGH_XHIGH,
           spec="responses"),
    _model("gpt-5.4", "GPT-5.4", "OpenAI",
           "frontier model for general complex work",
           context_window=400_000, reasoning=True, efforts=_HIGH_XHIGH,
           spec="responses"),
    _model("gpt-5.3-codex", "GPT-5.3 Codex", "OpenAI",
           "frontier coding model",
           context_window=400_000, reasoning=True, efforts=_HIGH_XHIGH,
           spec="responses"),
    _model("gpt-5.4-mini", "GPT-5.4 Mini", "OpenAI",
           "fast, cost-effective model for everyday tasks",
           context_window=400_000, reasoning=True,
           efforts=["low", "medium", "high"], spec="responses"),

    # -- DeepSeek ----------------------------------------------------------------
    _model("MiniMaxAI/MiniMax-M3-Free", "MiniMax M3", "Open Source",
           "frontier coding, agents & native multimodality",
           context_window=1_000_000, reasoning=True, free=True, hidden=True),
    _model("deepseek/deepseek-v4-pro", "DeepSeek V4 Pro (latest)", "Open Source",
           "hybrid-attention long-context reasoning",
           context_window=1_000_000, reasoning=True, efforts=_HIGH_MAX,
           modalities=["text"]),
    _model("deepseek/deepseek-v4-flash", "DeepSeek V4 Flash (latest)",
           "Open Source", "fast hybrid-attention reasoning",
           context_window=1_000_000, reasoning=True, efforts=_HIGH_MAX,
           modalities=["text"]),
    _model("deepseek/deepseek-v4-flash-vision-exp",
           "DeepSeek V4 Flash Vision (exp)", "Open Source",
           "fast hybrid-attention reasoning with vision",
           context_window=1_000_000, reasoning=True, efforts=_HIGH_MAX),

    # -- Moonshot -----------------------------------------------------------------
    _model("moonshotai/Kimi-K3", "Kimi K3", "Open Source",
           "long-horizon coding & knowledge work with 1M context",
           context_window=1_000_000, reasoning=True),
    _model("moonshotai/Kimi-K2.7-Code", "Kimi K2.7 Code", "Open Source",
           "improved long-horizon coding with vision",
           context_window=256_000, reasoning=True),
    _model("moonshotai/Kimi-K2.7-Code-Highspeed", "Kimi K2.7 Code HighSpeed",
           "Open Source", "high-speed long-horizon coding with vision",
           context_window=262_000, reasoning=True),
    _model("moonshotai/Kimi-K2.6", "Kimi K2.6", "Open Source",
           "long-horizon coding with vision", context_window=256_000),
    _model("moonshotai/Kimi-K2.5", "Kimi K2.5", "Open Source",
           "multimodal frontend coding", context_window=256_000),

    # -- Zhipu (GLM) ----------------------------------------------------------------
    _model("zai-org/GLM-5.3", "GLM-5.3", "Open Source",
           "frontier coding with emergent cyber capabilities",
           context_window=1_000_000, reasoning=True,
           efforts=["low", "high", "max"], modalities=["text"]),
    _model("zai-org/GLM-5.2", "GLM-5.2", "Open Source",
           "powerful coding with 1M context and long-horizon tasks",
           context_window=1_000_000, reasoning=True, efforts=_HIGH_MAX,
           modalities=["text"]),
    _model("zai-org/GLM-5.2-Fast", "GLM-5.2 Fast", "Open Source",
           "high-throughput GLM-5.2 with 1M context",
           context_window=1_000_000, modalities=["text"]),
    _model("zai-org/GLM-5.1", "GLM-5.1", "Open Source",
           "long-horizon autonomous coding agent", modalities=["text"]),
    _model("zai-org/GLM-5", "GLM-5", "Open Source",
           "multi-mode thinking & long-range planning",
           context_window=200_000, modalities=["text"]),

    # -- MiniMax ----------------------------------------------------------------------
    _model("MiniMaxAI/MiniMax-M3", "MiniMax M3", "Open Source",
           "frontier coding, agents & native multimodality",
           context_window=1_000_000, reasoning=True),
    _model("MiniMaxAI/MiniMax-M2.7", "MiniMax M2.7", "Open Source",
           "end-to-end software engineering agent", modalities=["text"]),
    _model("MiniMaxAI/MiniMax-M2.5", "MiniMax M2.5", "Open Source",
           "cross-platform full-stack agentic dev",
           context_window=200_000, modalities=["text"]),

    # -- Xiaomi -------------------------------------------------------------------------
    _model("xiaomi/mimo-v2.5-pro", "MiMo V2.5 Pro", "Open Source",
           "high-capability long-context agentic coding",
           context_window=1_000_000, modalities=["text"]),
    _model("xiaomi/mimo-v2.5", "MiMo V2.5", "Open Source",
           "efficient long-context agentic coding",
           context_window=1_000_000),

    # -- Alibaba (Qwen) --------------------------------------------------------------------
    _model("Qwen/Qwen3.8-Max", "Qwen 3.8 Max", "Open Source",
           "autonomous long-horizon coding & professional work",
           context_window=1_000_000, reasoning=True,
           efforts=["low", "medium", "xhigh"]),
    _model("Qwen/Qwen3.8-27B", "Qwen 3.8 27B", "Open Source",
           "compact vision-language coding & agentic work",
           context_window=262_144, max_output_tokens=32_768, reasoning=True,
           efforts=["low", "medium", "xhigh"]),
    _model("Qwen/Qwen3.7-Max", "Qwen 3.7 Max", "Open Source",
           "frontier coding & long-horizon agent execution",
           context_window=1_000_000, reasoning=True, modalities=["text"]),
    _model("Qwen/Qwen3.7-Plus", "Qwen 3.7 Plus", "Open Source",
           "agentic coding & reasoning at lower cost",
           context_window=1_000_000, reasoning=True),
    _model("Qwen/Qwen3.7-Flash", "Qwen 3.7 Flash", "Open Source",
           "fast low-cost agentic coding & reasoning",
           context_window=1_000_000, reasoning=True),
    _model("Qwen/Qwen3.6-Max-Preview", "Qwen 3.6 Max Preview", "Open Source",
           "vibe coding & efficient agent execution", reasoning=True,
           modalities=["text"]),
    _model("Qwen/Qwen3.6-Plus", "Qwen 3.6 Plus", "Open Source",
           "agentic coding & reasoning", reasoning=True),

    # -- StepFun ------------------------------------------------------------------------------
    _model("stepfun/Step-3.7-Flash", "Step 3.7 Flash", "Open Source",
           "multimodal sparse-MoE reasoning", context_window=256_000,
           reasoning=True),
    _model("stepfun/Step-3.5-Flash", "Step 3.5 Flash", "Open Source",
           "fast sparse-MoE agentic reasoning", context_window=1_000_000,
           reasoning=True, modalities=["text"]),

    # -- Tencent ----------------------------------------------------------------------------------
    _model("tencent/Hy3", "Tencent Hy3 (Free)", "Open Source",
           "sparse-MoE reasoning & agentic tool use",
           context_window=262_144, reasoning=True, free=True, hidden=True,
           modalities=["text"]),
    _model("tencent/hy3-paid", "Tencent Hy3", "Open Source",
           "sparse-MoE reasoning & agentic tool use",
           context_window=262_144, reasoning=True, modalities=["text"]),

    # -- Google ---------------------------------------------------------------------------------------
    _model("google/gemini-3.7-flash", "Gemini 3.7 Flash", "Google",
           "higher-quality coding & agentic workflows, fewer tokens",
           context_window=1_048_576, reasoning=True,
           efforts=["low", "medium", "high"]),
    _model("google/gemini-3.6-flash", "Gemini 3.6 Flash", "Google",
           "previous Gemini Flash, still fast & capable",
           context_window=1_000_000, reasoning=True,
           efforts=["low", "medium", "high"]),
    _model("google/gemini-3.5-flash", "Gemini 3.5 Flash", "Google",
           "Pro-level coding proficiency, parallel agentic execution",
           context_window=1_000_000, reasoning=True,
           efforts=["low", "medium", "high"]),
    _model("google/gemini-3.5-flash-lite", "Gemini 3.5 Flash Lite", "Google",
           "upgraded agentic capabilities, ideal for subagents",
           context_window=1_000_000, reasoning=True,
           efforts=["low", "medium", "high"]),
    _model("google/gemini-3.1-flash-lite", "Gemini 3.1 Flash Lite", "Google",
           "high-volume workhorse model with implicit caching",
           context_window=1_000_000, reasoning=True,
           efforts=["low", "medium", "high"]),

    # -- Others -----------------------------------------------------------------------------------------
    _model("sakana/fugu-ultra", "Fugu Ultra", "Sakana",
           "multi-agent orchestration across frontier models",
           context_window=1_000_000, reasoning=True,
           efforts=["high", "xhigh"]),
    _model("nvidia/nemotron-3-ultra-550b-a55b", "Nemotron 3 Ultra",
           "Open Source", "open reasoning model for long-horizon autonomous agents",
           context_window=1_000_000, reasoning=True, modalities=["text"]),
    _model("thinkingmachines/inkling", "Inkling", "Open Source",
           "multimodal MoE reasoning", context_window=256_000, reasoning=True),
    _model("thinkingmachines/inkling-small", "Inkling Small", "Open Source",
           "lightweight MoE reasoning at lower cost and latency",
           context_window=1_000_000, reasoning=True),
    _model("stealth/ox-alpha", "Ox Alpha", "Stealth",
           "long-horizon coding, agentic work & visual context",
           context_window=1_048_576, max_output_tokens=131_072, reasoning=True,
           efforts=["low", "high", "max"], free=True),
    _model("poolside/laguna-s-2.1-free", "Laguna S 2.1", "Open Source",
           "open-weight agentic coding and long-horizon work",
           context_window=256_000, max_output_tokens=32_768, reasoning=True,
           free=True, modalities=["text"]),
    _model("inclusionai/ling-3.0-flash-free", "Ling 3.0 Flash", "Open Source",
           "fast lightweight-MoE coding & agentic work",
           context_window=256_000, max_output_tokens=32_768, reasoning=True,
           free=True, modalities=["text"]),
    _model("meta/muse-spark-1.1", "Muse Spark 1.1", "Meta",
           "agentic performance, tool use, and computer use",
           context_window=1_048_576, reasoning=True),
    _model("meta/muse-spark-1.2", "Muse Spark 1.2", "Meta",
           "coding-optimized for agentic workflows and large codebases",
           context_window=1_048_576, reasoning=True,
           notice=_UPSTREAM_ISSUE_NOTICE),
    _model("meta/muse-spark-1.2-contributor", "Muse Spark 1.2 Contributor",
           "Meta", "Muse Spark 1.2 at ~95% off",
           context_window=1_048_576, reasoning=True,
           notice=_UPSTREAM_ISSUE_NOTICE),
    _model("xai/grok-4.5", "Grok 4.5", "xAI",
           "smartest model for coding, agentic tasks, knowledge work",
           context_window=500_000, reasoning=True,
           efforts=["low", "medium", "high"]),
    _model("xai/grok-4.6", "Grok 4.6", "xAI",
           "frontier performance on coding, knowledge work, and STEM",
           context_window=500_000, reasoning=True, efforts=_HIGH_XHIGH,
           modalities=["text"]),
]

# ---------------------------------------------------------------------------
# Aliases: client-supplied id -> canonical upstream id
# ---------------------------------------------------------------------------

# Dated/renamed Anthropic ids still emitted by older SDKs and config files
# (same mapping the CLI itself applies).
_LEGACY_ALIASES = {
    "claude-sonnet-4-20250514": "claude-sonnet-4-6",
    "claude-sonnet-4-5-20250929": "claude-sonnet-4-6",
    "claude-opus-4-5-20251101": "claude-opus-4-7",
    "claude-opus-4-6": "claude-opus-4-7",
    "claude-haiku-4-5": "claude-haiku-4-5-20251001",
}

# Gateway-specific slugs from the routing table, for clients that copied ids
# from other providers. Only ids that differ from their canonical form are
# listed; anything unknown passes through unchanged.
_GATEWAY_SLUGS = {
    "zai/glm-5": "zai-org/GLM-5",
    "zai-org/glm-5.3": "zai-org/GLM-5.3",
    "glm-5.2": "zai-org/GLM-5.2",
    "zai/glm-5.2": "zai-org/GLM-5.2",
    "zai/glm-5.2-fast": "zai-org/GLM-5.2-Fast",
    "zai/glm-5.1": "zai-org/GLM-5.1",
    "moonshotai/kimi-k3": "moonshotai/Kimi-K3",
    "moonshotai/kimi-k2.5": "moonshotai/Kimi-K2.5",
    "moonshotai/kimi-k2.6": "moonshotai/Kimi-K2.6",
    "kimi-k2.7-code": "moonshotai/Kimi-K2.7-Code",
    "moonshotai/kimi-k2.7-code": "moonshotai/Kimi-K2.7-Code",
    "moonshotai/kimi-k2.7-code-highspeed": "moonshotai/Kimi-K2.7-Code-Highspeed",
    "minimax/minimax-m3": "MiniMaxAI/MiniMax-M3",
    "minimax/minimax-m2.5": "MiniMaxAI/MiniMax-M2.5",
    "minimax/minimax-m2.7": "MiniMaxAI/MiniMax-M2.7",
    "morph-dsv4flash": "deepseek/deepseek-v4-flash",
    "alibaba/qwen-3.6-max-preview": "Qwen/Qwen3.6-Max-Preview",
    "qwen3.6-plus": "Qwen/Qwen3.6-Plus",
    "alibaba/qwen3.6-plus": "Qwen/Qwen3.6-Plus",
    "qwen3.7-max": "Qwen/Qwen3.7-Max",
    "alibaba/qwen3.7-max": "Qwen/Qwen3.7-Max",
    "qwen3.7-plus": "Qwen/Qwen3.7-Plus",
    "alibaba/qwen3.7-plus": "Qwen/Qwen3.7-Plus",
    "qwen3.8-max": "Qwen/Qwen3.8-Max",
    "alibaba/qwen3.8-max": "Qwen/Qwen3.8-Max",
    "alibaba/qwen3.8-27b": "Qwen/Qwen3.8-27B",
    "alibaba/qwen3.7-flash": "Qwen/Qwen3.7-Flash",
    "stepfun/step-3.7-flash": "stepfun/Step-3.7-Flash",
    "stepfun/step-3.5-flash": "stepfun/Step-3.5-Flash",
    "openai/gpt-5.6-sol": "gpt-5.6-sol",
    "openai/gpt-5.6-terra": "gpt-5.6-terra",
    "openai/gpt-5.6-luna": "gpt-5.6-luna",
    "tencent/hy3:free": "tencent/Hy3",
    "tencent/hy3": "tencent/hy3-paid",
}

ALIASES = {**_LEGACY_ALIASES, **_GATEWAY_SLUGS}

_BY_ID = {m["id"]: m for m in MODELS}


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

def resolve(model_id):
    """Map a client-supplied id to the canonical upstream id.

    Unknown or empty ids pass through unchanged, so brand-new upstream models
    keep working without a catalog update.
    """
    if not model_id:
        return model_id
    return ALIASES.get(model_id, model_id)


def get(model_id):
    """Return the catalog entry for a client-supplied id, or None."""
    return _BY_ID.get(resolve(model_id)) if model_id else None


def visible_models():
    """Catalog entries shown in the model picker (hidden ones excluded)."""
    return [m for m in MODELS if not m["hidden"]]


def openai_catalog():
    """Build the /v1/models payload from the visible catalog."""
    return [
        {"id": m["id"], "object": "model", "created": 0,
         "owned_by": m["vendor"]}
        for m in visible_models()
    ]
