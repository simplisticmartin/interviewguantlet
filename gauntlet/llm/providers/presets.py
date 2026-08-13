"""Known model providers.

Most vendors expose an OpenAI-compatible chat completions endpoint, which means one
adapter plus a table of base URLs covers nearly all of them. That is why this file is a
data table rather than twenty classes: adding a provider is a row, not an integration.

**On the model names below.** They are sensible defaults at the time of writing, not
guarantees. Vendors rename and retire models constantly. Every one is overridable with
``GAUNTLET_LLM_INTERVIEW_MODEL`` and ``GAUNTLET_LLM_EVALUATION_MODEL``, and if a call
fails with "model not found" that is the first thing to check against the vendor's docs.

**On capability flags.** Gauntlet needs structured JSON back from every call. Providers
differ in how they support that, and some reject a ``response_format`` parameter they do
not implement, which is a hard error rather than a graceful ignore. ``supports_json_mode``
records that, and the adapter falls back to schema-in-the-prompt plus validation, which
works everywhere.

**On embeddings.** Several strong chat providers have no embedding endpoint at all.
``embedding_model = None`` records that, and the embedder resolves independently, so you
can run interviews on DeepSeek while embedding with something else.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProviderPreset:
    """Everything needed to talk to one vendor."""

    key: str
    label: str
    api_key_env: str
    interview_model: str
    evaluation_model: str
    base_url: str | None = None  # None means the OpenAI SDK default
    embedding_model: str | None = None
    supports_json_mode: bool = True
    supports_tools: bool = True
    requires_key: bool = True
    docs: str = ""
    notes: str = ""

    @property
    def supports_embeddings(self) -> bool:
        return self.embedding_model is not None


def _p(**kwargs: object) -> ProviderPreset:
    return ProviderPreset(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Native adapters. These do not speak the OpenAI wire format.
# ---------------------------------------------------------------------------

NATIVE_PRESETS: tuple[ProviderPreset, ...] = (
    _p(
        key="anthropic",
        label="Anthropic (Claude)",
        api_key_env="ANTHROPIC_API_KEY",
        interview_model="claude-opus-4-6",
        evaluation_model="claude-sonnet-4-5",
        docs="https://docs.anthropic.com/en/api/overview",
        notes=(
            "Native adapter. Structured output uses forced tool use, so the schema is "
            "enforced by the API rather than requested in the prompt. No embedding "
            "endpoint, so embeddings resolve separately."
        ),
    ),
)

# ---------------------------------------------------------------------------
# OpenAI-compatible providers. One adapter serves all of these.
# ---------------------------------------------------------------------------

COMPATIBLE_PRESETS: tuple[ProviderPreset, ...] = (
    # --- First party frontier labs -----------------------------------------
    _p(
        key="openai",
        label="OpenAI",
        api_key_env="OPENAI_API_KEY",
        interview_model="gpt-4.1",
        evaluation_model="gpt-4.1-mini",
        embedding_model="text-embedding-3-small",
        docs="https://platform.openai.com/docs/api-reference",
    ),
    _p(
        key="gemini",
        label="Google Gemini",
        api_key_env="GEMINI_API_KEY",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        interview_model="gemini-2.5-pro",
        evaluation_model="gemini-2.5-flash",
        embedding_model="text-embedding-004",
        docs="https://ai.google.dev/gemini-api/docs/openai",
        notes=(
            "Uses Google's OpenAI-compatibility layer, which is the pragmatic choice. "
            "The native Gemini API additionally supports responseSchema for stricter "
            "structured output, which would be a worthwhile native adapter later. "
            "Get a key from Google AI Studio, not the Gemini consumer app."
        ),
    ),
    _p(
        key="deepseek",
        label="DeepSeek",
        api_key_env="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com/v1",
        interview_model="deepseek-chat",
        evaluation_model="deepseek-chat",
        embedding_model=None,
        docs="https://api-docs.deepseek.com/",
        notes=(
            "Very strong price to performance. deepseek-reasoner is the reasoning model "
            "and is a good fit for the judges, at higher latency. No embeddings."
        ),
    ),
    _p(
        key="xai",
        label="xAI (Grok)",
        api_key_env="XAI_API_KEY",
        base_url="https://api.x.ai/v1",
        interview_model="grok-4",
        evaluation_model="grok-3-mini",
        embedding_model=None,
        docs="https://docs.x.ai/",
        notes="Console at console.x.ai. The grok.com chat app is separate and has no API.",
    ),
    _p(
        key="moonshot",
        label="Moonshot AI (Kimi)",
        api_key_env="MOONSHOT_API_KEY",
        base_url="https://api.moonshot.ai/v1",
        interview_model="kimi-k2-0905-preview",
        evaluation_model="moonshot-v1-32k",
        embedding_model=None,
        docs="https://platform.moonshot.ai/docs",
        notes=(
            "Use api.moonshot.cn instead if your account is on the China platform. "
            "Kimi K2 is the strong open-weight model; moonshot-v1-* are the stable "
            "long-context aliases."
        ),
    ),
    _p(
        key="qwen",
        label="Alibaba Qwen (DashScope)",
        api_key_env="DASHSCOPE_API_KEY",
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        interview_model="qwen-max",
        evaluation_model="qwen-plus",
        embedding_model="text-embedding-v3",
        docs="https://www.alibabacloud.com/help/en/model-studio/",
        notes=(
            "This is the international endpoint. Mainland China accounts use "
            "https://dashscope.aliyuncs.com/compatible-mode/v1 instead. Qwen also has "
            "embeddings, which most non-OpenAI providers do not."
        ),
    ),
    _p(
        key="mistral",
        label="Mistral AI",
        api_key_env="MISTRAL_API_KEY",
        base_url="https://api.mistral.ai/v1",
        interview_model="mistral-large-latest",
        evaluation_model="mistral-small-latest",
        embedding_model="mistral-embed",
        docs="https://docs.mistral.ai/",
    ),
    _p(
        key="cohere",
        label="Cohere",
        api_key_env="COHERE_API_KEY",
        base_url="https://api.cohere.ai/compatibility/v1",
        interview_model="command-a-03-2025",
        evaluation_model="command-r-plus",
        embedding_model="embed-v4.0",
        docs="https://docs.cohere.com/docs/compatibility-api",
        notes="Cohere's OpenAI compatibility endpoint. Also has a strong native reranker.",
    ),
    # --- Fast inference hosts for open weight models ------------------------
    _p(
        key="groq",
        label="Groq",
        api_key_env="GROQ_API_KEY",
        base_url="https://api.groq.com/openai/v1",
        interview_model="llama-3.3-70b-versatile",
        evaluation_model="llama-3.1-8b-instant",
        embedding_model=None,
        docs="https://console.groq.com/docs",
        notes=(
            "Extremely fast, which matters here because a turn makes several calls. "
            "A strong choice for the classifier and router specifically."
        ),
    ),
    _p(
        key="cerebras",
        label="Cerebras",
        api_key_env="CEREBRAS_API_KEY",
        base_url="https://api.cerebras.ai/v1",
        interview_model="llama-3.3-70b",
        evaluation_model="llama3.1-8b",
        embedding_model=None,
        docs="https://inference-docs.cerebras.ai/",
        notes="Also optimised for very low latency inference.",
    ),
    _p(
        key="together",
        label="Together AI",
        api_key_env="TOGETHER_API_KEY",
        base_url="https://api.together.xyz/v1",
        interview_model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        evaluation_model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        embedding_model="BAAI/bge-large-en-v1.5",
        docs="https://docs.together.ai/",
        notes="Hosts most open weight models, including Llama, Qwen and DeepSeek.",
    ),
    _p(
        key="fireworks",
        label="Fireworks AI",
        api_key_env="FIREWORKS_API_KEY",
        base_url="https://api.fireworks.ai/inference/v1",
        interview_model="accounts/fireworks/models/llama-v3p3-70b-instruct",
        evaluation_model="accounts/fireworks/models/llama-v3p1-8b-instruct",
        embedding_model="nomic-ai/nomic-embed-text-v1.5",
        docs="https://docs.fireworks.ai/",
        notes="Supports grammar-constrained output, which is stronger than JSON mode.",
    ),
    _p(
        key="deepinfra",
        label="DeepInfra",
        api_key_env="DEEPINFRA_API_KEY",
        base_url="https://api.deepinfra.com/v1/openai",
        interview_model="meta-llama/Meta-Llama-3.1-70B-Instruct",
        evaluation_model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        embedding_model="BAAI/bge-large-en-v1.5",
        docs="https://deepinfra.com/docs",
    ),
    _p(
        key="nebius",
        label="Nebius AI Studio",
        api_key_env="NEBIUS_API_KEY",
        base_url="https://api.studio.nebius.ai/v1",
        interview_model="meta-llama/Meta-Llama-3.1-70B-Instruct",
        evaluation_model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        embedding_model="BAAI/bge-en-icl",
        docs="https://docs.nebius.com/studio/inference/",
    ),
    _p(
        key="sambanova",
        label="SambaNova",
        api_key_env="SAMBANOVA_API_KEY",
        base_url="https://api.sambanova.ai/v1",
        interview_model="Meta-Llama-3.3-70B-Instruct",
        evaluation_model="Meta-Llama-3.1-8B-Instruct",
        embedding_model=None,
        docs="https://docs.sambanova.ai/",
    ),
    _p(
        key="hyperbolic",
        label="Hyperbolic",
        api_key_env="HYPERBOLIC_API_KEY",
        base_url="https://api.hyperbolic.xyz/v1",
        interview_model="meta-llama/Meta-Llama-3.1-70B-Instruct",
        evaluation_model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        embedding_model=None,
        docs="https://docs.hyperbolic.xyz/",
    ),
    # --- Routers and aggregators -------------------------------------------
    _p(
        key="openrouter",
        label="OpenRouter",
        api_key_env="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
        interview_model="anthropic/claude-sonnet-4",
        evaluation_model="meta-llama/llama-3.3-70b-instruct",
        embedding_model=None,
        docs="https://openrouter.ai/docs",
        notes=(
            "One key, hundreds of models across every major vendor, selected by the "
            "model string. The single easiest way to try many providers, and the best "
            "answer if you want breadth without managing accounts."
        ),
    ),
    _p(
        key="perplexity",
        label="Perplexity",
        api_key_env="PERPLEXITY_API_KEY",
        base_url="https://api.perplexity.ai",
        interview_model="sonar-pro",
        evaluation_model="sonar",
        embedding_model=None,
        docs="https://docs.perplexity.ai/",
        notes="Search-grounded models. Poor fit for grading, which must not search.",
    ),
    _p(
        key="github",
        label="GitHub Models",
        api_key_env="GITHUB_TOKEN",
        base_url="https://models.inference.ai.azure.com",
        interview_model="gpt-4o",
        evaluation_model="gpt-4o-mini",
        embedding_model="text-embedding-3-small",
        docs="https://docs.github.com/en/github-models",
        notes="Free tier with a GitHub token, heavily rate limited. Good for a first look.",
    ),
    _p(
        key="azure",
        label="Azure OpenAI",
        api_key_env="AZURE_OPENAI_API_KEY",
        base_url="",  # must be supplied: https://<resource>.openai.azure.com/openai/v1
        interview_model="gpt-4.1",
        evaluation_model="gpt-4.1-mini",
        embedding_model="text-embedding-3-small",
        docs="https://learn.microsoft.com/azure/ai-services/openai/",
        notes=(
            "Set GAUNTLET_LLM_BASE_URL to your resource endpoint. Model names are your "
            "deployment names, not the upstream model names."
        ),
    ),
    # --- Local and self hosted ----------------------------------------------
    _p(
        key="ollama",
        label="Ollama (local)",
        api_key_env="OLLAMA_API_KEY",
        base_url="http://localhost:11434/v1",
        interview_model="qwen2.5:14b",
        evaluation_model="llama3.1:8b",
        embedding_model="nomic-embed-text",
        requires_key=False,
        docs="https://ollama.com/blog/openai-compatibility",
        notes=(
            "Runs entirely on your machine, no key and no data leaving the host. "
            "Pull models first, for example `ollama pull qwen2.5:14b`. Small local "
            "models are noticeably weaker at rubric grading, so check the eval "
            "benchmark before trusting the scores."
        ),
    ),
    _p(
        key="lmstudio",
        label="LM Studio (local)",
        api_key_env="LMSTUDIO_API_KEY",
        base_url="http://localhost:1234/v1",
        interview_model="local-model",
        evaluation_model="local-model",
        embedding_model=None,
        requires_key=False,
        docs="https://lmstudio.ai/docs/app/api",
        notes="Start the local server in LM Studio first. Model name is whatever is loaded.",
    ),
    _p(
        key="vllm",
        label="vLLM (self hosted)",
        api_key_env="VLLM_API_KEY",
        base_url="http://localhost:8000/v1",
        interview_model="meta-llama/Meta-Llama-3.1-70B-Instruct",
        evaluation_model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        embedding_model=None,
        requires_key=False,
        docs="https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html",
        notes="The usual choice for serving open weights at scale on your own GPUs.",
    ),
    _p(
        key="llamacpp",
        label="llama.cpp server (local)",
        api_key_env="LLAMACPP_API_KEY",
        base_url="http://localhost:8080/v1",
        interview_model="local-model",
        evaluation_model="local-model",
        embedding_model=None,
        requires_key=False,
        docs="https://github.com/ggml-org/llama.cpp/tree/master/tools/server",
        notes="Runs GGUF models on CPU or modest GPUs.",
    ),
    # --- Escape hatch ---------------------------------------------------------
    _p(
        key="custom",
        label="Custom OpenAI-compatible endpoint",
        api_key_env="GAUNTLET_LLM_API_KEY",
        base_url="",  # supplied by GAUNTLET_LLM_BASE_URL
        interview_model="",
        evaluation_model="",
        embedding_model=None,
        requires_key=False,
        docs="",
        notes=(
            "For anything not listed. Set GAUNTLET_LLM_BASE_URL, "
            "GAUNTLET_LLM_INTERVIEW_MODEL and GAUNTLET_LLM_EVALUATION_MODEL. "
            "Any gateway speaking the OpenAI chat completions format will work, "
            "including LiteLLM, vLLM behind a proxy, or an internal gateway."
        ),
    ),
)

# Providers that need no network and no key at all.
OFFLINE_PRESET = _p(
    key="scripted",
    label="Offline deterministic engine",
    api_key_env="",
    interview_model="scripted-heuristic-v1",
    evaluation_model="scripted-heuristic-v1",
    requires_key=False,
    notes=(
        "Rule based, no network. Runs a full interview with no key configured and is "
        "the baseline the real graders are measured against in the benchmark."
    ),
)

ALL_PRESETS: tuple[ProviderPreset, ...] = (
    *NATIVE_PRESETS,
    *COMPATIBLE_PRESETS,
    OFFLINE_PRESET,
)

PRESET_INDEX: dict[str, ProviderPreset] = {preset.key: preset for preset in ALL_PRESETS}
COMPATIBLE_KEYS: frozenset[str] = frozenset(preset.key for preset in COMPATIBLE_PRESETS)
NATIVE_KEYS: frozenset[str] = frozenset(preset.key for preset in NATIVE_PRESETS)


def get_preset(key: str) -> ProviderPreset | None:
    return PRESET_INDEX.get(key.strip().lower())


def is_openai_compatible(key: str) -> bool:
    return key.strip().lower() in COMPATIBLE_KEYS


def provider_keys() -> list[str]:
    return sorted(PRESET_INDEX)


def embedding_capable_keys() -> list[str]:
    """Providers that can serve embeddings, for picking an embedding backend."""
    return sorted(preset.key for preset in ALL_PRESETS if preset.supports_embeddings)
