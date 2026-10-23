import os
import random
import time
from functools import lru_cache
from typing import ClassVar

from langchain_core.rate_limiters import InMemoryRateLimiter

# provider id -> label shown in the UI, env var used as a fallback for the key,
# and the minimum seconds between requests (kept under each free tier's limit).
PROVIDERS = {
    "gemini": {"label": "Google Gemini", "env": ("GOOGLE_API_KEY", "GEMINI_API_KEY"), "min_interval": 6.0,
               "help": "Free key at aistudio.google.com. Best for long videos."},
    "groq":   {"label": "Groq", "env": ("GROQ_API_KEY",), "min_interval": 2.5,
               "help": "Free key at console.groq.com. Fast, but the free tier caps tokens per minute, so short videos work best."},
    "mistral": {"label": "Mistral", "env": ("MISTRAL_API_KEY",), "min_interval": 2.0,
                "help": "Key from console.mistral.ai."},
}
DEFAULT_PROVIDER = "gemini"

# Default model first, then fallbacks tried in order if it has been retired.
# Flash-Lite is the default because the free tier allows far more requests per day
# (about 500) than the full Flash models (about 20), and one video needs ~8+.
# Override the first choice with GEMINI_MODEL in .env.
GEMINI_MODELS = ("gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.6-flash", "gemini-3.8-flash")

MAX_RATE_LIMIT_RETRIES = 5


def env_api_key(provider: str) -> str:
    """Key from the environment / .env, used when none is typed in the UI."""
    for name in PROVIDERS[provider]["env"]:
        if os.getenv(name):
            return os.getenv(name)
    return ""


def _is_rate_limit(exc: Exception) -> bool:
    """True if an exception from any provider means HTTP 429 / quota exceeded."""
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    status = status or getattr(getattr(exc, "response", None), "status_code", None)
    if status == 429:
        return True
    text = str(exc).lower()
    return "429" in text or "rate limit" in text or "resource_exhausted" in text


def _is_model_unavailable(exc: Exception) -> bool:
    """True if the provider says the model id doesn't exist / was retired."""
    text = str(exc).lower()
    return "not_found" in text or "no longer available" in text or "is not found" in text


class _RetryOnRateLimit:
    """Mixin for chat models: retry rate limits, and fall back if a model is retired.

    - The stock clients either don't retry 429s at all (Mistral) or give up
      quickly, and one 429 would otherwise abort the whole pipeline.
    - Free-tier model ids are retired often (Gemini especially), so when the
      configured model is unavailable we move on to the next one in
      `fallback_models` instead of failing.
    """

    fallback_models: ClassVar[tuple] = ()

    def _generate(self, *args, **kwargs):
        tried = {self.model}
        attempt = 0
        while True:
            try:
                return super()._generate(*args, **kwargs)
            except Exception as e:
                if _is_model_unavailable(e):
                    nxt = next((m for m in self.fallback_models if m not in tried), None)
                    if nxt is None:
                        raise
                    print(f"Model '{self.model}' is unavailable, switching to '{nxt}'")
                    tried.add(nxt)
                    object.__setattr__(self, "model", nxt)
                    continue
                if not _is_rate_limit(e) or attempt == MAX_RATE_LIMIT_RETRIES:
                    raise
                wait = min(4 * 2 ** attempt, 60) + random.random()
                attempt += 1
                print(f"Rate limit hit, retrying in {wait:.1f}s "
                      f"(attempt {attempt}/{MAX_RATE_LIMIT_RETRIES})")
                time.sleep(wait)


# One shared limiter per provider, so the summary, extraction and chat calls are
# spaced out together instead of each starting with a fresh burst.
@lru_cache(maxsize=None)
def _limiter(provider: str) -> InMemoryRateLimiter:
    return InMemoryRateLimiter(
        requests_per_second=1 / PROVIDERS[provider]["min_interval"],
        check_every_n_seconds=0.1,
        max_bucket_size=1,
    )


@lru_cache(maxsize=None)
def _chat_class(provider: str):
    # Imported lazily so only the selected provider's package is loaded.
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI as Base
    elif provider == "groq":
        from langchain_groq import ChatGroq as Base
    elif provider == "mistral":
        from langchain_mistralai import ChatMistralAI as Base
    else:
        raise ValueError(f"Unknown provider: {provider!r}")
    fallbacks = GEMINI_MODELS if provider == "gemini" else ()
    # pydantic needs the ClassVar annotation on the dynamically built class too.
    namespace = {"fallback_models": fallbacks, "__annotations__": {"fallback_models": ClassVar[tuple]}}
    return type(f"_{provider.title()}Chat", (_RetryOnRateLimit, Base), namespace)


def get_llm(api_key: str = None, temperature: float = 0.3, provider: str = None, **kwargs):
    # No provider given (e.g. the CLI): use LLM_PROVIDER, else whichever has a key in .env.
    provider = (provider or os.getenv("LLM_PROVIDER")
                or next((p for p in PROVIDERS if env_api_key(p)), DEFAULT_PROVIDER))
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider!r}")

    key = api_key or env_api_key(provider)
    if not key:
        raise RuntimeError(f"No API key for {PROVIDERS[provider]['label']}. Enter it in the sidebar or set it in .env")

    common = dict(temperature=temperature, rate_limiter=_limiter(provider), **kwargs)
    cls = _chat_class(provider)

    if provider == "gemini":
        # The Gemini client has its own retry loop; keep it short so ours does the waiting.
        return cls(model=os.getenv("GEMINI_MODEL", GEMINI_MODELS[0]), google_api_key=key, max_retries=1, **common)
    if provider == "groq":
        return cls(model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"), groq_api_key=key, **common)
    return cls(model=os.getenv("MISTRAL_MODEL", "mistral-small-latest"), mistral_api_key=key, **common)
