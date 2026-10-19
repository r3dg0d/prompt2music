"""Optional local language models for prompt enhancement and lyrics.

Music generation never depends on this module. Providers:

* ``ollama``   — POST {endpoint}/api/generate (default http://127.0.0.1:11434)
* ``llamacpp`` — llama.cpp server, POST {endpoint}/v1/chat/completions
                 (default http://127.0.0.1:8080)
* ``transformers`` — a local text-generation pipeline in the current env

Endpoints must resolve to localhost unless ``llm.allow_remote_endpoint`` is
set, keeping the "no cloud inference" guarantee explicit.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlparse

from prompt2music.config import MusicConfig
from prompt2music.errors import ConfigError, MusicError

ENHANCE_SYSTEM = (
    "You rewrite short music descriptions into one detailed prompt for a text-to-music "
    "model. Keep every genre, mood and instrument the user named. Add arrangement, "
    "production and tempo detail. Reply with the prompt only, under 60 words."
)
LYRICS_SYSTEM = (
    "You write original song lyrics. Use section tags on their own lines such as "
    "[Intro], [Verse 1], [Pre-Chorus], [Chorus], [Verse 2], [Bridge], [Outro]. "
    "Keep lines singable and under 12 words. Reply with the lyrics only."
)
DEFAULT_ENDPOINTS = {"ollama": "http://127.0.0.1:11434", "llamacpp": "http://127.0.0.1:8080"}
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


class LocalLLM:
    def __init__(self, provider: str, endpoint: str, model: str) -> None:
        self.provider = provider
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self._pipe: Any = None

    def _post(self, path: str, payload: dict[str, Any], timeout: float = 300) -> dict[str, Any]:
        req = urllib.request.Request(self.endpoint + path, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - checked
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            raise MusicError(f"{self.provider} request to {self.endpoint} failed: {exc}",
                             hint="is the local LLM server running?") from exc

    def complete(self, system: str, user: str, *, max_tokens: int = 600) -> str:
        if self.provider == "ollama":
            data = self._post("/api/generate", {"model": self.model, "system": system,
                                                "prompt": user, "stream": False,
                                                "options": {"num_predict": max_tokens}})
            return str(data.get("response", ""))
        if self.provider == "llamacpp":
            data = self._post("/v1/chat/completions", {
                "model": self.model or "local",
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}],
                "max_tokens": max_tokens,
            })
            return str(data["choices"][0]["message"]["content"])
        if self.provider == "transformers":
            if self._pipe is None:
                try:
                    from transformers import pipeline
                except ImportError as exc:
                    raise MusicError("transformers is not installed in this environment") from exc
                self._pipe = pipeline("text-generation", model=self.model)
            out = self._pipe([{"role": "system", "content": system},
                              {"role": "user", "content": user}], max_new_tokens=max_tokens)
            generated = out[0]["generated_text"]
            if isinstance(generated, list):
                return str(generated[-1].get("content", ""))
            return str(generated)
        raise ConfigError(f"unknown llm provider {self.provider!r}")


def get_llm(cfg: MusicConfig, *, required: bool = False) -> LocalLLM | None:
    section = cfg["llm"]
    provider = section["provider"]
    if provider == "none":
        if required:
            raise ConfigError(
                "no local language model is configured",
                hint="set llm.provider (ollama, llamacpp or transformers) and llm.model, e.g. "
                "`prompt2music config set llm.provider ollama` and "
                "`prompt2music config set llm.model llama3.2`",
            )
        return None
    endpoint = section["endpoint"] or DEFAULT_ENDPOINTS.get(provider, "")
    if provider != "transformers":
        host = urlparse(endpoint).hostname or ""
        if host not in _LOCAL_HOSTS and not section["allow_remote_endpoint"]:
            raise ConfigError(f"llm endpoint {endpoint} is not local",
                              hint="set llm.allow_remote_endpoint = true to permit it")
    if provider in ("ollama", "transformers") and not section["model"]:
        raise ConfigError(f"llm.model must be set for provider {provider}")
    return LocalLLM(provider, endpoint, section["model"])
