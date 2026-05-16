from __future__ import annotations

import hashlib
import os
from typing import Protocol

import httpx
import numpy as np

from app.core.config import Settings


class EmbeddingProvider(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


class LLMProvider(Protocol):
    async def generate(self, prompt: str) -> str:
        ...


class ProviderError(RuntimeError):
    def __init__(
        self,
        provider: str,
        model: str,
        message: str,
        upstream_status: int | None = None,
    ):
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.upstream_status = upstream_status

    def as_detail(self) -> dict[str, str | int | None]:
        return {
            "provider": self.provider,
            "model": self.model,
            "upstream_status": self.upstream_status,
            "message": str(self),
        }


class OllamaEmbeddingProvider:
    def __init__(self, base_url: str, model: str, dimensions: int):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dimensions = dimensions

    def _fallback_embedding(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = np.frombuffer(digest * ((self.dimensions // len(digest)) + 1), dtype=np.uint8)[: self.dimensions]
        vector = values.astype(np.float32)
        norm = np.linalg.norm(vector) or 1.0
        return (vector / norm).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        try:
            with httpx.Client(timeout=60.0) as client:
                embeddings = []
                for text in texts:
                    response = client.post(
                        f"{self.base_url}/api/embeddings",
                        json={"model": self.model, "prompt": text},
                    )
                    response.raise_for_status()
                    embeddings.append(response.json()["embedding"])
                return embeddings
        except Exception:
            return [self._fallback_embedding(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class OllamaLLMProvider:
    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout_seconds: float,
        force_json: bool = False,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.force_json = force_json

    async def generate(self, prompt: str) -> str:
        try:
            payload: dict = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
            }
            if self.force_json:
                payload["format"] = "json"
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                )
                if response.is_error:
                    detail = _response_error_text(response)
                    hint = (
                        f"Ollama returned 404 for model '{self.model}'. Run `ollama pull {self.model}` "
                        f"and verify `ollama serve` is listening at {self.base_url}."
                        if response.status_code == 404
                        else f"Ollama request failed: {detail}"
                    )
                    raise ProviderError("ollama", self.model, hint, response.status_code)
                return response.json()["response"]
        except httpx.ConnectError as exc:
            raise ProviderError(
                "ollama",
                self.model,
                f"Could not connect to Ollama at {self.base_url}. Start it with `ollama serve`.",
            ) from exc
        except httpx.TimeoutException as exc:
            raise ProviderError(
                "ollama",
                self.model,
                f"Ollama timed out at {self.base_url} after {self.timeout_seconds:.0f}s.",
            ) from exc


class OpenAICompatibleLLMProvider:
    def __init__(self, base_url: str, model: str, api_key: str, temperature: float):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature

    async def generate(self, prompt: str) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(f"{self.base_url}/v1/chat/completions", headers=headers, json=payload)
                if response.is_error:
                    raise ProviderError("openai-compatible", self.model, _response_error_text(response), response.status_code)
                return response.json()["choices"][0]["message"]["content"]
        except httpx.ConnectError as exc:
            raise ProviderError("openai-compatible", self.model, f"Could not connect to {self.base_url}.") from exc
        except httpx.TimeoutException as exc:
            raise ProviderError("openai-compatible", self.model, f"Request timed out for {self.base_url}.") from exc


class GeminiLLMProvider:
    def __init__(self, base_url: str, model: str, api_key: str, temperature: float):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature

    async def generate(self, prompt: str) -> str:
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": self.temperature},
        }
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.base_url}/v1beta/models/{self.model}:generateContent?key={self.api_key}",
                    json=payload,
                )
                if response.is_error:
                    raise ProviderError("gemini", self.model, _response_error_text(response), response.status_code)
                data = response.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
        except httpx.ConnectError as exc:
            raise ProviderError("gemini", self.model, f"Could not connect to {self.base_url}.") from exc
        except httpx.TimeoutException as exc:
            raise ProviderError("gemini", self.model, f"Request timed out for {self.base_url}.") from exc


class HuggingFaceLLMProvider:
    def __init__(self, model: str, api_key: str, temperature: float, max_tokens: int, timeout_seconds: float, force_json: bool = False):
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.force_json = force_json

    async def generate(self, prompt: str) -> str:
        try:
            from huggingface_hub import AsyncInferenceClient
        except ImportError as exc:
            raise ProviderError("huggingface", self.model, "huggingface_hub is not installed. Add it to the environment first.") from exc

        if not self.api_key:
            raise ProviderError("huggingface", self.model, "Missing Hugging Face API token. Set JUDGE_API_KEY or HF_TOKEN.")

        messages = [{"role": "user", "content": prompt}]
        client = AsyncInferenceClient(model=self.model, token=self.api_key, timeout=self.timeout_seconds)
        try:
            kwargs: dict[str, object] = {
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            if self.force_json:
                kwargs["response_format"] = {"type": "json_object"}
            response = await client.chat_completion(**kwargs)
            return response.choices[0].message.content or ""
        except TypeError:
            response = await client.chat_completion(
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content or ""
        except httpx.ConnectError as exc:
            raise ProviderError("huggingface", self.model, "Could not reach Hugging Face Inference API.") from exc
        except httpx.TimeoutException as exc:
            raise ProviderError("huggingface", self.model, f"Request timed out after {self.timeout_seconds:.0f}s.") from exc
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            raise ProviderError("huggingface", self.model, str(exc), status) from exc


def _response_error_text(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error)
        if error:
            return str(error)
    return str(payload)


class LLMProviderFactory:
    def __init__(self, settings: Settings):
        self.settings = settings

    def build_llm_provider(self) -> LLMProvider:
        config = self.settings.providers.llm
        return self._build_provider_from_config(
            provider=config.provider,
            model=config.model,
            base_url=config.base_url,
            api_key_env=config.api_key_env,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout_seconds=config.timeout_seconds,
        )

    def build_judge_provider(self) -> LLMProvider:
        config = self.settings.providers.judge
        return self._build_provider_from_config(
            provider=config.provider,
            model=config.model,
            base_url=config.base_url,
            api_key_env=config.api_key_env,
            temperature=0.0,
            max_tokens=512,
            timeout_seconds=config.timeout_seconds,
            force_json=True,
        )

    def _build_provider_from_config(
        self,
        provider: str,
        model: str,
        base_url: str | None,
        api_key_env: str | None,
        temperature: float,
        max_tokens: int,
        timeout_seconds: float,
        force_json: bool = False,
    ) -> LLMProvider:
        provider = provider.lower()
        if provider == "ollama":
            return OllamaLLMProvider(
                base_url=base_url or "http://localhost:11434",
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
                force_json=force_json,
            )

        api_key = os.getenv(api_key_env or "", "")
        if provider in {"huggingface", "hf"}:
            return HuggingFaceLLMProvider(
                model=model,
                api_key=api_key or os.getenv("HF_TOKEN", ""),
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
                force_json=force_json,
            )
        if provider in {"gemini", "genai"}:
            return GeminiLLMProvider(
                base_url=base_url or "https://generativelanguage.googleapis.com",
                model=model,
                api_key=api_key,
                temperature=temperature,
            )
        return OpenAICompatibleLLMProvider(
            base_url=base_url or "https://api.openai.com",
            model=model,
            api_key=api_key,
            temperature=temperature,
        )

    def build_embedding_provider(self) -> EmbeddingProvider:
        config = self.settings.providers.embeddings
        return OllamaEmbeddingProvider(
            base_url=config.base_url or "http://localhost:11434",
            model=config.model,
            dimensions=config.dimensions,
        )
