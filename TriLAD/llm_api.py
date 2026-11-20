from __future__ import annotations

import json
import os
import random
import time
from typing import Any

import requests
from requests.exceptions import ChunkedEncodingError, ConnectionError, ReadTimeout, SSLError, Timeout


_RETRYABLE_EXC = (
    SSLError,
    ConnectionError,
    Timeout,
    ReadTimeout,
    ChunkedEncodingError,
    json.JSONDecodeError,
    ValueError,
)


def _is_retryable_error(err: Exception) -> bool:
    if isinstance(err, _RETRYABLE_EXC):
        return True
    if isinstance(err, requests.HTTPError):
        code = err.response.status_code if err.response is not None else None
        return code in {408, 429, 500, 502, 503, 504}
    return False


class OpenAICompatibleClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "qwen/qwen3-vl-30b-a3b-instruct",
        timeout: int = 120,
        max_retries: int | None = None,
        site_url: str | None = None,
        app_name: str | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
        self.base_url = (
            base_url
            or os.getenv("OPENROUTER_BASE_URL", "")
            or os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
        ).rstrip("/")
        self.model = model
        self.timeout = timeout
        env_retries = os.getenv("OPENROUTER_MAX_RETRIES", "")
        if max_retries is None:
            self.max_retries = int(env_retries) if env_retries.strip().isdigit() else 8
        else:
            self.max_retries = max(1, int(max_retries))
        self.site_url = site_url or os.getenv("OPENROUTER_SITE_URL", "")
        self.app_name = app_name or os.getenv("OPENROUTER_APP_NAME", "")
        if not self.api_key:
            raise ValueError("Missing API key. Set OPENROUTER_API_KEY/OPENAI_API_KEY or pass --api_key.")

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if "openrouter.ai" in self.base_url:
            if self.site_url:
                headers["HTTP-Referer"] = self.site_url
            if self.app_name:
                headers["X-Title"] = self.app_name
        return headers

    def _backoff_seconds(self, attempt: int) -> float:
        base = min(60.0, 2.0 ** attempt)
        return base + random.uniform(0.0, 0.5)

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        last_err: Exception | None = None

        for attempt in range(self.max_retries):
            session = requests.Session()
            try:
                resp = session.post(url, headers=self._headers(), json=body, timeout=self.timeout)
                if resp.status_code >= 400:
                    err_text = resp.text.strip()
                    hint = ""
                    if resp.status_code == 404:
                        hint = " Hint: for OpenRouter, this is often caused by an unavailable/invalid model id."
                    http_err = requests.HTTPError(
                        f"{resp.status_code} error for {url}. Response: {err_text[:500]}{hint}",
                        response=resp,
                    )
                    raise http_err
                text = resp.text.strip()
                if not text:
                    raise ValueError("Empty response body")
                try:
                    return json.loads(text)
                except json.JSONDecodeError as err:
                    raise ValueError(
                        f"Non-JSON response (status={resp.status_code}): {text[:200]!r}"
                    ) from err
            except Exception as err:
                last_err = err
                if not _is_retryable_error(err):
                    break
                if attempt + 1 < self.max_retries:
                    wait_s = self._backoff_seconds(attempt)
                    time.sleep(wait_s)
            finally:
                session.close()

        assert last_err is not None
        raise last_err

    def _extract_content(self, data: dict[str, Any]) -> str:
        msg = data.get("choices", [{}])[0].get("message", {})
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts: list[str] = []
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    text_parts.append(part["text"])
            if text_parts:
                return "\n".join(text_parts)
        return '{"anomaly_ranges_local":[],"anomaly_score":0.0,"rationale":"empty model content"}'

    def chat(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            body["max_tokens"] = int(max_tokens)
        data = self._post(body)
        return self._extract_content(data)

    def list_models(self) -> list[str]:
        url = f"{self.base_url}/models"
        session = requests.Session()
        try:
            resp = session.get(url, headers=self._headers(), timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
        finally:
            session.close()
        items = data.get("data", [])
        models: list[str] = []
        for it in items:
            model_id = it.get("id")
            if isinstance(model_id, str) and model_id.strip():
                models.append(model_id.strip())
        return models
