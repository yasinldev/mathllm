from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import httpx


@dataclass
class TeacherConfig:
    api_base: str
    model: str
    api_key: Optional[str] = None
    temperature: float = 0.2
    top_p: float = 0.9
    max_new_tokens: int = 800
    timeout: float = 30.0
    system_prompt: Optional[str] = None
    extra_headers: Dict[str, str] = field(default_factory=dict)
    cache_enabled: bool = True

    @classmethod
    def from_env(cls) -> TeacherConfig:
        api_base = os.environ.get("TEACHER_API_BASE", "https://teacher.example.com/v1")
        model = os.environ.get("TEACHER_MODEL", "openmath-14b-tir")
        api_key = os.environ.get("TEACHER_API_KEY")
        temperature = float(os.environ.get("TEACHER_TEMPERATURE", "0.2"))
        top_p = float(os.environ.get("TEACHER_TOP_P", "0.9"))
        max_new_tokens = int(os.environ.get("TEACHER_MAX_NEW_TOKENS", "800"))
        timeout = float(os.environ.get("TEACHER_TIMEOUT", "30"))
        system_prompt = os.environ.get("TEACHER_SYSTEM_PROMPT")
        cache_enabled = os.environ.get("TEACHER_CACHE", "1") == "1"
        headers: Dict[str, str] = {}
        user_agent = os.environ.get("TEACHER_USER_AGENT")
        if user_agent:
            headers["User-Agent"] = user_agent
        return cls(
            api_base=api_base,
            model=model,
            api_key=api_key,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            timeout=timeout,
            system_prompt=system_prompt,
            extra_headers=headers,
            cache_enabled=cache_enabled,
        )


@dataclass
class TeacherResult:
    text: str
    raw: Dict[str, Any]


class TeacherCache:
    def __init__(self) -> None:
        self._data: Dict[str, TeacherResult] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[TeacherResult]:
        with self._lock:
            return self._data.get(key)

    def put(self, key: str, value: TeacherResult) -> None:
        with self._lock:
            self._data[key] = value


class TeacherLLM:
    def __init__(self, config: TeacherConfig, *, cache: Optional[TeacherCache] = None, client: Optional[httpx.Client] = None) -> None:
        self.config = config
        self.cache = cache if cache is not None else (TeacherCache() if config.cache_enabled else None)
        self._client = client

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            headers = {"Content-Type": "application/json"}
            headers.update(self.config.extra_headers)
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            self._client = httpx.Client(base_url=self.config.api_base, headers=headers, timeout=self.config.timeout)
        return self._client

    def shutdown(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def _cache_key(self, prompt: str, metadata: Optional[Dict[str, Any]]) -> str:
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "metadata": metadata,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_new_tokens": self.config.max_new_tokens,
            "system_prompt": self.config.system_prompt,
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        return digest

    def generate_plan(self, prompt: str, *, metadata: Optional[Dict[str, Any]] = None) -> TeacherResult:
        cache_entry = None
        cache_key = None
        if self.cache is not None:
            cache_key = self._cache_key(prompt, metadata)
            cache_entry = self.cache.get(cache_key)
            if cache_entry is not None:
                return cache_entry
        messages = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        messages.append({"role": "user", "content": prompt})
        body = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_new_tokens,
            "messages": messages,
            "response_format": {"type": "json_object"},
        }
        if metadata:
            body["metadata"] = metadata
        response = self.client.post("/chat/completions", json=body)
        response.raise_for_status()
        payload = response.json()
        choices = payload.get("choices")
        if not choices:
            raise RuntimeError("Teacher response missing choices")
        message = choices[0].get("message")
        if not message:
            raise RuntimeError("Teacher response missing message")
        content = message.get("content")
        if content is None:
            raise RuntimeError("Teacher response missing content")
        result = TeacherResult(text=content, raw=payload)
        if self.cache is not None and cache_key is not None:
            self.cache.put(cache_key, result)
        return result

    def generate_plan_json(self, prompt: str, *, metadata: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], TeacherResult]:
        result = self.generate_plan(prompt, metadata=metadata)
        try:
            parsed = json.loads(result.text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Teacher returned invalid JSON: {exc}") from exc
        steps = parsed.get("steps")
        if not isinstance(steps, list):
            raise RuntimeError("Teacher JSON missing steps array")
        for step in steps:
            if not isinstance(step, dict):
                raise RuntimeError("Teacher JSON has malformed step entry")
            tool = step.get("tool")
            if step.get("type") == "tool_call" and tool not in {"integrate", "diff", "solve_equation", "simplify", "verify_equal", "ode_solve_stub"}:
                raise RuntimeError(f"Teacher used unsupported tool: {tool}")
        return parsed, result


def load_default_teacher() -> TeacherLLM:
    return TeacherLLM(TeacherConfig.from_env())
