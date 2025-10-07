from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

import httpx


class ExplanationStyle(str, Enum):
    ACADEMIC = "academic"
    FRIENDLY = "friendly"
    ORAL_EXAM = "oral_exam"
    ONE_MINUTE = "one_minute"


@dataclass
class TalkerConfig:
    mode: str = "api"
    api_base: str = "http://localhost:8010/v1"
    api_key: Optional[str] = None
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    temperature: float = 0.4
    top_p: float = 0.9
    max_tokens: int = 256
    timeout_seconds: int = 30

    @classmethod
    def from_env(cls) -> TalkerConfig:
        return cls(
            mode=os.getenv("TALKER_MODE", "api"),
            api_base=os.getenv("TALKER_API_BASE", "http://localhost:8010/v1"),
            api_key=os.getenv("TALKER_API_KEY"),
            model_name=os.getenv("TALKER_MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
            temperature=float(os.getenv("TALKER_TEMPERATURE", "0.4")),
            top_p=float(os.getenv("TALKER_TOP_P", "0.9")),
            max_tokens=int(os.getenv("TALKER_MAX_TOKENS", "256")),
            timeout_seconds=int(os.getenv("TALKER_TIMEOUT_SECONDS", "30")),
        )


@dataclass
class ExplanationResult:
    text: str
    style: ExplanationStyle
    guard_passed: bool
    redrafts: int
    cached: bool = False
    latency_ms: float = 0.0


class TalkerClient:
    def __init__(self, config: Optional[TalkerConfig] = None):
        self.config = config or TalkerConfig.from_env()
        self._client: Optional[httpx.Client] = None
        self._cache: Dict[str, str] = {}
        self._cache_path = Path("runs/talker_cache.json")
        self._load_cache()

    def _load_cache(self) -> None:
        if self._cache_path.exists():
            try:
                with self._cache_path.open("r", encoding="utf-8") as f:
                    self._cache = json.load(f)
            except Exception:
                self._cache = {}

    def _save_cache(self) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self._cache_path.open("w", encoding="utf-8") as f:
                json.dump(self._cache, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            headers = {}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            self._client = httpx.Client(
                base_url=self.config.api_base,
                headers=headers,
                timeout=self.config.timeout_seconds,
            )
        return self._client

    def shutdown(self) -> None:
        if self._client:
            self._client.close()
            self._client = None
        self._save_cache()

    def _make_cache_key(
        self,
        problem_latex: str,
        result_latex: str,
        style: ExplanationStyle,
    ) -> str:
        payload = f"{problem_latex}|{result_latex}|{style.value}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def generate_explanation(
        self,
        problem_latex: str,
        result_latex: str,
        style: ExplanationStyle = ExplanationStyle.ACADEMIC,
        code_preview: Optional[str] = None,
        objective: Optional[str] = None,
    ) -> str:
        cache_key = self._make_cache_key(problem_latex, result_latex, style)
        if cache_key in self._cache:
            return self._cache[cache_key]

        prompt = self._build_prompt(
            problem_latex=problem_latex,
            result_latex=result_latex,
            style=style,
            code_preview=code_preview,
            objective=objective,
        )

        client = self._get_client()
        response = client.post(
            "/chat/completions",
            json={
                "model": self.config.model_name,
                "messages": [
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt},
                ],
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "max_tokens": self.config.max_tokens,
            },
        )
        response.raise_for_status()
        data = response.json()
        text = data["choices"][0]["message"]["content"].strip()

        self._cache[cache_key] = text
        return text

    def redraft_explanation(
        self,
        result_latex: str,
        previous_text: str,
        style: ExplanationStyle,
    ) -> str:
        prompt = self._build_redraft_prompt(
            result_latex=result_latex,
            previous_text=previous_text,
            style=style,
        )

        client = self._get_client()
        response = client.post(
            "/chat/completions",
            json={
                "model": self.config.model_name,
                "messages": [
                    {"role": "system", "content": self._get_redraft_system_prompt()},
                    {"role": "user", "content": prompt},
                ],
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "max_tokens": self.config.max_tokens,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    def _get_system_prompt(self) -> str:
        return (
            "You are an experienced educator. Explain the mathematical result below. "
            "DO NOT CHANGE the result or numbers. Keep mathematical expressions in LaTeX format."
        )

    def _get_redraft_system_prompt(self) -> str:
        return (
            "The previous explanation altered the result. "
            "Keep the LaTeX result exactly as given below."
        )

    def _build_prompt(
        self,
        problem_latex: str,
        result_latex: str,
        style: ExplanationStyle,
        code_preview: Optional[str],
        objective: Optional[str],
    ) -> str:
        style_instructions = self._get_style_instructions(style)
        
        parts = [
            f"Problem (LaTeX):\n{problem_latex}",
            f"\nVerified result (LaTeX):\n{result_latex}",
        ]
        
        if code_preview:
            parts.append(f"\nEngineering code (summary):\n{code_preview}")
        
        if objective:
            parts.append(f"\nObjective: {objective}")
        
        parts.append(f"\nStyle: {style.value}")
        parts.append(f"\n{style_instructions}")
        
        return "\n".join(parts)

    def _build_redraft_prompt(
        self,
        result_latex: str,
        previous_text: str,
        style: ExplanationStyle,
    ) -> str:
        style_instructions = self._get_style_instructions(style)
        return (
            f"Original result (LaTeX): {result_latex}\n"
            f"Broken text: {previous_text}\n\n"
            f"REQUEST: Rewrite in the same style but without changing the result. "
            f"Maximum 5 sentences.\n{style_instructions}"
        )

    def _get_style_instructions(self, style: ExplanationStyle) -> str:
        if style == ExplanationStyle.ACADEMIC:
            return (
                "REQUEST:\n"
                "- 4-7 clear sentences (academic tone)\n"
                "- Precise terminology, one brief example if needed\n"
                "- NEVER change the result or numerical values"
            )
        elif style == ExplanationStyle.FRIENDLY:
            return (
                "REQUEST:\n"
                "- 4-7 clear sentences (conversational tone)\n"
                "- Minimal jargon, one practical example if helpful\n"
                "- NEVER change the result or numerical values"
            )
        elif style == ExplanationStyle.ORAL_EXAM:
            return (
                "REQUEST:\n"
                "- 5-7 sentences (explain like answering an oral exam)\n"
                "- Show reasoning steps, mention key concepts\n"
                "- NEVER change the result or numerical values"
            )
        elif style == ExplanationStyle.ONE_MINUTE:
            return (
                "REQUEST:\n"
                "- 2-3 short sentences (one-minute summary)\n"
                "- Essential information only, maximum 260 characters\n"
                "- NEVER change the result or numerical values"
            )
        return ""
