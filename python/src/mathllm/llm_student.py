from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
import sympy as sp
try:
    import torch
except ImportError:
    torch = None

try:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
except ImportError:
    AutoConfig = AutoModelForCausalLM = AutoTokenizer = None

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None

LOGGER = logging.getLogger(__name__)


@dataclass
class StudentConfig:
    model_name_or_path: str
    mode: str = "local"
    adapter_path: Optional[str] = None
    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.05
    device_map: str = "auto"
    torch_dtype: Optional[str] = "bfloat16"
    enable_adapter: bool = True
    use_cache: bool = True
    eos_token: Optional[str] = None
    pad_token: Optional[str] = None
    system_prompt: Optional[str] = None
    generation_kwargs: Dict[str, Any] = field(default_factory=dict)
    use_stub: bool = False
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    api_timeout: float = 30.0
    api_headers: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "StudentConfig":
        model = os.environ.get("STUDENT_MODEL", "meta-llama/Llama-2-7b-hf")
        mode = os.environ.get("STUDENT_MODE", "local").strip().lower()
        adapter = os.environ.get("STUDENT_ADAPTER")
        dtype = os.environ.get("STUDENT_DTYPE", "bfloat16")
        temperature = float(os.environ.get("STUDENT_TEMPERATURE", "0.2"))
        top_p = float(os.environ.get("STUDENT_TOP_P", "0.9"))
        top_k = int(os.environ.get("STUDENT_TOP_K", "50"))
        max_new_tokens = int(os.environ.get("STUDENT_MAX_NEW_TOKENS", "512"))
        repetition_penalty = float(os.environ.get("STUDENT_REP_PENALTY", "1.05"))
        device_map = os.environ.get("STUDENT_DEVICE_MAP", "auto")
        use_stub = model.lower() == "stub" or os.environ.get("STUDENT_STUB", "0") == "1"
        if mode == "api":
            use_stub = False
        system_prompt = os.environ.get("STUDENT_SYSTEM_PROMPT")
        timeout = float(os.environ.get("STUDENT_TIMEOUT", os.environ.get("STUDENT_API_TIMEOUT", "30")))
        api_base = os.environ.get("STUDENT_API_BASE")
        if mode == "api" and not api_base:
            api_base = "http://localhost:8009/v1"
        api_key = os.environ.get("STUDENT_API_KEY")
        headers: Dict[str, str] = {}
        user_agent = os.environ.get("STUDENT_USER_AGENT")
        if user_agent:
            headers["User-Agent"] = user_agent
        return cls(
            model_name_or_path=model,
            mode=mode,
            adapter_path=adapter,
            torch_dtype=dtype,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            device_map=device_map,
            enable_adapter=not use_stub and mode != "api",
            use_stub=use_stub,
            system_prompt=system_prompt,
            api_base=api_base,
            api_key=api_key,
            api_timeout=timeout,
            api_headers=headers,
        )


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    raw: Dict[str, Any]


class StudentLLM:
    def __init__(self, config: StudentConfig) -> None:
        self.config = config
        self._tokenizer = None
        self._model = None
        self._resolved_eos_id: Optional[int] = None
        self._resolved_pad_id: Optional[int] = None
        self._client: Optional[httpx.Client] = None

    @property
    def tokenizer(self):
        if self.config.use_stub:
            raise RuntimeError("Stub mode does not provide a tokenizer")
        if self.config.mode == "api":
            raise RuntimeError("API mode does not expose a tokenizer")
        if AutoTokenizer is None:
            raise RuntimeError("transformers must be installed to load the tokenizer")
        if self._tokenizer is None:
            LOGGER.info("Loading student tokenizer from %s", self.config.model_name_or_path)
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name_or_path, use_fast=True)
            if self.config.pad_token:
                self._tokenizer.pad_token = self.config.pad_token
            elif self._tokenizer.pad_token is None and self._tokenizer.eos_token is not None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            if self.config.eos_token:
                self._tokenizer.eos_token = self.config.eos_token
        return self._tokenizer

    @property
    def model(self):
        if self.config.use_stub:
            raise RuntimeError("Stub mode does not load a transformer model")
        if self.config.mode == "api":
            raise RuntimeError("API mode does not load a local transformer model")
        if torch is None:
            raise RuntimeError("torch is required to load the student model")
        if AutoConfig is None or AutoModelForCausalLM is None:
            raise RuntimeError("transformers must be installed to load the student model")
        if self._model is None:
            LOGGER.info("Loading student model from %s", self.config.model_name_or_path)
            auto_config = AutoConfig.from_pretrained(self.config.model_name_or_path)
            dtype = self._resolve_dtype(self.config.torch_dtype)
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name_or_path,
                config=auto_config,
                device_map=self.config.device_map,
                torch_dtype=dtype,
                use_cache=self.config.use_cache,
            )
            if self.config.enable_adapter and self.config.adapter_path:
                if PeftModel is None:
                    raise RuntimeError("peft is not installed but adapter_path is set")
                adapter_path = Path(self.config.adapter_path)
                if not adapter_path.exists():
                    raise FileNotFoundError(f"Adapter path not found: {adapter_path}")
                LOGGER.info("Loading adapter from %s", adapter_path)
                model = PeftModel.from_pretrained(model, adapter_path)
            self._model = model.eval()
            if self.tokenizer.pad_token_id is not None:
                self._resolved_pad_id = self.tokenizer.pad_token_id
            if self.tokenizer.eos_token_id is not None:
                self._resolved_eos_id = self.tokenizer.eos_token_id
        return self._model

    def generate(self, prompt: str, *, max_new_tokens: Optional[int] = None, temperature: Optional[float] = None,
                 top_p: Optional[float] = None, top_k: Optional[int] = None) -> GenerationResult:
        if self.config.use_stub:
            raise RuntimeError("Stub mode does not support raw generation")
        if self.config.mode == "api":
            raise RuntimeError("Direct text generation is not available in API mode")
        if torch is None:
            raise RuntimeError("torch must be installed to perform generation")
        tokenizer = self.tokenizer
        model = self.model
        device = next(model.parameters()).device
        tokens = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_length = tokens.input_ids.shape[-1]
        generation_kwargs = {
            "max_new_tokens": max_new_tokens or self.config.max_new_tokens,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "top_p": top_p if top_p is not None else self.config.top_p,
            "top_k": top_k if top_k is not None else self.config.top_k,
            "repetition_penalty": self.config.repetition_penalty,
            "do_sample": True,
            "pad_token_id": self._resolved_pad_id,
            "eos_token_id": self._resolved_eos_id,
        }
        generation_kwargs.update(self.config.generation_kwargs)
        with torch.no_grad():
            outputs = model.generate(**tokens, **generation_kwargs)
        completion = outputs[0, prompt_length:]
        full = outputs[0]
        text = tokenizer.decode(full, skip_special_tokens=True)
        completion_text = tokenizer.decode(completion, skip_special_tokens=True)
        prompt_tokens = prompt_length
        completion_tokens = completion.shape[-1]
        total_tokens = full.shape[-1]
        raw = {
            "prompt": prompt,
            "completion": completion_text,
            "full_text": text,
            "generation_kwargs": generation_kwargs,
        }
        return GenerationResult(text=text, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                                total_tokens=total_tokens, raw=raw)

    def generate_plan(self, prompt: str) -> GenerationResult:
        prompt_with_system = self._build_prompt(prompt, mode="plan")
        if self.config.use_stub:
            return self._stub_generate(prompt_with_system)
        if self.config.mode == "api":
            system_prompt = self._system_prompt()
            user_prompt = self._build_prompt(prompt, mode="plan", include_header=False)
            return self._api_generate(system_prompt, user_prompt)
        return self.generate(prompt_with_system)

    def repair_plan(self, prompt: str) -> GenerationResult:
        prompt_with_system = self._build_prompt(prompt, mode="repair")
        if self.config.use_stub:
            return self._stub_generate(prompt_with_system)
        if self.config.mode == "api":
            system_prompt = self._system_prompt()
            user_prompt = self._build_prompt(prompt, mode="repair", include_header=False)
            return self._api_generate(system_prompt, user_prompt, temperature=0.1)
        return self.generate(prompt_with_system, temperature=0.1)

    def _build_prompt(self, user_prompt: str, mode: str, *, include_header: bool = True) -> str:
        header = self._system_prompt() if include_header else ""
        meta = json.dumps({"mode": mode})
        parts = []
        if header:
            parts.append(header)
        parts.append(f"[meta]{meta}")
        parts.append(user_prompt)
        return "\n".join(parts).strip()

    def _system_prompt(self) -> str:
        return self.config.system_prompt or "You are MathLLM's student planner."

    def _get_client(self) -> httpx.Client:
        if not self.config.api_base:
            raise RuntimeError("Student API base URL is not configured")
        if self._client is None:
            headers = {"Content-Type": "application/json"}
            headers.update(self.config.api_headers)
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            self._client = httpx.Client(base_url=self.config.api_base, timeout=self.config.api_timeout, headers=headers)
        return self._client

    def shutdown(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def _api_generate(self, system_prompt: str, user_prompt: str, *,
                      temperature: Optional[float] = None,
                      top_p: Optional[float] = None,
                      max_new_tokens: Optional[int] = None) -> GenerationResult:
        client = self._get_client()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        body = {
            "model": self.config.model_name_or_path,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "top_p": top_p if top_p is not None else self.config.top_p,
            "max_tokens": max_new_tokens or self.config.max_new_tokens,
        }
        response = client.post("/chat/completions", json=body)
        response.raise_for_status()
        payload = response.json()
        choices = payload.get("choices") or []
        if not choices:
            raise RuntimeError("Student API response missing choices")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if content is None:
            raise RuntimeError("Student API response missing message content")
        usage = payload.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
        raw = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "request": body,
            "payload": payload,
        }
        return GenerationResult(
            text=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            raw=raw,
        )

    @staticmethod
    def _resolve_dtype(dtype: Optional[str]) -> Optional[Any]:
        if dtype is None:
            return None
        if torch is None:
            raise RuntimeError("torch must be installed to resolve dtype")
        norm = dtype.lower()
        mapping = {
            "float16": torch.float16,
            "half": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if norm not in mapping:
            raise ValueError(f"Unsupported torch dtype: {dtype}")
        return mapping[norm]
    
    # stub helpers 
    def _stub_generate(self, prompt: str) -> GenerationResult:
        context = self._stub_extract_context(prompt)
        plan = self._stub_build_plan(context)
        raw = {
            "prompt": prompt,
            "completion": plan,
            "full_text": plan,
            "generation_kwargs": {"stub": True},
        }
        return GenerationResult(text=plan, prompt_tokens=0, completion_tokens=len(plan), total_tokens=len(plan), raw=raw)

    def _stub_extract_context(self, prompt: str) -> Dict[str, Any]:
        meta_start = prompt.find("[meta]")
        if meta_start == -1:
            raise ValueError("Stub planner expected metadata block")
        newline = prompt.find("\n", meta_start)
        meta_payload = prompt[meta_start + len("[meta]"): newline if newline != -1 else None].strip()
        meta = json.loads(meta_payload) if meta_payload else {}
        body = prompt[newline + 1:] if newline != -1 else ""
        latex = None
        summary_raw = None
        objective = None
        for line in body.splitlines():
            stripped = line.strip()
            if stripped.startswith("Problem (LaTeX):"):
                latex = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("MIR summary:"):
                summary_raw = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("Objective:"):
                objective = stripped.split(":", 1)[1].strip()
        if summary_raw is None:
            raise ValueError("Stub planner missing MIR summary")
        summary = json.loads(summary_raw)
        if objective is None:
            objective = summary.get("objective")
        return {"meta": meta, "latex": latex, "summary": summary, "objective": objective}

    def _stub_build_plan(self, context: Dict[str, Any]) -> str:
        summary = context["summary"]
        objective = (context.get("objective") or summary.get("objective") or "integrate").lower()
        expr = summary.get("expr", "0")
        variables = summary.get("variables") or []
        var = variables[0] if variables else self._infer_primary_symbol(expr)
        if objective == "integrate":
            plan = self._stub_integrate_plan(expr, var)
        elif objective in {"differentiate", "diff"}:
            plan = self._stub_diff_plan(expr, var)
        elif objective == "solve":
            plan = self._stub_solve_plan(expr, var)
        else:
            plan = self._stub_fallback_plan(expr)
        return json.dumps({"steps": plan}, ensure_ascii=False)

    @staticmethod
    def _infer_primary_symbol(expr: str) -> str:
        try:
            symbols = sorted(sp.sympify(expr).free_symbols, key=lambda s: s.name)
            if symbols:
                return symbols[0].name
        except Exception:  # pragma: no cover, heuristics
            pass
        return "x"

    @staticmethod
    def _stub_integrate_plan(expr: str, var: str) -> list[Dict[str, Any]]:
        bind = "I1"
        return [
            {"type": "tool_call", "tool": "integrate", "args": {"expr": expr, "var": var}, "bind": bind},
            {"type": "verify", "lhs": f"diff({bind}, {var})", "rhs": expr},
            {"type": "final", "result": bind},
        ]

    @staticmethod
    def _stub_diff_plan(expr: str, var: str) -> list[Dict[str, Any]]:
        bind = "D1"
        return [
            {"type": "tool_call", "tool": "diff", "args": {"expr": expr, "var": var}, "bind": bind},
            {"type": "verify", "lhs": bind, "rhs": f"diff({expr}, {var})"},
            {"type": "final", "result": bind},
        ]

    @staticmethod
    def _stub_solve_plan(expr: str, var: str) -> list[Dict[str, Any]]:
        bind = "S1"
        return [
            {"type": "tool_call", "tool": "solve_equation", "args": {"lhs": expr, "rhs": "0", "var": var}, "bind": bind},
            {"type": "verify", "lhs": f"simplify(({expr}).subs({var}, {bind}))", "rhs": "0"},
            {"type": "final", "result": bind},
        ]

    @staticmethod
    def _stub_fallback_plan(expr: str) -> list[Dict[str, Any]]:
        bind = "R1"
        return [
            {"type": "tool_call", "tool": "simplify", "args": {"expr": expr}, "bind": bind},
            {"type": "verify", "lhs": bind, "rhs": f"simplify({expr})"},
            {"type": "final", "result": bind},
        ]


def load_default_student() -> StudentLLM:
    config = StudentConfig.from_env()
    return StudentLLM(config)
