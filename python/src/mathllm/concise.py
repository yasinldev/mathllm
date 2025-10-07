from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import Any, Dict, Optional

import sympy as sp

from .guard import GuardConfig, GuardResult, preserve_result
from .mir import MIRProblem, Objective
from .policy import PolicyOutcome


@dataclass
class ConciseConfig:
    max_characters: int = 400
    numeric_guard_samples: int = 4
    guard_threshold: float = 1e-6
    include_teacher_metadata: bool = False


class ConciseError(RuntimeError):
    pass

def render_concise(
    problem: MIRProblem,
    outcome: PolicyOutcome,
    *,
    objective: Objective,
    candidate_expr: sp.Expr,
    verification: Optional[Any],
    eng_payload: Optional[Dict[str, Any]] = None,
    config: Optional[ConciseConfig] = None,
) -> Dict[str, Any]:
    if outcome.best_attempt is None or not outcome.attempts:
        raise ConciseError("no_successful_attempt")
    config = config or ConciseConfig()
    verification_payload = verification.to_json() if verification else None
    checks = {
        "symbolic": getattr(verification, "symbolic", False) if verification else False,
        "numeric": getattr(verification, "numeric", False) if verification else False,
        "units": getattr(verification, "units", "n/a") if verification else "n/a",
    }
    result_latex = sp.latex(candidate_expr)
    verification_ok = bool(getattr(verification, "ok", False) if verification else False)
    explanation = _short_explanation(problem, objective, result_latex, verification_ok)
    if len(result_latex) + len(explanation) > config.max_characters:
        explanation = _trim_text(explanation, config.max_characters - len(result_latex))
    symbols = problem.variables
    guard_config = GuardConfig(
        numeric_samples=config.numeric_guard_samples,
        numeric_threshold=config.guard_threshold,
    )
    guard_result: GuardResult = preserve_result(candidate_expr, result_latex, symbols, config=guard_config)
    if not guard_result.ok:
        raise ConciseError(f"guard_failed: {guard_result.reason}")
    execution_metrics = outcome.best_attempt.execution.metrics.to_json()
    timings = {
        "planner_total": outcome.best_attempt.execution.metrics.total_time_ms,
        "policy_attempts": len(outcome.attempts),
    }
    payload: Dict[str, Any] = {
        "ok": outcome.ok,
        "objective": objective.value,
        "result_latex": result_latex,
        "code_preview": _code_snippet(eng_payload, objective),
        "explanation": explanation,
        "verified": verification_ok,
        "checks": checks,
        "timings_ms": timings,
        "execution_metrics": execution_metrics,
    }
    if config.include_teacher_metadata:
        payload["teacher"] = {
            "used": outcome.teacher_used,
            "attempted": outcome.teacher_attempted,
            "latency_ms": outcome.teacher_latency_ms,
        }
    if eng_payload:
        payload["engineering"] = {
            "unit_status": eng_payload.get("unit_status"),
            "symbols": eng_payload.get("symbols"),
        }
    return payload


def _short_explanation(
    problem: MIRProblem,
    objective: Objective,
    result_latex: str,
    verified: bool,
) -> str:
    base = "Result" if not verified else "Verified result"
    if objective == Objective.INTEGRATE:
        return f"{base}: antiderivative simplifies to {result_latex}."
    if objective == Objective.DIFFERENTIATE:
        return f"{base}: derivative equals {result_latex}."
    if objective == Objective.SOLVE:
        return f"{base}: solution {result_latex}."
    return f"{base}: {result_latex}."


def _trim_text(text: str, max_length: int) -> str:
    if max_length <= 0:
        return ""
    text = text.strip()
    if len(text) <= max_length:
        return text
    ellipsis = "…"
    trimmed = textwrap.shorten(text, width=max_length, placeholder=ellipsis)
    if len(trimmed) > max_length:
        trimmed = trimmed[: max_length - len(ellipsis)] + ellipsis
    return trimmed


def _code_snippet(eng_payload: Optional[Dict[str, Any]], objective: Objective) -> Optional[str]:
    if not eng_payload:
        return None
    if objective != Objective.INTEGRATE and objective != Objective.DIFFERENTIATE:
        return None
    snippet = eng_payload.get("numpy_fn_preview")
    if snippet:
        return snippet.strip()
    return None
