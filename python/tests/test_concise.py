from __future__ import annotations

import pytest
import sympy as sp

import mathllm.concise as concise_module
from mathllm.concise import ConciseConfig, ConciseError, render_concise
from mathllm.guard import GuardConfig, GuardResult, preserve_result
from mathllm.mir import Objective, from_sympy
from mathllm.policy import AttemptLog, PolicyOutcome, VerificationOutcome
from mathllm.planner import Plan, PlanStep
from mathllm.tool_runtime import ExecutionMetrics, ExecutionResult, StepResult


def _make_attempt(candidate: sp.Expr) -> AttemptLog:
    plan = Plan(
        steps=[
            PlanStep(type="final", payload={"result": "R1"}),
        ],
        raw_text="{}",
    )
    metrics = ExecutionMetrics(
        total_time_ms=12.0,
        tool_use_rate=1.0,
        verify_success_rate=1.0,
        step_count=1,
        tool_count=0,
        verify_count=0,
    )
    execution = ExecutionResult(
        ok=True,
        steps=[StepResult(index=0, type="final", status="ok", duration_ms=1.0, output="R1")],
        metrics=metrics,
        context={"R1": str(candidate)},
        sympy_context={"R1": candidate},
    )
    verification = VerificationOutcome(
        ok=True,
        symbolic=True,
        numeric=True,
        units="ok",
        details={"note": "stub"},
    )
    return AttemptLog(
        attempt_index=0,
        repair_round=0,
        plan=plan,
        execution=execution,
        verification=verification,
        success=True,
    )


def test_render_concise_produces_short_payload():
    x = sp.Symbol("x")
    candidate = x**2 / 2
    problem = from_sympy(x, objective=Objective.INTEGRATE, variables=[x])
    attempt = _make_attempt(candidate)
    outcome = PolicyOutcome(
        ok=True,
        best_attempt=attempt,
        attempts=[attempt],
        logs_path=None,
        teacher_used=False,
        teacher_latency_ms=None,
        teacher_attempted=False,
        teacher_error=None,
    )
    payload = render_concise(
        problem,
        outcome,
        objective=Objective.INTEGRATE,
        candidate_expr=candidate,
        verification=attempt.verification,
    )
    assert payload["ok"] is True
    assert payload["result_latex"] == sp.latex(candidate)
    assert len(payload["result_latex"]) + len(payload["explanation"]) <= 400
    assert payload["checks"]["numeric"] is True


def test_guard_detects_modified_result():
    x = sp.Symbol("x")
    reference = x + 1
    mutated = sp.latex(x + 2)
    config = GuardConfig(numeric_samples=2)
    result: GuardResult = preserve_result(reference, mutated, [x], config=config)
    assert not result.ok


def test_render_concise_guard_failure(monkeypatch):
    x = sp.Symbol("x")
    candidate = x + 1
    problem = from_sympy(x, objective=Objective.DIFFERENTIATE, variables=[x])
    attempt = _make_attempt(candidate)
    outcome = PolicyOutcome(
        ok=True,
        best_attempt=attempt,
        attempts=[attempt],
        logs_path=None,
        teacher_used=False,
        teacher_latency_ms=None,
        teacher_attempted=False,
        teacher_error=None,
    )
    config = ConciseConfig(max_characters=120)

    original_sp = concise_module.sp

    class _TamperedSympy:
        def __init__(self, original):
            self._original = original

        def latex(self, expr, *args, **kwargs):
            return self._original.latex(expr + 1, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._original, name)

    monkeypatch.setattr(concise_module, "sp", _TamperedSympy(original_sp))
    with pytest.raises(ConciseError):
        render_concise(
            problem,
            outcome,
            objective=Objective.DIFFERENTIATE,
            candidate_expr=candidate,
            verification=attempt.verification,
            config=config,
        )
