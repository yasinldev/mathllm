import json

import sympy as sp

from mathllm.llm_student import StudentConfig, StudentLLM
from mathllm.llm_teacher import TeacherResult
from mathllm.mir import Objective, from_sympy
from mathllm.planner import Planner, PlanError
from mathllm.policy import PolicyConfig, VerifierFirstPolicy
from mathllm.tool_runtime import ExecutionMetrics, ExecutionResult, StepResult


class FailingPlanner(Planner):
    def propose(self, *args, **kwargs):
        raise PlanError("student planner failed")

    def repair(self, *args, **kwargs):
        raise PlanError("student repair failed")


class StubRuntime:
    def __init__(self, result: ExecutionResult):
        self.result = result

    def execute_plan(self, plan):
        return self.result


class FakeTeacher:
    def __init__(self, plan_json):
        self.plan_json = plan_json
        self.calls = 0

    def generate_plan_json(self, prompt, metadata=None):
        self.calls += 1
        return self.plan_json, TeacherResult(text=json.dumps(self.plan_json), raw={"cached": False})


def _make_success_execution(expr_text: str = "x**2/2") -> ExecutionResult:
    sym_expr = sp.sympify(expr_text)
    metrics = ExecutionMetrics(
        total_time_ms=42.0,
        tool_use_rate=1.0,
        verify_success_rate=1.0,
        step_count=3,
        tool_count=1,
        verify_count=1,
    )
    steps = [
        StepResult(index=0, type="tool_call", status="ok", duration_ms=1.0, output={"bind": "R1", "expr": expr_text}),
        StepResult(index=1, type="verify", status="ok", duration_ms=1.0, verify_flag=True),
        StepResult(index=2, type="final", status="ok", duration_ms=1.0, output="R1"),
    ]
    return ExecutionResult(
        ok=True,
        steps=steps,
        metrics=metrics,
        context={"R1": expr_text},
        sympy_context={"R1": sym_expr},
    )


def _make_problem() -> tuple:
    x = sp.Symbol("x")
    problem = from_sympy(x, objective=Objective.INTEGRATE, variables=[x])
    latex = "\\int x\\,dx"
    return problem, latex


def test_policy_teacher_fallback_success():
    student = StudentLLM(StudentConfig(model_name_or_path="stub", use_stub=True))
    planner = FailingPlanner(student)
    execution = _make_success_execution()
    runtime = StubRuntime(execution)
    plan_json = {
        "steps": [
            {"type": "tool_call", "tool": "simplify", "args": {"expr": "x"}, "bind": "R1"},
            {"type": "verify", "lhs": "R1", "rhs": "R1"},
            {"type": "final", "result": "R1"},
        ]
    }
    teacher = FakeTeacher(plan_json)
    config = PolicyConfig(self_consistency=1, teacher_rate_limit=1.0, teacher_warmup_runs=0)
    policy = VerifierFirstPolicy(student, config=config, planner=planner, runtime=runtime, teacher=teacher)

    problem, latex = _make_problem()
    outcome = policy.run(problem, latex=latex, objective=Objective.INTEGRATE)

    assert outcome.ok
    assert outcome.teacher_used
    assert outcome.teacher_attempted
    assert outcome.teacher_error is None
    assert teacher.calls == 1
    assert outcome.best_attempt is not None
    assert outcome.best_attempt.plan.metadata.get("origin") == "teacher"
    stats = policy.teacher_stats()
    assert stats["teacher_requests"] == 1
    assert stats["teacher_successes"] == 1


def test_policy_respects_teacher_rate_limit():
    student = StudentLLM(StudentConfig(model_name_or_path="stub", use_stub=True))
    planner = FailingPlanner(student)
    execution = _make_success_execution()
    runtime = StubRuntime(execution)
    plan_json = {
        "steps": [
            {"type": "tool_call", "tool": "simplify", "args": {"expr": "x"}, "bind": "R1"},
            {"type": "verify", "lhs": "R1", "rhs": "R1"},
            {"type": "final", "result": "R1"},
        ]
    }
    teacher = FakeTeacher(plan_json)
    config = PolicyConfig(self_consistency=1, teacher_rate_limit=0.1, teacher_warmup_runs=0)
    policy = VerifierFirstPolicy(student, config=config, planner=planner, runtime=runtime, teacher=teacher)

    problem, latex = _make_problem()

    first = policy.run(problem, latex=latex, objective=Objective.INTEGRATE)
    assert first.teacher_used
    assert first.teacher_attempted

    second = policy.run(problem, latex=latex, objective=Objective.INTEGRATE)
    assert not second.teacher_used
    assert not second.teacher_attempted
    assert teacher.calls == 1
    assert not second.ok
    stats = policy.teacher_stats()
    assert stats["teacher_requests"] == 1
    assert stats["teacher_successes"] == 1
    assert stats["total_runs"] == 2