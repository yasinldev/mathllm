from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import sympy as sp

from .planner import Plan, PlanStep
from .verify import _import_mathcore, numeric_probe, symbolic_equal

LOGGER = logging.getLogger(__name__)


@dataclass
class StepResult:
    index: int
    type: str
    status: str
    duration_ms: float
    output: Optional[Any] = None
    verify_flag: Optional[bool] = None
    error: Optional[str] = None

    def to_json(self) -> Dict[str, Any]:
        payload = {
            "index": self.index,
            "type": self.type,
            "status": self.status,
            "duration_ms": round(self.duration_ms, 3),
        }
        if self.output is not None:
            payload["output"] = self.output
        if self.verify_flag is not None:
            payload["verify_flag"] = self.verify_flag
        if self.error:
            payload["error"] = self.error
        return payload


@dataclass
class ExecutionMetrics:
    total_time_ms: float
    tool_use_rate: float
    verify_success_rate: float
    step_count: int
    tool_count: int
    verify_count: int

    def to_json(self) -> Dict[str, Any]:
        return {
            "total_time_ms": round(self.total_time_ms, 3),
            "tool_use_rate": self.tool_use_rate,
            "verify_success_rate": self.verify_success_rate,
            "step_count": self.step_count,
            "tool_count": self.tool_count,
            "verify_count": self.verify_count,
        }


@dataclass
class ExecutionResult:
    ok: bool
    steps: List[StepResult]
    metrics: ExecutionMetrics
    context: Dict[str, str]
    sympy_context: Dict[str, sp.Expr] = field(default_factory=dict)
    error: Optional[str] = None

    def to_json(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "error": self.error,
            "metrics": self.metrics.to_json(),
            "steps": [step.to_json() for step in self.steps],
            "context": self.context,
        }


class ToolRuntime:
    def __init__(self, *, tool_whitelist: Optional[Dict[str, Any]] = None) -> None:
        self.mathcore = _import_mathcore()
        self.tool_whitelist = set(tool_whitelist or {"integrate", "diff", "solve_equation", "verify_equal", "simplify", "ode_solve_stub"})

    def execute_plan(self, plan: Plan) -> ExecutionResult:
        context: Dict[str, sp.Expr] = {}
        recorded_steps: List[StepResult] = []
        tool_success = 0
        verify_success = 0
        tool_total = 0
        verify_total = 0
        total_start = time.perf_counter()
        for index, step in enumerate(plan.steps):
            start = time.perf_counter()
            try:
                if step.type == "tool_call":
                    tool_total += 1
                    output = self._execute_tool(step, context)
                    tool_success += 1
                    result = StepResult(index=index, type=step.type, status="ok",
                                         duration_ms=(time.perf_counter() - start) * 1000,
                                         output=output)
                elif step.type == "verify":
                    verify_total += 1
                    verify_flag = self._execute_verify(step, context)
                    if verify_flag:
                        verify_success += 1
                    status = "ok" if verify_flag else "fail"
                    result = StepResult(index=index, type=step.type, status=status,
                                         duration_ms=(time.perf_counter() - start) * 1000,
                                         verify_flag=verify_flag)
                    if not verify_flag:
                        recorded_steps.append(result)
                        metrics = self._build_metrics(total_start, tool_total, tool_success, verify_total, verify_success, len(plan.steps))
                        return ExecutionResult(ok=False, steps=recorded_steps, metrics=metrics,
                                               context=self._stringify_context(context), sympy_context=dict(context),
                                               error=f"Verify failed at step {index}")
                elif step.type == "derive":
                    output = self._execute_derive(step, context)
                    result = StepResult(index=index, type=step.type, status="ok",
                                         duration_ms=(time.perf_counter() - start) * 1000,
                                         output=output)
                elif step.type == "final":
                    self._execute_final(step, context)
                    result = StepResult(index=index, type=step.type, status="ok",
                                         duration_ms=(time.perf_counter() - start) * 1000,
                                         output=step.payload.get("result"))
                else:
                    result = StepResult(index=index, type=step.type, status="skipped",
                                         duration_ms=(time.perf_counter() - start) * 1000,
                                         error="Unknown step type")
            except Exception as exc:  # pragma: no cover - runtime errors captured in higher level tests
                duration = (time.perf_counter() - start) * 1000
                result = StepResult(index=index, type=step.type, status="error", duration_ms=duration,
                                     error=str(exc))
                recorded_steps.append(result)
                metrics = self._build_metrics(total_start, tool_total, tool_success, verify_total, verify_success, len(plan.steps))
                return ExecutionResult(ok=False, steps=recorded_steps, metrics=metrics,
                                       context=self._stringify_context(context), sympy_context=dict(context),
                                       error=f"Step {index} failed: {exc}")
            recorded_steps.append(result)
        metrics = self._build_metrics(total_start, tool_total, tool_success, verify_total, verify_success, len(plan.steps))
        return ExecutionResult(ok=True, steps=recorded_steps, metrics=metrics,
                               context=self._stringify_context(context), sympy_context=dict(context))

    def _execute_tool(self, step: PlanStep, context: Dict[str, sp.Expr]) -> Any:
        payload = step.payload
        tool = payload.get("tool")
        if tool not in self.tool_whitelist:
            raise RuntimeError(f"Tool {tool} not allowed")
        args = payload.get("args", {})
        bind = payload.get("bind")
        if not isinstance(bind, str) or not bind:
            raise RuntimeError("Tool call missing bind")
        handler_name = f"_tool_{tool}"
        handler = getattr(self, handler_name, None)
        if handler is None:
            raise RuntimeError(f"Tool handler not implemented: {tool}")
        result_expr = handler(args, context)
        if isinstance(result_expr, bool):
            context[bind] = sp.sympify(result_expr)
            return {"bind": bind, "value": bool(result_expr)}
        if not isinstance(result_expr, sp.Expr):
            result_expr = sp.sympify(result_expr)
        context[bind] = result_expr
        return {"bind": bind, "expr": sp.sstr(result_expr)}

    def _execute_verify(self, step: PlanStep, context: Dict[str, sp.Expr]) -> bool:
        payload = step.payload
        lhs_expr = self._evaluate_expression(payload.get("lhs"), context)
        rhs_expr = self._evaluate_expression(payload.get("rhs"), context)
        symbolic_ok = symbolic_equal(lhs_expr, rhs_expr)
        if symbolic_ok:
            return True
        symbols = sorted(lhs_expr.free_symbols.union(rhs_expr.free_symbols), key=lambda s: s.name)
        return numeric_probe(lhs_expr, rhs_expr, symbols)

    def _execute_derive(self, step: PlanStep, context: Dict[str, sp.Expr]) -> Dict[str, Any]:
        payload = step.payload
        expr_text = payload.get("expr")
        bind = payload.get("bind")
        if not expr_text:
            return {"note": "no-op"}
        expr = self._evaluate_expression(expr_text, context)
        if isinstance(bind, str) and bind:
            context[bind] = expr
            return {"bind": bind, "expr": sp.sstr(expr)}
        return {"expr": sp.sstr(expr)}

    def _execute_final(self, step: PlanStep, context: Dict[str, sp.Expr]) -> None:
        result = step.payload.get("result")
        if not isinstance(result, str):
            raise RuntimeError("Final step missing result binding")
        if result not in context:
            raise RuntimeError(f"Final binding {result} not found in context")

    def _evaluate_expression(self, expression: Optional[str], context: Dict[str, sp.Expr]) -> sp.Expr:
        if not isinstance(expression, str):
            raise RuntimeError("Expression missing or not string")
        locals_map = {name: value for name, value in context.items()}
        try:
            return sp.sympify(expression, locals=locals_map)
        except Exception as exc:
            raise RuntimeError(f"Failed to parse expression '{expression}': {exc}") from exc

    def _tool_integrate(self, args: Dict[str, Any], _: Dict[str, sp.Expr]) -> sp.Expr:
        expr = args.get("expr")
        var = args.get("var")
        if not isinstance(expr, str) or not isinstance(var, str):
            raise RuntimeError("integrate requires expr and var strings")
        result = self.mathcore.integrate(expr, var)
        return sp.sympify(result)

    def _tool_diff(self, args: Dict[str, Any], _: Dict[str, sp.Expr]) -> sp.Expr:
        expr = args.get("expr")
        var = args.get("var")
        if not isinstance(expr, str) or not isinstance(var, str):
            raise RuntimeError("diff requires expr and var strings")
        result = self.mathcore.diff(expr, var)
        return sp.sympify(result)

    def _tool_solve_equation(self, args: Dict[str, Any], _: Dict[str, sp.Expr]) -> sp.Expr:
        lhs = args.get("lhs")
        rhs = args.get("rhs")
        var = args.get("var")
        if not all(isinstance(item, str) for item in (lhs, rhs, var)):
            raise RuntimeError("solve_equation requires lhs, rhs, var strings")
        raw = self.mathcore.solve_equation(lhs, rhs, var)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = [raw]
        solutions = [sp.sympify(sol) for sol in parsed]
        return sp.Matrix(solutions) if len(solutions) > 1 else solutions[0]

    def _tool_verify_equal(self, args: Dict[str, Any], _: Dict[str, sp.Expr]) -> bool:
        lhs = args.get("lhs")
        rhs = args.get("rhs")
        if not isinstance(lhs, str) or not isinstance(rhs, str):
            raise RuntimeError("verify_equal requires lhs and rhs strings")
        return bool(self.mathcore.verify_equal(lhs, rhs))

    def _tool_simplify(self, args: Dict[str, Any], _: Dict[str, sp.Expr]) -> sp.Expr:
        expr = args.get("expr")
        if not isinstance(expr, str):
            raise RuntimeError("simplify requires expr string")
        return sp.simplify(expr)

    def _tool_ode_solve_stub(self, args: Dict[str, Any], _: Dict[str, sp.Expr]) -> sp.Expr:
        expr = args.get("expr", "0")
        return sp.sympify(expr)

    @staticmethod
    def _stringify_context(context: Dict[str, sp.Expr]) -> Dict[str, str]:
        return {name: sp.sstr(value) for name, value in context.items()}

    @staticmethod
    def _build_metrics(total_start: float, tool_total: int, tool_success: int,
                       verify_total: int, verify_success: int, total_steps: int) -> ExecutionMetrics:
        elapsed_ms = (time.perf_counter() - total_start) * 1000
        tool_rate = (tool_success / tool_total) if tool_total else 0.0
        verify_rate = (verify_success / verify_total) if verify_total else 0.0
        return ExecutionMetrics(
            total_time_ms=elapsed_ms,
            tool_use_rate=tool_rate,
            verify_success_rate=verify_rate,
            step_count=total_steps,
            tool_count=tool_total,
            verify_count=verify_total,
        )


def execute_plan(plan: Plan) -> ExecutionResult:
    runtime = ToolRuntime()
    return runtime.execute_plan(plan)
