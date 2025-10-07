from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import sympy as sp

from .llm_student import StudentLLM
from .llm_teacher import TeacherLLM
from .mir import MIRProblem, Objective
from .planner import Plan, Planner, PlanError
from .tool_runtime import ExecutionResult, ToolRuntime
from .verify import VerificationResult, verify_all

LOGGER = logging.getLogger(__name__)


@dataclass
class PolicyConfig:
    max_repair_attempts: int = 2
    self_consistency: int = 3
    retry_on_execution_failure: bool = True
    retry_on_verify_failure: bool = True
    retry_on_final_failure: bool = True
    log_dir: str = "runs"
    enable_logging: bool = True
    teacher_enabled: bool = True
    teacher_rate_limit: float = 0.1
    teacher_warmup_runs: int = 5
    teacher_latency_budget_ms: float = 2500.0


@dataclass
class VerificationOutcome:
    ok: bool
    symbolic: bool
    numeric: bool
    units: Optional[str]
    details: Dict[str, Any]

    def to_json(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "symbolic": self.symbolic,
            "numeric": self.numeric,
            "units": self.units,
            "details": self.details,
        }


@dataclass
class AttemptLog:
    attempt_index: int
    repair_round: int
    plan: Plan
    execution: ExecutionResult
    verification: Optional[VerificationOutcome]
    success: bool

    def to_json(self) -> Dict[str, Any]:
        return {
            "attempt": self.attempt_index,
            "repair_round": self.repair_round,
            "success": self.success,
            "plan": {
                "text": self.plan.raw_text,
                "generation": self.plan.generation.raw if self.plan.generation else None,
                "metadata": self.plan.metadata,
            },
            "execution": self.execution.to_json(),
            "verification": self.verification.to_json() if self.verification else None,
        }


@dataclass
class PolicyOutcome:
    ok: bool
    best_attempt: Optional[AttemptLog]
    attempts: List[AttemptLog]
    logs_path: Optional[str]
    teacher_used: bool = False
    teacher_latency_ms: Optional[float] = None
    teacher_attempted: bool = False
    teacher_error: Optional[str] = None

    def metrics(self) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {
            "teacher_used": self.teacher_used,
            "teacher_attempted": self.teacher_attempted,
        }
        if self.teacher_latency_ms is not None:
            metrics["teacher_latency_ms"] = self.teacher_latency_ms
        if self.teacher_error:
            metrics["teacher_error"] = self.teacher_error
        if self.best_attempt is None:
            return metrics
        execution_metrics = self.best_attempt.execution.metrics.to_json()
        verification_metrics = self.best_attempt.verification.to_json() if self.best_attempt.verification else {}
        metrics["execution"] = execution_metrics
        metrics["verification"] = verification_metrics
        return metrics


@dataclass
class TeacherFallbackResult:
    plan: Optional[Plan]
    latency_ms: Optional[float]
    error: Optional[str] = None


class VerifierFirstPolicy:
    def __init__(self, student: StudentLLM, *, config: Optional[PolicyConfig] = None,
                 planner: Optional[Planner] = None, runtime: Optional[ToolRuntime] = None,
                 teacher: Optional[TeacherLLM] = None) -> None:
        self.config = config or PolicyConfig()
        self.planner = planner or Planner(student)
        self.runtime = runtime or ToolRuntime()
        self.teacher = teacher
        self._total_runs = 0
        self._teacher_requests = 0
        self._teacher_successes = 0
        self._teacher_latency_total_ms = 0.0
        self._teacher_latency_samples = 0

    def run(self, problem: MIRProblem, *, latex: str, objective: Optional[Objective] = None,
            extra_context: Optional[str] = None) -> PolicyOutcome:
        attempts: List[AttemptLog] = []
        best_attempt: Optional[AttemptLog] = None
        teacher_used = False
        teacher_attempted = False
        teacher_latency_ms: Optional[float] = None
        teacher_error: Optional[str] = None
        for attempt_idx in range(self.config.self_consistency):
            try:
                plan = self.planner.propose(problem, latex=latex, objective=objective, extra_context=extra_context)
            except PlanError as exc:
                LOGGER.error("Planner failed on attempt %s: %s", attempt_idx, exc)
                continue
            LOGGER.info("Planner attempt %s produced %s steps", attempt_idx, len(plan.steps))
            attempt_logs = self._execute_with_repairs(problem, plan, attempt_idx)
            attempts.extend(attempt_logs)
            winner = self._select_successful_attempt(attempt_logs)
            if winner and (best_attempt is None or not best_attempt.success):
                best_attempt = winner
            if best_attempt and best_attempt.success:
                break
        if (best_attempt is None or not best_attempt.success) and self._should_use_teacher():
            teacher_result = self._request_teacher_plan(problem, latex=latex, objective=objective, extra_context=extra_context)
            if teacher_result is not None:
                teacher_attempted = True
                teacher_latency_ms = teacher_result.latency_ms
                teacher_error = teacher_result.error
                if teacher_result.plan is not None:
                    teacher_used = True
                    teacher_attempt_idx = self.config.self_consistency
                    teacher_logs = self._execute_with_repairs(problem, teacher_result.plan, teacher_attempt_idx)
                    attempts.extend(teacher_logs)
                    winner = self._select_successful_attempt(teacher_logs)
                    if winner and (best_attempt is None or not best_attempt.success):
                        best_attempt = winner

        logs_path = self._write_logs(attempts)
        ok = bool(best_attempt and best_attempt.success)
        self._record_teacher_usage(
            attempted=teacher_attempted,
            used=teacher_used,
            latency_ms=teacher_latency_ms,
        )
        return PolicyOutcome(
            ok=ok,
            best_attempt=best_attempt,
            attempts=attempts,
            logs_path=logs_path,
            teacher_used=teacher_used,
            teacher_latency_ms=teacher_latency_ms,
            teacher_attempted=teacher_attempted,
            teacher_error=teacher_error,
        )

    def _execute_with_repairs(self, problem: MIRProblem, plan: Plan, attempt_idx: int) -> List[AttemptLog]:
        current_plan = plan
        attempt_logs: List[AttemptLog] = []
        for repair_round in range(self.config.max_repair_attempts + 1):
            execution = self.runtime.execute_plan(current_plan)
            verification = None
            success = False
            if execution.ok:
                verification = self._final_verification(problem, current_plan, execution)
                success = verification.ok
            attempt_log = AttemptLog(
                attempt_index=attempt_idx,
                repair_round=repair_round,
                plan=current_plan,
                execution=execution,
                verification=verification,
                success=success,
            )
            attempt_logs.append(attempt_log)
            if success:
                break
            if not execution.ok:
                if not self.config.retry_on_execution_failure or repair_round == self.config.max_repair_attempts:
                    break
                failure_step = self._infer_failure_step(execution)
                try:
                    repaired_plan = self.planner.repair(current_plan, failure_step, execution.error or "execution failure",
                                                        context=json.dumps(execution.to_json()))
                    current_plan = self._merge_plans(current_plan, repaired_plan, failure_step)
                    continue
                except PlanError as exc:
                    LOGGER.error("Plan repair failed: %s", exc)
                    break
            elif verification and not verification.ok:
                if not self.config.retry_on_final_failure or repair_round == self.config.max_repair_attempts:
                    break
                failure_step = self._infer_failure_step(execution)
                try:
                    repaired_plan = self.planner.repair(current_plan, failure_step, "final verification failed",
                                                        context=json.dumps(verification.to_json()))
                    current_plan = self._merge_plans(current_plan, repaired_plan, failure_step)
                    continue
                except PlanError as exc:
                    LOGGER.error("Plan repair after final failure failed: %s", exc)
                    break
        return attempt_logs

    @staticmethod
    def _infer_failure_step(execution: ExecutionResult) -> int:
        for step in execution.steps:
            if step.status in {"fail", "error"}:
                return step.index
        return execution.steps[-1].index if execution.steps else 0

    @staticmethod
    def _merge_plans(original: Plan, repaired: Plan, start_index: int) -> Plan:
        merged_steps = list(original.steps[:start_index]) + list(repaired.steps)
        metadata = dict(original.metadata)
        metadata.setdefault("origin", original.metadata.get("origin", "student"))
        metadata["mode"] = "repair"
        if repaired.metadata:
            metadata.update({k: v for k, v in repaired.metadata.items() if k not in {"origin"}})
            metadata["repaired_by"] = repaired.metadata.get("origin", metadata.get("origin"))
        return Plan(steps=merged_steps, raw_text=repaired.raw_text, generation=repaired.generation, metadata=metadata)

    def _select_successful_attempt(self, logs: List[AttemptLog]) -> Optional[AttemptLog]:
        successful = [log for log in logs if log.success]
        if not successful:
            return None
        successful.sort(key=lambda log: (log.execution.metrics.total_time_ms, log.repair_round))
        return successful[0]

    def _should_use_teacher(self) -> bool:
        if not self.teacher or not self.config.teacher_enabled:
            return False
        if self._total_runs < self.config.teacher_warmup_runs:
            return True
        max_allowed = max(1, int(self.config.teacher_rate_limit * (self._total_runs + 1)))
        if self._teacher_requests + 1 > max_allowed:
            LOGGER.info(
                "Skipping teacher fallback to respect rate limit: %s/%s > %.3f",
                self._teacher_requests + 1,
                self._total_runs + 1,
                self.config.teacher_rate_limit,
            )
            return False
        return True

    def _record_teacher_usage(self, *, attempted: bool, used: bool, latency_ms: Optional[float]) -> None:
        self._total_runs += 1
        if attempted:
            self._teacher_requests += 1
            if latency_ms is not None:
                self._teacher_latency_total_ms += latency_ms
                self._teacher_latency_samples += 1
        if used:
            self._teacher_successes += 1

    def teacher_stats(self) -> Dict[str, Any]:
        total_runs = self._total_runs
        requests = self._teacher_requests
        successes = self._teacher_successes
        avg_latency = (
            self._teacher_latency_total_ms / self._teacher_latency_samples
            if self._teacher_latency_samples
            else None
        )
        request_rate = (requests / total_runs) if total_runs else 0.0
        success_rate = (successes / requests) if requests else 0.0
        return {
            "total_runs": total_runs,
            "teacher_requests": requests,
            "teacher_successes": successes,
            "teacher_request_rate": request_rate,
            "teacher_success_rate": success_rate,
            "avg_teacher_latency_ms": avg_latency,
        }

    def _request_teacher_plan(self, problem: MIRProblem, *, latex: str, objective: Optional[Objective],
                              extra_context: Optional[str]) -> Optional[TeacherFallbackResult]:
        if not self.teacher:
            return None
        objective_value = objective or problem.objective
        prompt = self.planner._build_plan_prompt(problem, latex=latex, objective=objective_value, extra_context=extra_context)
        teacher_metadata = {
            "objective": objective_value.value,
            "problem": problem.to_dict(),
        }
        start = time.perf_counter()
        try:
            parsed, result = self.teacher.generate_plan_json(prompt, metadata=teacher_metadata)
        except Exception as exc:  # pragma: no cover - network/teacher issues
            latency_ms = (time.perf_counter() - start) * 1000.0
            LOGGER.error("Teacher fallback failed: %s", exc)
            return TeacherFallbackResult(plan=None, latency_ms=latency_ms, error=str(exc))
        latency_ms = (time.perf_counter() - start) * 1000.0
        if latency_ms > self.config.teacher_latency_budget_ms:
            LOGGER.warning(
                "Teacher latency %.1fms exceeded budget %.1fms", latency_ms, self.config.teacher_latency_budget_ms
            )
        steps_payload = parsed.get("steps")
        if not isinstance(steps_payload, list):
            LOGGER.error("Teacher response missing steps")
            return TeacherFallbackResult(plan=None, latency_ms=latency_ms, error="missing steps")
        try:
            steps = self.planner._validate_steps(steps_payload)
        except PlanError as exc:
            LOGGER.error("Teacher plan validation failed: %s", exc)
            return TeacherFallbackResult(plan=None, latency_ms=latency_ms, error=str(exc))
        raw_text = json.dumps(parsed, ensure_ascii=False)
        teacher_metadata_out = {
            "origin": "teacher",
            "mode": "propose",
            "teacher_latency_ms": latency_ms,
        }
        teacher_metadata_out["teacher_prompt_metadata"] = teacher_metadata
        if isinstance(result.raw, dict):
            teacher_metadata_out["teacher_raw"] = {
                k: result.raw.get(k)
                for k in ("id", "model", "cached", "usage")
                if k in result.raw
            }
        plan = Plan(steps=steps, raw_text=raw_text, metadata=teacher_metadata_out)
        return TeacherFallbackResult(plan=plan, latency_ms=latency_ms)

    def _final_verification(self, problem: MIRProblem, plan: Plan, execution: ExecutionResult) -> VerificationOutcome:
        final_binding = plan.steps[-1].payload.get("result")
        if not isinstance(final_binding, str):
            raise RuntimeError("Final step missing result binding")
        if final_binding not in execution.sympy_context:
            raise RuntimeError(f"Final binding {final_binding} missing in execution context")
        candidate_expr = execution.sympy_context[final_binding]
        objective = problem.objective
        if objective == Objective.INTEGRATE:
            return self._verify_integral(problem, candidate_expr)
        if objective == Objective.DIFFERENTIATE:
            return self._verify_derivative(problem, candidate_expr)
        if objective == Objective.SOLVE:
            return self._verify_solutions(problem, candidate_expr)
        raise RuntimeError(f"Objective {objective} not supported in policy")

    def _verify_integral(self, problem: MIRProblem, candidate_expr: sp.Expr) -> VerificationOutcome:
        var = problem.variables[0]
        integrand = problem.expr.sympy_expr
        derivative = sp.diff(candidate_expr, var)
        verification = verify_all(problem.expr, derivative, reference_expr=integrand, unit_subject=candidate_expr)
        return self._wrap_verification(verification)

    def _verify_derivative(self, problem: MIRProblem, candidate_expr: sp.Expr) -> VerificationOutcome:
        var = problem.variables[0]
        expected = sp.diff(problem.expr.sympy_expr, var)
        verification = verify_all(problem.expr, candidate_expr, reference_expr=expected)
        return self._wrap_verification(verification)

    def _verify_solutions(self, problem: MIRProblem, candidate_expr: sp.Expr) -> VerificationOutcome:
        var = problem.variables[0]
        eq_expr = problem.expr.sympy_expr
        solutions = self._extract_solutions(candidate_expr)
        symbolic_checks: List[bool] = []
        numeric_checks: List[bool] = []
        for solution in solutions:
            substituted = eq_expr.subs(var, solution)
            symbolic_checks.append(sp.simplify(substituted) == 0)
            numeric_checks.append(sp.N(substituted).equals(0))
        ok = all(symbolic_checks) and all(numeric_checks)
        details = {
            "solutions": [sp.sstr(sol) for sol in solutions],
            "symbolic_checks": symbolic_checks,
            "numeric_checks": numeric_checks,
        }
        return VerificationOutcome(ok=ok, symbolic=all(symbolic_checks), numeric=all(numeric_checks), units=None, details=details)

    @staticmethod
    def _extract_solutions(candidate_expr: sp.Expr) -> List[sp.Expr]:
        if isinstance(candidate_expr, sp.MatrixBase):
            return [candidate_expr[i, 0] for i in range(candidate_expr.rows)]
        is_iterable = getattr(candidate_expr, "is_Iterable", False)
        if is_iterable:
            try:
                return list(candidate_expr)
            except TypeError:
                pass
        return [candidate_expr]

    @staticmethod
    def _wrap_verification(verification: VerificationResult) -> VerificationOutcome:
        return VerificationOutcome(
            ok=verification.ok,
            symbolic=verification.symbolic,
            numeric=verification.numeric,
            units=verification.units,
            details=verification.details,
        )

    def _write_logs(self, attempts: List[AttemptLog]) -> Optional[str]:
        if not attempts or not self.config.enable_logging:
            return None
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log_dir = Path(self.config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / f"plan_{timestamp}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for entry in attempts:
                handle.write(json.dumps(entry.to_json(), ensure_ascii=False) + "\n")
        return str(path)