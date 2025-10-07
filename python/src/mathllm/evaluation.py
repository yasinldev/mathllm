from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import sympy as sp

from .latex import LatexParseResult, parse_expression_from_input
from .mir import MIRProblem, Objective, from_sympy
from .policy import PolicyOutcome, VerifierFirstPolicy


@dataclass(frozen=True)
class BenchEntry:
    """Single evaluation example loaded from a bench JSONL file."""

    bench_index: int
    latex: str
    objective: Objective
    name: Optional[str] = None
    assumptions: Optional[Dict[str, Any]] = None
    extra_context: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def display_name(self) -> str:
        if self.name:
            return self.name
        return f"item_{self.bench_index:03d}"


@dataclass
class EvaluationRecord:
    """Result of running the planner policy against a single bench entry."""

    entry: BenchEntry
    ok: bool
    attempts: int
    runtime_ms: float
    total_time_ms: Optional[float]
    logs_path: Optional[str]
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    teacher_used: bool = False
    teacher_attempted: bool = False
    teacher_latency_ms: Optional[float] = None
    attempt_successes: List[bool] = field(default_factory=list)
    verification_attempts: int = 0
    verification_successes: int = 0

    def to_json(self) -> Dict[str, Any]:
        payload = {
            "name": self.entry.display_name,
            "latex": self.entry.latex,
            "objective": self.entry.objective.value,
            "assumptions": self.entry.assumptions,
            "extra_context": self.entry.extra_context,
            "metadata": self.entry.metadata or None,
            "ok": self.ok,
            "attempts": self.attempts,
            "runtime_ms": round(self.runtime_ms, 3),
            "total_time_ms": round(self.total_time_ms, 3) if self.total_time_ms is not None else None,
            "logs_path": self.logs_path,
            "metrics": self.metrics or None,
            "error": self.error,
            "teacher_used": self.teacher_used,
            "teacher_attempted": self.teacher_attempted,
            "teacher_latency_ms": round(self.teacher_latency_ms, 3) if self.teacher_latency_ms is not None else None,
            "attempt_successes": self.attempt_successes or None,
            "verification_attempts": self.verification_attempts or None,
            "verification_successes": self.verification_successes or None,
        }
        return {key: value for key, value in payload.items() if value is not None}


@dataclass
class EvaluationSummary:
    bench_name: str
    records: List[EvaluationRecord]

    def successes(self) -> int:
        return sum(1 for record in self.records if record.ok)

    def total(self) -> int:
        return len(self.records)

    def success_rate(self) -> float:
        return self.successes() / self.total() if self.records else 0.0

    def average_runtime(self) -> float:
        if not self.records:
            return 0.0
        return sum(record.runtime_ms for record in self.records) / len(self.records)

    def average_total_time(self) -> float:
        totals = [record.total_time_ms for record in self.records if record.total_time_ms is not None]
        if not totals:
            return 0.0
        return sum(totals) / len(totals)

    def max_attempts(self) -> int:
        lengths = [len(record.attempt_successes) for record in self.records]
        return max(lengths) if lengths else 0

    def pass_at_k(self, k: int) -> float:
        if not self.records or k <= 0:
            return 0.0
        successes = 0
        for record in self.records:
            attempts = record.attempt_successes
            if not attempts:
                continue
            limit = min(k, len(attempts))
            if any(attempts[idx] for idx in range(limit)):
                successes += 1
        return successes / len(self.records)

    def verify_success_rate(self) -> float:
        total = sum(record.verification_attempts for record in self.records)
        if total == 0:
            return 0.0
        successes = sum(record.verification_successes for record in self.records)
        return successes / total

    def teacher_attempt_rate(self) -> float:
        if not self.records:
            return 0.0
        attempted = sum(1 for record in self.records if record.teacher_attempted)
        return attempted / len(self.records)

    def teacher_use_rate(self) -> float:
        if not self.records:
            return 0.0
        used = sum(1 for record in self.records if record.teacher_used)
        return used / len(self.records)

    def teacher_success_rate(self) -> float:
        attempts = sum(1 for record in self.records if record.teacher_attempted)
        if attempts == 0:
            return 0.0
        successes = sum(1 for record in self.records if record.teacher_used)
        return successes / attempts

    def average_teacher_latency(self) -> float:
        latencies = [record.teacher_latency_ms for record in self.records if record.teacher_latency_ms is not None]
        if not latencies:
            return 0.0
        return sum(latencies) / len(latencies)

    def _subset_summary(self, predicate) -> Dict[str, Any]:
        subset = [record for record in self.records if predicate(record)]
        if not subset:
            return {"count": 0, "success_rate": 0.0, "average_runtime_ms": 0.0}
        success_rate = sum(1 for record in subset if record.ok) / len(subset)
        avg_runtime = sum(record.runtime_ms for record in subset) / len(subset)
        return {
            "count": len(subset),
            "success_rate": round(success_rate, 4),
            "average_runtime_ms": round(avg_runtime, 3),
        }

    def to_json(self) -> Dict[str, Any]:
        max_attempts = self.max_attempts()
        pass_at = {str(k): round(self.pass_at_k(k), 4) for k in range(1, max_attempts + 1)} if max_attempts else {}
        return {
            "bench": self.bench_name,
            "total": self.total(),
            "successes": self.successes(),
            "success_rate": round(self.success_rate(), 4),
            "average_runtime_ms": round(self.average_runtime(), 3),
            "average_total_time_ms": round(self.average_total_time(), 3),
            "verify_success_rate": round(self.verify_success_rate(), 4),
            "teacher_attempt_rate": round(self.teacher_attempt_rate(), 4),
            "teacher_use_rate": round(self.teacher_use_rate(), 4),
            "teacher_success_rate": round(self.teacher_success_rate(), 4),
            "average_teacher_latency_ms": round(self.average_teacher_latency(), 3),
            "pass_at_k": pass_at or None,
            "ablations": {
                "teacher_used": self._subset_summary(lambda record: record.teacher_used),
                "teacher_not_used": self._subset_summary(lambda record: not record.teacher_used),
                "teacher_attempted": self._subset_summary(lambda record: record.teacher_attempted),
            },
            "records": [record.to_json() for record in self.records],
        }

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_bench(path: Path) -> List[BenchEntry]:
    entries: List[BenchEntry] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            stripped = line.strip()
            if not stripped:
                continue
            raw = json.loads(stripped)
            objective_value = raw.get("objective")
            if not objective_value:
                raise ValueError(f"Bench entry {index} missing 'objective'")
            try:
                objective = Objective(objective_value)
            except ValueError as exc:
                raise ValueError(f"Bench entry {index} has unsupported objective: {objective_value}") from exc
            entry = BenchEntry(
                bench_index=index,
                latex=raw["latex"],
                objective=objective,
                name=raw.get("name") or raw.get("id"),
                assumptions=raw.get("assumptions"),
                extra_context=raw.get("context"),
                metadata={k: v for k, v in raw.items() if k not in {"latex", "objective", "name", "id", "assumptions", "context"}},
            )
            entries.append(entry)
    return entries


def _prepare_problem(entry: BenchEntry) -> Tuple[MIRProblem, LatexParseResult]:
    parse_result = parse_expression_from_input(entry.latex)
    expr = parse_result.expression
    objective = entry.objective
    assumptions = entry.assumptions
    if objective == Objective.INTEGRATE:
        integrand: sp.Expr
        variables: Sequence[sp.Symbol]
        if isinstance(expr, sp.Integral):
            integrand = expr.function
            variables = expr.variables
        else:
            integrand = sp.sympify(expr)
            variables = sorted(integrand.free_symbols, key=lambda symbol: symbol.name)
        if not variables:
            raise ValueError(f"Integral bench '{entry.display_name}' requires at least one variable")
        var = variables[0]
        problem = from_sympy(integrand, objective=Objective.INTEGRATE, variables=[var], assumptions=assumptions)
    elif objective == Objective.DIFFERENTIATE:
        base_expr: sp.Expr
        variables: Sequence[sp.Symbol]
        if isinstance(expr, sp.Derivative):
            base_expr = expr.expr
            variables = expr.variables
        else:
            base_expr = sp.sympify(expr)
            variables = sorted(base_expr.free_symbols, key=lambda symbol: symbol.name)
        if not variables:
            raise ValueError(f"Differentiate bench '{entry.display_name}' requires at least one variable")
        var = variables[0]
        problem = from_sympy(base_expr, objective=Objective.DIFFERENTIATE, variables=[var], assumptions=assumptions)
    elif objective == Objective.SOLVE:
        if isinstance(expr, sp.Equality):
            eq_expr = expr.lhs - expr.rhs
            variables = sorted(expr.free_symbols, key=lambda symbol: symbol.name)
        else:
            raise ValueError(f"Solve bench '{entry.display_name}' must be an equation with '=' sign")
        if len(variables) != 1:
            raise ValueError(f"Solve bench '{entry.display_name}' must reference exactly one variable")
        problem = from_sympy(eq_expr, objective=Objective.SOLVE, variables=[variables[0]], assumptions=assumptions)
    else:
        raise ValueError(f"Objective {objective} is not supported in evaluation benches")
    return problem, parse_result


def _summarize_attempts(outcome: PolicyOutcome) -> Tuple[List[bool], int, int]:
    attempt_groups: Dict[int, List[Any]] = {}
    for log in outcome.attempts:
        attempt_groups.setdefault(log.attempt_index, []).append(log)
    attempt_successes: List[bool] = []
    verification_attempts = 0
    verification_successes = 0
    for attempt_index in sorted(attempt_groups.keys()):
        logs = attempt_groups[attempt_index]
        attempt_successes.append(any(log.success for log in logs))
        for log in logs:
            if log.verification is not None:
                verification_attempts += 1
                if log.verification.ok:
                    verification_successes += 1
    return attempt_successes, verification_attempts, verification_successes


def _extract_metrics(outcome: PolicyOutcome) -> Tuple[Dict[str, Any], Optional[float], List[bool], int, int]:
    metrics = outcome.metrics()
    attempt_successes, verification_attempts, verification_successes = _summarize_attempts(outcome)
    total_time: Optional[float] = None
    if outcome.best_attempt is not None:
        total_time = outcome.best_attempt.execution.metrics.total_time_ms
    elif outcome.attempts:
        total_time = outcome.attempts[-1].execution.metrics.total_time_ms
    metrics.setdefault("policy", {})
    metrics["policy"].update(
        {
            "attempts": len(attempt_successes),
            "pass_trace": attempt_successes,
            "total_logs": len(outcome.attempts),
        }
    )
    return metrics, total_time, attempt_successes, verification_attempts, verification_successes


def run_bench(policy: VerifierFirstPolicy, entries: Iterable[BenchEntry], *, bench_name: str) -> EvaluationSummary:
    records: List[EvaluationRecord] = []
    for entry in entries:
        start = time.perf_counter()
        try:
            problem, parse_result = _prepare_problem(entry)
            extra_context = entry.extra_context or parse_result.raw_input
            outcome = policy.run(problem, latex=entry.latex, objective=entry.objective, extra_context=extra_context)
            metrics, total_time, attempt_successes, verification_attempts, verification_successes = _extract_metrics(outcome)
            ok = outcome.ok
            logs_path = outcome.logs_path
            attempts = len(outcome.attempts)
            error = None
        except Exception as exc:
            ok = False
            metrics = {}
            total_time = None
            logs_path = None
            attempts = 0
            error = str(exc)
            attempt_successes = []
            verification_attempts = 0
            verification_successes = 0
            teacher_used = False
            teacher_attempted = False
            teacher_latency_ms = None
        else:
            teacher_used = outcome.teacher_used
            teacher_attempted = outcome.teacher_attempted
            teacher_latency_ms = outcome.teacher_latency_ms
        runtime_ms = (time.perf_counter() - start) * 1000
        records.append(
            EvaluationRecord(
                entry=entry,
                ok=ok,
                attempts=attempts,
                runtime_ms=runtime_ms,
                total_time_ms=total_time,
                logs_path=logs_path,
                metrics=metrics,
                error=error,
                teacher_used=teacher_used,
                teacher_attempted=teacher_attempted,
                teacher_latency_ms=teacher_latency_ms,
                attempt_successes=attempt_successes,
                verification_attempts=verification_attempts,
                verification_successes=verification_successes,
            )
        )
    return EvaluationSummary(bench_name=bench_name, records=records)
