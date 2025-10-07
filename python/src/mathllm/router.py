from __future__ import annotations

import ast
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Any

import sympy as sp

from .concise import ConciseConfig, ConciseError, render_concise
from .explain import TalkerClient, ExplanationStyle, ExplanationResult
from .guard import preserve_explanation, GuardConfig
from .latex import LatexParseResult, LatexParseError, parse_expression_from_input
from .mir import MIRProblem, Objective, from_sympy, expr_to_mathcore_string
from .compile import to_numpy_fn, to_octave, to_matlab_stub, to_c_stub, sample_numpy_grid
from .verify import VerificationResult, verify_all, symbolic_equal, unit_check

__all__ = ["RouterError", "RouterRequest", "RouterResponse", "MathRouter"]


class RouterError(RuntimeError):
    pass


@dataclass(frozen=True)
class RouterRequest:
    latex: str
    mode: str = "academic"
    objective: Optional[str] = None
    assumptions: Optional[Dict[str, object]] = None
    emit_c_stub: bool = True
    sample_points: int = 4
    mode_params: Optional[Dict[str, Any]] = None
    concise: bool = True
    verbose: bool = False
    concise_max_chars: Optional[int] = None
    explain: bool = True
    style: str = "academic"


@dataclass(frozen=True)
class RouterResponse:
    ok: bool
    objective: Objective
    latex_out: str
    sympy_out: str
    checks: Dict[str, object]
    timings_ms: Dict[str, float]
    metadata: Dict[str, object]
    eng: Optional[Dict[str, object]] = None
    planner: Optional[Dict[str, object]] = None
    concise: Optional[Dict[str, object]] = None
    explanation: Optional[Dict[str, object]] = None


LATEX_HINTS = {
    Objective.INTEGRATE: ("\\int",),
    Objective.DIFFERENTIATE: ("\\frac{d", "d/d", "\\mathrm{d}"),
    Objective.SOLVE: ("=",),
}


class MathRouter:
    def __init__(self) -> None:
        self._mathcore = None
        self._planner_policy = None
        self._talker_client: Optional[TalkerClient] = None

    def _get_talker(self) -> TalkerClient:
        if self._talker_client is None:
            self._talker_client = TalkerClient()
        return self._talker_client

    @staticmethod
    def _is_engineering_mode(mode: Optional[str]) -> bool:
        if not mode:
            return False
        normalized = mode.lower()
        return normalized in {"eng", "engineering", "engineering-mode"}

    @staticmethod
    def _symbols_for_codegen(problem: MIRProblem, expr: sp.Expr, objective: Objective) -> List[sp.Symbol]:
        symbol_set = set(expr.free_symbols)
        symbol_set.update(problem.expr.free_symbols)
        if objective in {Objective.INTEGRATE, Objective.DIFFERENTIATE}:
            symbol_set.update(problem.variables)
        elif objective is Objective.SOLVE:
            for solved in problem.variables:
                symbol_set.discard(solved)
        return sorted(symbol_set, key=lambda s: s.name)

    def _build_engineering_payload(
        self,
        expr: sp.Expr,
        symbols: Sequence[sp.Symbol],
        unit_domains: Dict[str, Optional[Tuple[float, float]]],
        *,
        emit_c: bool,
        samples: int,
    ) -> Dict[str, object]:
        numpy_fn, numpy_source = to_numpy_fn(expr, symbols)
        octave_stub = to_octave(expr, symbols)
        matlab_stub = to_matlab_stub(expr, symbols)
        c_stub = to_c_stub(expr, symbols) if emit_c else None
        sample_eval = sample_numpy_grid(numpy_fn, symbols, unit_domains, samples=samples)
        payload: Dict[str, object] = {
            "numpy_fn_preview": numpy_source,
            "octave_stub": octave_stub,
            "matlab_stub": matlab_stub,
            "sample_eval": sample_eval,
            "symbols": [str(sym) for sym in symbols],
        }
        if c_stub is not None:
            payload["c_stub"] = c_stub
        return payload

    def _load_mathcore(self):
        if self._mathcore is None:
            from .verify import _import_mathcore

            self._mathcore = _import_mathcore()
        return self._mathcore

    @staticmethod
    def _planner_requested(request: RouterRequest) -> bool:
        return bool(request.mode_params and request.mode_params.get("planner"))

    def _get_policy(self):
        if self._planner_policy is None:
            from .llm_student import load_default_student
            from .policy import VerifierFirstPolicy, PolicyConfig

            student = load_default_student()
            config = PolicyConfig()
            teacher = None
            if os.environ.get("TEACHER_ENABLED", "0") == "1":
                from .llm_teacher import TeacherConfig, TeacherLLM

                teacher = TeacherLLM(TeacherConfig.from_env())
            self._planner_policy = VerifierFirstPolicy(student, config=config, teacher=teacher)
        return self._planner_policy

    def _run_planner_flow(self, problem: MIRProblem, request: RouterRequest, objective: Objective,
                           extra_context: Optional[str]) -> Optional[Dict[str, Any]]:
        policy = self._get_policy()
        outcome = policy.run(problem, latex=request.latex, objective=objective, extra_context=extra_context)
        planner_metadata = {
            "planner": {
                "enabled": True,
                "ok": outcome.ok,
                "attempts": len(outcome.attempts),
                "log_path": outcome.logs_path,
                "teacher_used": outcome.teacher_used,
                "teacher_latency_ms": outcome.teacher_latency_ms,
                "teacher_attempted": outcome.teacher_attempted,
                "teacher_error": outcome.teacher_error,
                "teacher_stats": policy.teacher_stats(),
                "concise_enabled": request.concise,
            }
        }
        if not outcome.best_attempt:
            return {"ok": False, "metadata": planner_metadata, "outcome": outcome}
        best_attempt = outcome.best_attempt
        planner_payload = self._build_planner_payload(best_attempt, outcome)
        final_binding = best_attempt.plan.steps[-1].payload.get("result")
        if not isinstance(final_binding, str):
            raise RouterError("Planner final step missing result binding")
        candidate_expr = best_attempt.execution.sympy_context.get(final_binding)
        if candidate_expr is None:
            raise RouterError("Planner execution missing final binding")
        verification = best_attempt.verification
        concise_payload = None
        concise_error: Optional[str] = None
        if request.concise:
            try:
                concise_config = ConciseConfig(
                    max_characters=request.concise_max_chars or 400,
                    include_teacher_metadata=request.verbose,
                )
                concise_payload = render_concise(
                    problem,
                    outcome,
                    objective=objective,
                    candidate_expr=candidate_expr,
                    verification=verification,
                    eng_payload=planner_payload,
                    config=concise_config,
                )
            except ConciseError as exc:
                concise_error = str(exc)
        if concise_error:
            planner_metadata["planner"]["concise_error"] = concise_error
        return {
            "ok": best_attempt.success,
            "candidate_expr": candidate_expr,
            "verification": verification,
            "payload": planner_payload,
            "metadata": planner_metadata,
            "concise": concise_payload,
        }

    def _build_planner_payload(self, attempt, outcome) -> Dict[str, Any]:
        plan_dict = self._parse_plan_text(attempt.plan.raw_text)
        payload = {
            "plan": plan_dict,
            "plan_text": attempt.plan.raw_text,
            "plan_metadata": attempt.plan.metadata,
            "execution": attempt.execution.to_json(),
            "steps": [step.to_json() for step in attempt.execution.steps],
            "verify_checks": attempt.verification.to_json() if attempt.verification else None,
            "metrics": {
                "execution": attempt.execution.metrics.to_json(),
                "policy": {
                    "attempts": len(outcome.attempts),
                    "log_path": outcome.logs_path,
                },
            },
            "log_path": outcome.logs_path,
        }
        return payload

    @staticmethod
    def _parse_plan_text(plan_text: str) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(plan_text)
        except json.JSONDecodeError:
            return None

    def _build_planner_response(self, objective: Objective, planner_data: Dict[str, Any],
                                metadata: Dict[str, object], timings: Dict[str, float]) -> RouterResponse:
        verification = planner_data.get("verification")
        candidate_expr = planner_data["candidate_expr"]
        latex_out = sp.latex(candidate_expr)
        sympy_out = sp.sstr(candidate_expr)
        checks = {
            "symbolic": getattr(verification, "symbolic", False),
            "numeric": getattr(verification, "numeric", False),
            "units": getattr(verification, "units", "n/a") or "n/a",
        }
        metadata.update(planner_data.get("metadata", {}))
        if verification is not None:
            metadata["details"] = verification.details
        timings.setdefault("planner_total", 0.0)
        timings["total"] = timings.get("parse", 0.0) + timings.get("planner_total", 0.0)
        return RouterResponse(
            ok=getattr(verification, "ok", False),
            objective=objective,
            latex_out=latex_out,
            sympy_out=sympy_out,
            checks=checks,
            timings_ms=timings,
            metadata=metadata,
            planner=planner_data.get("payload"),
            concise=planner_data.get("concise"),
        )

    def _generate_explanation(
        self,
        problem_latex: str,
        result_latex: str,
        style: ExplanationStyle,
        code_preview: Optional[str],
        objective: str,
        symbols: Sequence[sp.Symbol],
    ) -> ExplanationResult:
        max_redrafts = 2
        guard_config = GuardConfig()
        talker = self._get_talker()
        
        gen_start = time.perf_counter()
        try:
            text = talker.generate_explanation(
                problem_latex=problem_latex,
                result_latex=result_latex,
                style=style,
                code_preview=code_preview,
                objective=objective,
            )
            cached = False
        except Exception as exc:
            return ExplanationResult(
                text=f"Explanation unavailable: {exc}",
                style=style,
                guard_passed=False,
                redrafts=0,
                cached=False,
                latency_ms=(time.perf_counter() - gen_start) * 1000,
            )
        
        guard_result = preserve_explanation(result_latex, text, symbols, config=guard_config)
        redrafts = 0
        
        while not guard_result.ok and redrafts < max_redrafts:
            redrafts += 1
            try:
                text = talker.redraft_explanation(
                    result_latex=result_latex,
                    previous_text=text,
                    style=style,
                )
                guard_result = preserve_explanation(result_latex, text, symbols, config=guard_config)
            except Exception:
                break
        
        latency_ms = (time.perf_counter() - gen_start) * 1000
        
        return ExplanationResult(
            text=text,
            style=style,
            guard_passed=guard_result.ok,
            redrafts=redrafts,
            cached=cached,
            latency_ms=latency_ms,
        )

    def _detect_objective(self, request: RouterRequest, parse_result: LatexParseResult, expr: sp.Expr) -> Objective:
        if request.objective:
            try:
                return Objective(request.objective)
            except ValueError as exc:
                raise RouterError(f"Unsupported objective: {request.objective}") from exc
        if isinstance(expr, sp.Integral):
            return Objective.INTEGRATE
        if isinstance(expr, sp.Derivative):
            return Objective.DIFFERENTIATE
        if isinstance(expr, sp.Equality):
            return Objective.SOLVE
        for objective, hints in LATEX_HINTS.items():
            if any(hint in parse_result.raw_input for hint in hints):
                return objective
        raise RouterError("Could not infer objective. Supply the objective explicitly.")

    def _prepare_integrate(self, expr: sp.Expr, assumptions: Optional[Dict[str, object]]) -> Dict[str, object]:
        if isinstance(expr, sp.Integral):
            integrand = expr.function
            variables = list(expr.variables)
        else:
            integrand = expr
            variables = sorted(integrand.free_symbols, key=lambda s: s.name)
        if not variables:
            raise RouterError("Integral requires at least one variable")
        var = variables[0]
        problem = from_sympy(integrand, objective=Objective.INTEGRATE, variables=[var], assumptions=assumptions)
        return {"problem": problem, "integrand": integrand, "variable": var}

    def _prepare_diff(self, expr: sp.Expr, assumptions: Optional[Dict[str, object]]) -> Dict[str, object]:
        if isinstance(expr, sp.Derivative):
            base_expr = expr.expr
            variables = list(expr.variables)
        else:
            base_expr = expr
            variables = sorted(base_expr.free_symbols, key=lambda s: s.name)
        if not variables:
            raise RouterError("Differentiation requires at least one variable")
        var = variables[0]
        problem = from_sympy(base_expr, objective=Objective.DIFFERENTIATE, variables=[var], assumptions=assumptions)
        return {"problem": problem, "base": base_expr, "variable": var}

    def _prepare_solve(self, expr: sp.Expr, assumptions: Optional[Dict[str, object]]) -> Dict[str, object]:
        if isinstance(expr, sp.Equality):
            lhs, rhs = expr.lhs, expr.rhs
        else:
            raise RouterError("Solving expects an equation with '=' sign")
        variables = sorted(expr.free_symbols, key=lambda s: s.name)
        if len(variables) != 1:
            raise RouterError("Only single-variable equations are supported in this sprint")
        var = variables[0]
        problem = from_sympy(lhs - rhs, objective=Objective.SOLVE, variables=[var], assumptions=assumptions)
        return {"problem": problem, "lhs": lhs, "rhs": rhs, "variable": var}

    def _execute_integrate(self, prepared: Dict[str, object]) -> tuple[sp.Expr, str]:
        mathcore = self._load_mathcore()
        integrand = prepared["integrand"]
        var = prepared["variable"]
        try:
            result_str = mathcore.integrate(expr_to_mathcore_string(integrand), str(var))
            return sp.sympify(result_str), "mathcore"
        except RuntimeError:
            return sp.integrate(integrand, var), "sympy"

    def _execute_diff(self, prepared: Dict[str, object]) -> tuple[sp.Expr, str]:
        mathcore = self._load_mathcore()
        base_expr = prepared["base"]
        var = prepared["variable"]
        try:
            result_str = mathcore.diff(expr_to_mathcore_string(base_expr), str(var))
            return sp.sympify(result_str), "mathcore"
        except RuntimeError:
            return sp.diff(base_expr, var), "sympy"

    def _execute_solve(self, prepared: Dict[str, object]) -> tuple[List[sp.Expr], str]:
        mathcore = self._load_mathcore()
        lhs = prepared["lhs"]
        rhs = prepared["rhs"]
        var = prepared["variable"]
        try:
            result_str = mathcore.solve_equation(expr_to_mathcore_string(lhs), expr_to_mathcore_string(rhs), str(var))
            try:
                parsed = ast.literal_eval(result_str)
                if not isinstance(parsed, list):
                    parsed = [parsed]
            except (ValueError, SyntaxError):
                parsed = [result_str]
            solutions = [sp.sympify(item) for item in parsed]
            return solutions, "mathcore"
        except RuntimeError:
            solutions = sp.solve(sp.Eq(lhs, rhs), var)
            if not isinstance(solutions, list):
                solutions = [solutions]
            return solutions, "sympy"

    def _verify_integrate(self, prepared: Dict[str, object], candidate: sp.Expr) -> VerificationResult:
        var = prepared["variable"]
        integrand = prepared["integrand"]
        mir_expr = prepared["problem"].expr
        derivative = sp.diff(candidate, var)
        return verify_all(mir_expr, derivative, reference_expr=integrand, unit_subject=candidate)

    def _verify_diff(self, prepared: Dict[str, object], candidate: sp.Expr) -> VerificationResult:
        var = prepared["variable"]
        base_expr = prepared["base"]
        expected = sp.diff(base_expr, var)
        mir_expr = prepared["problem"].expr
        return verify_all(mir_expr, candidate, reference_expr=expected)

    def _verify_solve(self, prepared: Dict[str, object], candidates: List[sp.Expr]) -> VerificationResult:
        lhs = prepared["lhs"]
        rhs = prepared["rhs"]
        var = prepared["variable"]
        symbolic_checks: List[bool] = []
        numeric_checks: List[bool] = []
        for solution in candidates:
            subs_lhs = lhs.subs(var, solution)
            subs_rhs = rhs.subs(var, solution)
            symbolic_checks.append(symbolic_equal(subs_lhs, subs_rhs))
            numeric_checks.append(sp.simplify(subs_lhs - subs_rhs) == 0)
        unit_result, unit_env = unit_check(prepared["problem"].expr, prepared["problem"].expr.sympy_expr)
        details = {
            "symbolic_checks": symbolic_checks,
            "numeric_checks": numeric_checks,
            "unit_status": {
                "status": unit_result.status,
                "issues": unit_result.issues,
                "dimensionality": str(unit_result.dimensionality),
                "warnings": unit_result.warnings,
            },
            "unit_domains": unit_env.symbol_domains,
        }
        return VerificationResult(
            symbolic=all(symbolic_checks),
            numeric=all(numeric_checks),
            units=unit_result.status,
            details=details,
        )

    def route(self, request: RouterRequest) -> RouterResponse:
        timings: Dict[str, float] = {}
        total_start = time.perf_counter()
        parse_start = total_start
        parse_result = parse_expression_from_input(request.latex)
        timings["parse"] = (time.perf_counter() - parse_start) * 1000
        expr = parse_result.expression
        objective = self._detect_objective(request, parse_result, expr)
        metadata: Dict[str, object] = {
            "raw_input": parse_result.raw_input,
            "objective": objective.value,
            "mode": request.mode,
        }
        if request.assumptions:
            metadata["assumptions"] = request.assumptions
        eng_mode = self._is_engineering_mode(request.mode)
        eng_payload: Optional[Dict[str, object]] = None
        planner_enabled = self._planner_requested(request)
        planner_data: Optional[Dict[str, Any]] = None
        if objective is Objective.INTEGRATE:
            prepared = self._prepare_integrate(expr, request.assumptions)
            metadata["variable"] = str(prepared["variable"])
            if planner_enabled:
                planner_start = time.perf_counter()
                planner_data = self._run_planner_flow(prepared["problem"], request, objective, parse_result.raw_input)
                timings["planner_total"] = (time.perf_counter() - planner_start) * 1000
                if planner_data and planner_data.get("ok"):
                    return self._build_planner_response(objective, planner_data, metadata, timings)
            start = time.perf_counter()
            candidate_expr, engine = self._execute_integrate(prepared)
            timings["execute"] = (time.perf_counter() - start) * 1000
            verification = self._verify_integrate(prepared, candidate_expr)
            latex_out = sp.latex(candidate_expr)
            sympy_out = str(candidate_expr)
            checks = {
                "symbolic": verification.symbolic,
                "numeric": verification.numeric,
                "units": verification.units,
            }
            metadata["details"] = verification.details
            metadata["unit_status"] = verification.details.get("unit_status")
            metadata["engine"] = engine
            if planner_data and not planner_data.get("ok"):
                metadata.update(planner_data.get("metadata", {}))
            if eng_mode:
                unit_domains = verification.details.get("unit_domains", {})
                symbols = self._symbols_for_codegen(prepared["problem"], candidate_expr, objective)
                codegen_start = time.perf_counter()
                eng_payload = self._build_engineering_payload(
                    candidate_expr,
                    symbols,
                    unit_domains,
                    emit_c=request.emit_c_stub,
                    samples=max(1, request.sample_points),
                )
                eng_payload["unit_status"] = verification.details.get("unit_status")
                eng_payload["engine"] = engine
                timings["codegen"] = (time.perf_counter() - codegen_start) * 1000
            ok = verification.ok
        elif objective is Objective.DIFFERENTIATE:
            prepared = self._prepare_diff(expr, request.assumptions)
            metadata["variable"] = str(prepared["variable"])
            if planner_enabled:
                planner_start = time.perf_counter()
                planner_data = self._run_planner_flow(prepared["problem"], request, objective, parse_result.raw_input)
                timings["planner_total"] = (time.perf_counter() - planner_start) * 1000
                if planner_data and planner_data.get("ok"):
                    return self._build_planner_response(objective, planner_data, metadata, timings)
            start = time.perf_counter()
            candidate_expr, engine = self._execute_diff(prepared)
            timings["execute"] = (time.perf_counter() - start) * 1000
            verification = self._verify_diff(prepared, candidate_expr)
            latex_out = sp.latex(candidate_expr)
            sympy_out = str(candidate_expr)
            checks = {
                "symbolic": verification.symbolic,
                "numeric": verification.numeric,
                "units": verification.units,
            }
            metadata["details"] = verification.details
            metadata["unit_status"] = verification.details.get("unit_status")
            metadata["engine"] = engine
            if planner_data and not planner_data.get("ok"):
                metadata.update(planner_data.get("metadata", {}))
            if eng_mode:
                unit_domains = verification.details.get("unit_domains", {})
                symbols = self._symbols_for_codegen(prepared["problem"], candidate_expr, objective)
                codegen_start = time.perf_counter()
                eng_payload = self._build_engineering_payload(
                    candidate_expr,
                    symbols,
                    unit_domains,
                    emit_c=request.emit_c_stub,
                    samples=max(1, request.sample_points),
                )
                eng_payload["unit_status"] = verification.details.get("unit_status")
                eng_payload["engine"] = engine
                timings["codegen"] = (time.perf_counter() - codegen_start) * 1000
            ok = verification.ok
        elif objective is Objective.SOLVE:
            prepared = self._prepare_solve(expr, request.assumptions)
            metadata["variable"] = str(prepared["variable"])
            if planner_enabled:
                planner_start = time.perf_counter()
                planner_data = self._run_planner_flow(prepared["problem"], request, objective, parse_result.raw_input)
                timings["planner_total"] = (time.perf_counter() - planner_start) * 1000
                if planner_data and planner_data.get("ok"):
                    return self._build_planner_response(objective, planner_data, metadata, timings)
            start = time.perf_counter()
            solutions, engine = self._execute_solve(prepared)
            timings["execute"] = (time.perf_counter() - start) * 1000
            verification = self._verify_solve(prepared, solutions)
            latex_out = ", ".join(sp.latex(sol) for sol in solutions)
            sympy_out = "[" + ", ".join(str(sol) for sol in solutions) + "]"
            checks = {
                "symbolic": verification.symbolic,
                "numeric": verification.numeric,
                "units": verification.units,
            }
            metadata["solutions"] = [str(sol) for sol in solutions]
            metadata["details"] = verification.details
            metadata["unit_status"] = verification.details.get("unit_status")
            metadata["engine"] = engine
            if planner_data and not planner_data.get("ok"):
                metadata.update(planner_data.get("metadata", {}))
            if eng_mode and solutions:
                eng_expr = sp.Matrix(solutions) if len(solutions) > 1 else solutions[0]
                unit_domains = verification.details.get("unit_domains", {})
                symbols = self._symbols_for_codegen(prepared["problem"], eng_expr, objective)
                codegen_start = time.perf_counter()
                eng_payload = self._build_engineering_payload(
                    eng_expr,
                    symbols,
                    unit_domains,
                    emit_c=request.emit_c_stub,
                    samples=max(1, request.sample_points),
                )
                eng_payload["solutions"] = [str(sol) for sol in solutions]
                eng_payload["unit_status"] = verification.details.get("unit_status")
                eng_payload["engine"] = engine
                timings["codegen"] = (time.perf_counter() - codegen_start) * 1000
            ok = verification.ok
        else:
            raise RouterError("The 'prove' objective is not implemented in this sprint")
        
        explanation_payload: Optional[Dict[str, object]] = None
        if request.explain:
            explain_start = time.perf_counter()
            try:
                style = ExplanationStyle(request.style)
            except ValueError:
                style = ExplanationStyle.ACADEMIC
            
            code_preview = eng_payload.get("numpy_fn_preview") if eng_payload else None
            explanation_result = self._generate_explanation(
                problem_latex=request.latex,
                result_latex=latex_out,
                style=style,
                code_preview=code_preview,
                objective=objective.value,
                symbols=expr.free_symbols,
            )
            
            explain_ms = (time.perf_counter() - explain_start) * 1000
            timings["explain"] = explain_ms
            
            explanation_payload = {
                "style": style.value,
                "text": explanation_result.text,
                "guard": {
                    "changed": not explanation_result.guard_passed,
                    "redrafts": explanation_result.redrafts,
                },
                "cached": explanation_result.cached,
                "latency_ms": explanation_result.latency_ms if not explanation_result.cached else explain_ms,
            }
        
        timings["total"] = (time.perf_counter() - total_start) * 1000
        return RouterResponse(
            ok=ok,
            objective=objective,
            latex_out=latex_out,
            sympy_out=sympy_out,
            checks=checks,
            timings_ms=timings,
            metadata=metadata,
            eng=eng_payload,
            planner=planner_data.get("payload") if planner_data else None,
            concise=planner_data.get("concise") if planner_data else None,
            explanation=explanation_payload,
        )
