from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

import sympy as sp

from mathllm.explain import ExplanationResult, ExplanationStyle, TalkerClient
from mathllm.guard import GuardConfig, GuardResult, preserve_explanation, preserve_result
from mathllm.parse import parse_latex
from mathllm.policy import PolicyClient
from mathllm.router import RouterClient, RouterRequest, RouterResponse
from mathllm.verifier import VerifierClient


class SolverMode(str, Enum):
    STUDENT = "student"
    TEACHER = "teacher"
    DIRECT = "direct"


@dataclass
class OrchestratorConfig:
    use_student: bool = True
    use_teacher_fallback: bool = True
    use_explanation: bool = True
    explanation_style: str = "akademik"
    verifier_threshold: float = 0.7
    max_teacher_attempts: int = 2
    guard_config: Optional[GuardConfig] = None


@dataclass
class OrchestratorMetrics:
    solver_mode: str
    student_attempts: int
    teacher_attempts: int
    verification_passed: bool
    explanation_generated: bool
    explanation_cached: bool
    total_ms: float
    parse_ms: float
    solve_ms: float
    verify_ms: float
    explain_ms: float


@dataclass
class OrchestratorResult:
    ok: bool
    objective: str
    latex_in: str
    latex_out: str
    sympy_out: str
    verified: bool
    confidence: float
    explanation: Optional[ExplanationResult]
    metrics: OrchestratorMetrics
    error: Optional[str] = None


class Orchestrator:
    def __init__(
        self,
        policy: PolicyClient,
        router: RouterClient,
        verifier: VerifierClient,
        talker: Optional[TalkerClient] = None,
        config: Optional[OrchestratorConfig] = None,
    ):
        self.policy = policy
        self.router = router
        self.verifier = verifier
        self.talker = talker
        self.config = config or OrchestratorConfig()

    def solve(
        self,
        latex_in: str,
        objective: Optional[str] = None,
        symbols: Optional[Sequence[sp.Symbol]] = None,
        explain: Optional[bool] = None,
        style: Optional[str] = None,
    ) -> OrchestratorResult:
        t_start = time.time()
        
        explain = explain if explain is not None else self.config.use_explanation
        style = style or self.config.explanation_style
        
        parse_start = time.time()
        try:
            parsed_expr = parse_latex(latex_in)
            if symbols is None:
                symbols = sorted(parsed_expr.free_symbols, key=lambda s: s.name)
        except Exception as exc:
            return OrchestratorResult(
                ok=False,
                objective="",
                latex_in=latex_in,
                latex_out="",
                sympy_out="",
                verified=False,
                confidence=0.0,
                explanation=None,
                metrics=OrchestratorMetrics(
                    solver_mode="none",
                    student_attempts=0,
                    teacher_attempts=0,
                    verification_passed=False,
                    explanation_generated=False,
                    explanation_cached=False,
                    total_ms=(time.time() - t_start) * 1000,
                    parse_ms=(time.time() - parse_start) * 1000,
                    solve_ms=0,
                    verify_ms=0,
                    explain_ms=0,
                ),
                error=f"parse_failed: {exc}",
            )
        parse_ms = (time.time() - parse_start) * 1000
        
        if objective is None:
            objective = self.policy.infer_objective(latex_in)
        
        solve_start = time.time()
        result = self._solve_with_fallback(latex_in, objective, symbols)
        solve_ms = (time.time() - solve_start) * 1000
        
        if not result.ok:
            return OrchestratorResult(
                ok=False,
                objective=objective,
                latex_in=latex_in,
                latex_out="",
                sympy_out="",
                verified=False,
                confidence=0.0,
                explanation=None,
                metrics=OrchestratorMetrics(
                    solver_mode=result.solver_mode,
                    student_attempts=result.student_attempts,
                    teacher_attempts=result.teacher_attempts,
                    verification_passed=False,
                    explanation_generated=False,
                    explanation_cached=False,
                    total_ms=(time.time() - t_start) * 1000,
                    parse_ms=parse_ms,
                    solve_ms=solve_ms,
                    verify_ms=0,
                    explain_ms=0,
                ),
                error=result.error,
            )
        
        verify_start = time.time()
        confidence = self.verifier.verify(
            latex_in=latex_in,
            latex_out=result.latex_out,
            objective=objective,
        )
        verified = confidence >= self.config.verifier_threshold
        verify_ms = (time.time() - verify_start) * 1000
        
        explanation_result = None
        explain_ms = 0.0
        if explain and verified and self.talker:
            explain_start = time.time()
            explanation_result = self._generate_explanation(
                result.latex_out,
                latex_in,
                objective,
                style,
                symbols,
            )
            explain_ms = (time.time() - explain_start) * 1000
        
        total_ms = (time.time() - t_start) * 1000
        
        return OrchestratorResult(
            ok=True,
            objective=objective,
            latex_in=latex_in,
            latex_out=result.latex_out,
            sympy_out=result.sympy_out,
            verified=verified,
            confidence=confidence,
            explanation=explanation_result,
            metrics=OrchestratorMetrics(
                solver_mode=result.solver_mode,
                student_attempts=result.student_attempts,
                teacher_attempts=result.teacher_attempts,
                verification_passed=verified,
                explanation_generated=explanation_result is not None,
                explanation_cached=explanation_result.cached if explanation_result else False,
                total_ms=total_ms,
                parse_ms=parse_ms,
                solve_ms=solve_ms,
                verify_ms=verify_ms,
                explain_ms=explain_ms,
            ),
        )

    def _solve_with_fallback(
        self,
        latex_in: str,
        objective: str,
        symbols: Sequence[sp.Symbol],
    ) -> _SolveResult:
        student_attempts = 0
        teacher_attempts = 0
        
        if self.config.use_student:
            req = RouterRequest(
                latex=latex_in,
                objective=objective,
                mode="student",
                explain=False,
            )
            try:
                resp = self.router.solve(req)
                student_attempts = 1
                
                guard = preserve_result(
                    reference=resp.sympy_out,
                    rendered_latex=resp.latex_out,
                    symbols=symbols,
                    config=self.config.guard_config,
                )
                
                if guard.ok:
                    return _SolveResult(
                        ok=True,
                        latex_out=resp.latex_out,
                        sympy_out=str(resp.sympy_out),
                        solver_mode=SolverMode.STUDENT,
                        student_attempts=student_attempts,
                        teacher_attempts=teacher_attempts,
                    )
            except Exception:
                pass
        
        if self.config.use_teacher_fallback:
            for attempt in range(self.config.max_teacher_attempts):
                teacher_attempts = attempt + 1
                req = RouterRequest(
                    latex=latex_in,
                    objective=objective,
                    mode="teacher",
                    explain=False,
                )
                try:
                    resp = self.router.solve(req)
                    
                    guard = preserve_result(
                        reference=resp.sympy_out,
                        rendered_latex=resp.latex_out,
                        symbols=symbols,
                        config=self.config.guard_config,
                    )
                    
                    if guard.ok:
                        return _SolveResult(
                            ok=True,
                            latex_out=resp.latex_out,
                            sympy_out=str(resp.sympy_out),
                            solver_mode=SolverMode.TEACHER,
                            student_attempts=student_attempts,
                            teacher_attempts=teacher_attempts,
                        )
                except Exception:
                    continue
        
        return _SolveResult(
            ok=False,
            latex_out="",
            sympy_out="",
            solver_mode=SolverMode.STUDENT if student_attempts > 0 else SolverMode.DIRECT,
            student_attempts=student_attempts,
            teacher_attempts=teacher_attempts,
            error="all_solvers_failed",
        )

    def _generate_explanation(
        self,
        result_latex: str,
        problem_latex: str,
        objective: str,
        style: str,
        symbols: Sequence[sp.Symbol],
    ) -> Optional[ExplanationResult]:
        if not self.talker:
            return None
        
        try:
            explanation_style = ExplanationStyle(style)
        except ValueError:
            explanation_style = ExplanationStyle.AKADEMIK
        
        max_redrafts = 2
        for attempt in range(max_redrafts + 1):
            try:
                explanation = self.talker.generate(
                    problem_latex=problem_latex,
                    result_latex=result_latex,
                    objective=objective,
                    style=explanation_style,
                )
                
                guard = preserve_explanation(
                    result_latex=result_latex,
                    explanation_text=explanation.text,
                    symbols=symbols,
                    config=self.config.guard_config,
                )
                
                if guard.ok:
                    return explanation
                
                if attempt < max_redrafts:
                    explanation = self.talker.redraft(
                        original_text=explanation.text,
                        result_latex=result_latex,
                        guard_reason=guard.reason or "unknown",
                    )
                    
                    guard_redraft = preserve_explanation(
                        result_latex=result_latex,
                        explanation_text=explanation.text,
                        symbols=symbols,
                        config=self.config.guard_config,
                    )
                    
                    if guard_redraft.ok:
                        return explanation
            except Exception:
                continue
        
        return None


@dataclass
class _SolveResult:
    ok: bool
    latex_out: str
    sympy_out: str
    solver_mode: SolverMode
    student_attempts: int
    teacher_attempts: int
    error: Optional[str] = None
