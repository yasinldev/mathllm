from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence

import sympy as sp
from sympy.parsing.latex import parse_latex


@dataclass
class GuardConfig:
    numeric_samples: int = 3
    sample_min: float = -3.0
    sample_max: float = 3.0
    numeric_threshold: float = 1e-6


@dataclass
class GuardResult:
    ok: bool
    reason: Optional[str] = None


class GuardError(RuntimeError):
    pass

def preserve_result(
    reference: Any,
    rendered_latex: str,
    symbols: Sequence[sp.Symbol],
    *,
    config: Optional[GuardConfig] = None,
) -> GuardResult:
    config = config or GuardConfig()
    normalized_reference = _normalize(reference)
    baseline = sp.latex(normalized_reference)
    if _sanitize(baseline) == _sanitize(rendered_latex):
        return GuardResult(ok=True)
    try:
        parsed = parse_latex(rendered_latex)
    except Exception as exc:
        return GuardResult(ok=False, reason=f"latex_parse_failed: {exc}")
    symbol_list = list(symbols) if symbols else sorted(normalized_reference.free_symbols, key=lambda s: s.name)
    try:
        if _expressions_equal(normalized_reference, parsed, symbol_list, config):
            return GuardResult(ok=True)
    except GuardError as exc:
        return GuardResult(ok=False, reason=str(exc))
    return GuardResult(ok=False, reason="expression_mismatch")


def _sanitize(payload: str) -> str:
    return "".join(payload.split())


def _expressions_equal(
    lhs: Any,
    rhs: Any,
    symbols: Sequence[sp.Symbol],
    config: GuardConfig,
) -> bool:
    lhs_norm = _normalize(lhs)
    rhs_norm = _normalize(rhs)
    try:
        if sp.simplify(lhs_norm - rhs_norm) == 0:
            return True
    except Exception:
        pass
    if not symbols:
        raise GuardError("no_symbols_for_numeric_check")
    samples = _sample_points(symbols, config)
    for point in samples:
        try:
            lhs_val = lhs_norm.subs(point)
            rhs_val = rhs_norm.subs(point)
            if _is_close(lhs_val, rhs_val, config.numeric_threshold):
                continue
            return False
        except Exception:
            return False
    return True


def _normalize(expr: Any) -> sp.Expr:
    if isinstance(expr, sp.MatrixBase):
        return expr
    if isinstance(expr, (list, tuple)):
        return sp.Matrix(expr)
    return sp.sympify(expr)


def _sample_points(symbols: Sequence[sp.Symbol], config: GuardConfig) -> Iterable[dict]:
    for _ in range(config.numeric_samples):
        sample = {}
        for symbol in symbols:
            value = random.uniform(config.sample_min, config.sample_max)
            if math.isclose(value, 0.0, abs_tol=1e-9):
                value += 0.5
            sample[symbol] = value
        yield sample


def _is_close(lhs, rhs, threshold: float) -> bool:
    try:
        diff = sp.N(lhs - rhs)
    except Exception:
        return False
    if hasattr(diff, "is_zero") and diff.is_zero is True:
        return True
    try:
        return abs(float(diff)) <= threshold
    except Exception:
        return False


def preserve_explanation(
    result_latex: str,
    explanation_text: str,
    symbols: Sequence[sp.Symbol],
    *,
    config: Optional[GuardConfig] = None,
) -> GuardResult:
    config = config or GuardConfig()
    
    latex_patterns = _extract_latex_from_text(explanation_text)
    if not latex_patterns:
        return GuardResult(ok=False, reason="no_latex_in_explanation")
    
    try:
        reference_expr = parse_latex(result_latex)
    except Exception as exc:
        return GuardResult(ok=False, reason=f"reference_parse_failed: {exc}")
    
    for latex_match in latex_patterns:
        try:
            parsed = parse_latex(latex_match)
            symbol_list = list(symbols) if symbols else sorted(reference_expr.free_symbols, key=lambda s: s.name)
            
            if _expressions_equal(reference_expr, parsed, symbol_list, config):
                numeric_ok = _check_numeric_values(result_latex, explanation_text, config.numeric_threshold)
                if numeric_ok:
                    return GuardResult(ok=True)
                else:
                    return GuardResult(ok=False, reason="numeric_values_altered")
        except Exception:
            continue
    
    return GuardResult(ok=False, reason="latex_result_altered")


def _extract_latex_from_text(text: str) -> List[str]:
    patterns = []
    
    inline_matches = re.findall(r'\$([^\$]+)\$', text)
    patterns.extend(inline_matches)
    
    display_matches = re.findall(r'\$\$([^\$]+)\$\$', text)
    patterns.extend(display_matches)
    
    backslash_matches = re.findall(r'\\[a-zA-Z]+\{[^}]+\}', text)
    patterns.extend(backslash_matches)
    
    frac_matches = re.findall(r'\\frac\{[^}]+\}\{[^}]+\}', text)
    patterns.extend(frac_matches)
    
    return patterns


def _check_numeric_values(result_latex: str, explanation_text: str, threshold: float) -> bool:
    result_numbers = _extract_numbers(result_latex)
    explanation_numbers = _extract_numbers(explanation_text)
    
    if not result_numbers:
        return True
    
    for num in result_numbers:
        found = False
        for exp_num in explanation_numbers:
            if abs(num - exp_num) <= threshold:
                found = True
                break
        if not found and abs(num) > threshold:
            return False
    
    return True


def _extract_numbers(text: str) -> List[float]:
    number_pattern = r'-?\d+\.?\d*'
    matches = re.findall(number_pattern, text)
    numbers = []
    for match in matches:
        try:
            numbers.append(float(match))
        except ValueError:
            continue
    return numbers
