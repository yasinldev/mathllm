from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import sympy as sp
from sympy.parsing.latex import parse_latex

__all__ = ["LatexParseError", "LatexParseResult", "parse_expression_from_input"]

FORBIDDEN_LATEX_TOKENS = {
    "\\input",
    "\\include",
    "\\write",
    "\\read",
    "\\open",
    "\\file",
    "\\loop",
    "\\catcode",
    "\\csname",
    "\\expandafter",
    "\\edef",
}


class LatexParseError(RuntimeError):
    pass


@dataclass(frozen=True)
class LatexParseResult:
    expression: sp.Basic
    is_latex: bool
    raw_input: str
    normalized_input: str


def _looks_like_latex(value: str) -> bool:
    stripped = value.strip()
    if not stripped:
        return False
    if "\\" in stripped:
        return True
    latex_indicators = ("\\int", "\\frac", "\\sum", "\\prod", "\\left", "\\right")
    return any(indicator in stripped for indicator in latex_indicators)


def _parse_piece(piece: str) -> sp.Basic:
    try:
        return parse_latex(piece)
    except Exception:
        return sp.sympify(piece)


def _ensure_safe_latex(value: str) -> None:
    lowered = value.lower()
    for token in FORBIDDEN_LATEX_TOKENS:
        if token in lowered:
            raise LatexParseError(f"Forbidden LaTeX token detected: {token}")


def _normalize_input(value: str) -> str:
    return " ".join(value.strip().split())


def parse_expression_from_input(value: str, *, allow_text: bool = True) -> LatexParseResult:
    if value is None:
        raise LatexParseError("Input is None")
    normalized = _normalize_input(value)
    if not normalized:
        raise LatexParseError("Input is empty")
    try:
        if _looks_like_latex(normalized):
            _ensure_safe_latex(normalized)
            expr = parse_latex(normalized)
            return LatexParseResult(expression=expr, is_latex=True, raw_input=value, normalized_input=normalized)
        if not allow_text:
            raise LatexParseError("Plain text expressions are disabled")
        try:
            expr = parse_latex(normalized)
            return LatexParseResult(expression=expr, is_latex=True, raw_input=value, normalized_input=normalized)
        except Exception:
            pass
        if "=" in normalized:
            parts = normalized.split("=")
            if len(parts) != 2:
                raise LatexParseError("Only single '=' equations are supported")
            lhs, rhs = (_parse_piece(part) for part in parts)
            expr = sp.Eq(lhs, rhs)
        else:
            expr = _parse_piece(normalized)
        return LatexParseResult(expression=expr, is_latex=False, raw_input=value, normalized_input=normalized)
    except (sp.SympifyError, ValueError) as exc:
        raise LatexParseError(f"Failed to parse expression: {exc}") from exc
