from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import sympy as sp

from .mir import MIRExpr
from .units import build_environment, check_dimensions, UnitEnvironment, UnitCheckResult, dimension_as_string

__all__ = [
    "VerificationError",
    "VerificationResult",
    "symbolic_equal",
    "numeric_probe",
    "unit_check",
    "verify_all",
]


class VerificationError(RuntimeError):
    pass


@dataclass(frozen=True)
class VerificationResult:
    symbolic: bool
    numeric: bool
    units: Optional[str]
    details: Dict[str, object]

    @property
    def ok(self) -> bool:
        if self.units == "error":
            return False
        return self.symbolic and self.numeric


def _import_mathcore(module_path: Optional[str] = None):
    if "mathcore" in sys.modules:
        return sys.modules["mathcore"]
    search_paths = []
    if module_path:
        search_paths.append(module_path)
    package_dir = os.path.dirname(__file__)
    src_dir = os.path.dirname(package_dir)
    python_dir = os.path.dirname(src_dir)
    project_root = os.path.dirname(python_dir)
    candidate_paths = [
        os.path.join(python_dir, "cpp", "build"),
        os.path.join(project_root, "cpp", "build"),
    ]
    search_paths.extend(candidate_paths)
    for path in search_paths:
        if path and path not in sys.path and os.path.isdir(path):
            sys.path.insert(0, path)
    try:
        return importlib.import_module("mathcore")
    except ModuleNotFoundError as exc:
        raise VerificationError("mathcore module is not available. Did you build the C++ core?") from exc


def symbolic_equal(lhs: sp.Expr, rhs: sp.Expr) -> bool:
    simplified_lhs = sp.simplify(lhs)
    simplified_rhs = sp.simplify(rhs)
    lhs_str = str(simplified_lhs)
    rhs_str = str(simplified_rhs)
    try:
        mathcore = _import_mathcore()
    except VerificationError:
        mathcore = None
    if mathcore is not None:
        try:
            if mathcore.verify_equal(lhs_str, rhs_str):
                return True
        except RuntimeError:
            pass
    return bool(sp.simplify(simplified_lhs - simplified_rhs) == 0)


def _generate_numeric_samples(symbols: List[sp.Symbol], trials: int, default_domain: Tuple[float, float],
                              domains: Optional[Dict[str, Optional[Tuple[float, float]]]]) -> np.ndarray:
    low, high = default_domain
    if low >= high:
        raise VerificationError("Invalid numeric probe domain")
    rng = np.random.default_rng(seed=1234)
    samples = np.empty((trials, len(symbols)))
    for idx, symbol in enumerate(symbols):
        bounds = default_domain
        if domains:
            override = domains.get(symbol.name)
            if override is not None:
                if override[0] >= override[1]:
                    raise VerificationError(f"Invalid domain bounds for symbol {symbol}")
                bounds = override
        samples[:, idx] = rng.uniform(bounds[0], bounds[1], size=trials)
    return samples


def numeric_probe(lhs: sp.Expr, rhs: sp.Expr, symbols: Iterable[sp.Symbol], *, trials: int = 5,
                  domain: Tuple[float, float] = (0.5, 2.0),
                  domains: Optional[Dict[str, Optional[Tuple[float, float]]]] = None) -> bool:
    symbol_list = list(symbols)
    if not symbol_list:
        return bool(sp.simplify(lhs - rhs) == 0)
    samples = _generate_numeric_samples(symbol_list, trials, domain, domains)
    lhs_func = sp.lambdify(symbol_list, lhs, "numpy")
    rhs_func = sp.lambdify(symbol_list, rhs, "numpy")
    try:
        lhs_vals = lhs_func(*samples.T)
        rhs_vals = rhs_func(*samples.T)
    except Exception:
        return False
    return np.allclose(lhs_vals, rhs_vals, atol=1e-6, rtol=1e-5, equal_nan=True)


def unit_check(problem_expr: MIRExpr, candidate_expr: sp.Expr) -> Tuple[UnitCheckResult, UnitEnvironment]:
    env = build_environment(problem_expr.free_symbols, problem_expr.assumptions)
    result = check_dimensions(candidate_expr, env)
    return result, env


def verify_all(problem_expr: MIRExpr, candidate_expr: sp.Expr, reference_expr: Optional[sp.Expr] = None,
               *, numeric_trials: int = 5, unit_subject: Optional[sp.Expr] = None) -> VerificationResult:
    expr_to_compare = reference_expr if reference_expr is not None else problem_expr.sympy_expr
    symbolic_result = symbolic_equal(candidate_expr, expr_to_compare)
    unit_expr = unit_subject if unit_subject is not None else candidate_expr
    unit_result, unit_env = unit_check(problem_expr, unit_expr)
    numeric_result = numeric_probe(candidate_expr, expr_to_compare, problem_expr.free_symbols,
                                   trials=numeric_trials, domains=unit_env.symbol_domains)
    detail = {
        "candidate": str(candidate_expr),
        "reference": str(expr_to_compare),
        "symbols": [str(sym) for sym in problem_expr.free_symbols],
        "unit_status": {
            "status": unit_result.status,
            "issues": unit_result.issues,
            "dimensionality": dimension_as_string(unit_result.dimensionality),
            "warnings": unit_result.warnings,
        },
        "unit_domains": unit_env.symbol_domains,
    }
    return VerificationResult(symbolic=symbolic_result, numeric=numeric_result, units=unit_result.status, details=detail)
