from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Tuple

import sympy as sp
from pint import UnitRegistry


class UnitError(RuntimeError):
    pass


@dataclass(frozen=True)
class SymbolSpec:
    name: str
    unit: str
    domain: Optional[Tuple[float, float]] = None


@dataclass
class UnitEnvironment:
    registry: UnitRegistry
    specs: Dict[str, SymbolSpec]
    symbol_dims: Dict[str, object]
    symbol_domains: Dict[str, Optional[Tuple[float, float]]]
    warnings: List[str]

    @property
    def dimensionless(self) -> object:
        return self.registry.dimensionless.dimensionality


@dataclass
class UnitCheckResult:
    ok: bool
    status: str
    issues: List[str]
    dimensionality: object
    warnings: List[str]


@lru_cache(maxsize=1)
def get_registry() -> UnitRegistry:
    registry = UnitRegistry()
    registry.default_system = "SI"
    return registry


def _parse_assumptions(raw: Optional[Dict[str, object]]) -> Dict[str, SymbolSpec]:
    specs: Dict[str, SymbolSpec] = {}
    if not raw:
        return specs
    for name, payload in raw.items():
        if isinstance(payload, str):
            unit_value = payload
            domain_value: Optional[Tuple[float, float]] = None
        elif isinstance(payload, dict):
            unit_candidate = payload.get("unit")
            if unit_candidate is None:
                raise UnitError(f"missing unit for symbol {name}")
            unit_value = unit_candidate
            domain_raw = payload.get("domain")
            if domain_raw is None:
                domain_value = None
            else:
                if not isinstance(domain_raw, (list, tuple)) or len(domain_raw) != 2:
                    raise UnitError(f"invalid domain specification for {name}")
                domain_value = (float(domain_raw[0]), float(domain_raw[1]))
        else:
            raise UnitError(f"invalid assumption payload for {name}")
        specs[name] = SymbolSpec(name=name, unit=unit_value, domain=domain_value)
    return specs


def build_environment(symbols: Iterable[sp.Symbol], assumptions: Optional[Dict[str, object]]) -> UnitEnvironment:
    registry = get_registry()
    specs = _parse_assumptions(assumptions)
    symbol_dims: Dict[str, object] = {}
    symbol_domains: Dict[str, Optional[Tuple[float, float]]] = {}
    warnings: List[str] = []
    for symbol in symbols:
        spec = specs.get(symbol.name)
        if spec is None:
            symbol_dims[symbol.name] = registry.dimensionless.dimensionality
            symbol_domains[symbol.name] = None
            warnings.append(f"symbol {symbol} treated as dimensionless")
            continue
        try:
            quantity = registry.Quantity(1, spec.unit)
        except Exception as exc:
            raise UnitError(f"invalid unit '{spec.unit}' for symbol {symbol}") from exc
        symbol_dims[symbol.name] = quantity.dimensionality
        symbol_domains[symbol.name] = spec.domain
    return UnitEnvironment(registry=registry, specs=specs, symbol_dims=symbol_dims, symbol_domains=symbol_domains, warnings=warnings)


def _ensure_equal(lhs: object, rhs: object, message: str) -> object:
    if lhs != rhs:
        raise UnitError(message)
    return lhs


def _combine_mul(dimensions: List[object], registry: UnitRegistry) -> object:
    result = registry.dimensionless.dimensionality
    for dim in dimensions:
        result = result * dim
    return result


def _pow_dimension(base_dim: object, exponent: sp.Expr, registry: UnitRegistry, env: UnitEnvironment) -> object:
    if exponent.is_Number:
        return base_dim ** float(exponent)
    exponent_dim = _infer_dimension(exponent, env)
    if exponent_dim != registry.dimensionless.dimensionality:
        raise UnitError("exponent must be dimensionless")
    return base_dim


def _function_dimension(func: sp.FunctionClass, args: Tuple[sp.Expr, ...], env: UnitEnvironment) -> object:
    registry = env.registry
    if func in {sp.sin, sp.cos, sp.tan, sp.asin, sp.acos, sp.atan, sp.sinh, sp.cosh, sp.tanh, sp.exp, sp.log}:
        arg_dim = _infer_dimension(args[0], env)
        if arg_dim != registry.dimensionless.dimensionality:
            raise UnitError("function argument must be dimensionless")
        return registry.dimensionless.dimensionality
    if func in {sp.sqrt}:
        arg_dim = _infer_dimension(args[0], env)
        return arg_dim ** 0.5
    return _infer_dimension(args[0], env)


def _infer_dimension(expr: sp.Expr, env: UnitEnvironment) -> object:
    registry = env.registry
    if expr is sp.S.NaN:
        return registry.dimensionless.dimensionality
    if expr.is_Number:
        return registry.dimensionless.dimensionality
    if expr.is_Symbol:
        return env.symbol_dims.get(expr.name, registry.dimensionless.dimensionality)
    if isinstance(expr, sp.Equality):
        left_dim = _infer_dimension(expr.lhs, env)
        right_dim = _infer_dimension(expr.rhs, env)
        _ensure_equal(left_dim, right_dim, "equation sides must share dimensions")
        return registry.dimensionless.dimensionality
    if expr.is_Add:
        dims = [_infer_dimension(arg, env) for arg in expr.args]
        base = dims[0]
        for candidate in dims[1:]:
            _ensure_equal(base, candidate, "addition operands must share dimensions")
        return base
    if expr.is_Mul:
        dims = [_infer_dimension(arg, env) for arg in expr.args]
        return _combine_mul(dims, registry)
    if expr.is_Pow:
        base_dim = _infer_dimension(expr.base, env)
        return _pow_dimension(base_dim, expr.exp, registry, env)
    if expr.is_Function:
        return _function_dimension(expr.func, expr.args, env)
    if expr.is_Matrix:
        entries = [
            _infer_dimension(item, env)
            for item in expr
        ]
        base = entries[0] if entries else registry.dimensionless.dimensionality
        for candidate in entries[1:]:
            _ensure_equal(base, candidate, "matrix entries must share dimensions")
        return base
    return registry.dimensionless.dimensionality


def check_dimensions(expr: sp.Expr, env: UnitEnvironment) -> UnitCheckResult:
    try:
        dimensionality = _infer_dimension(expr, env)
        status = "ok"
        issues: List[str] = []
    except UnitError as exc:
        dimensionality = env.registry.dimensionless.dimensionality
        status = "error"
        issues = [str(exc)]
        return UnitCheckResult(ok=False, status=status, issues=issues, dimensionality=dimensionality, warnings=env.warnings)
    issues = env.warnings.copy()
    ok = status == "ok"
    if issues:
        status = "warning"
    return UnitCheckResult(ok=ok, status=status, issues=issues, dimensionality=dimensionality, warnings=env.warnings)


def resolve_symbol_domain(name: str, env: UnitEnvironment) -> Optional[Tuple[float, float]]:
    return env.symbol_domains.get(name)


def dimension_as_string(dimensionality: object) -> str:
    return str(dimensionality)