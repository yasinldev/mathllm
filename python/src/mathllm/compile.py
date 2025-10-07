from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import sympy as sp

# NumPy backend is optional at runtime; imports are lazy inside helpers.


@dataclass(frozen=True)
class CodegenArtifacts:
    numpy_source: str
    octave_stub: str
    matlab_stub: str
    c_stub: Optional[str]


def _symbol_names(symbols: Sequence[sp.Symbol]) -> List[str]:
    return [str(sym) for sym in symbols]


def _serialize_output(value: object) -> object:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, sp.Basic):
        try:
            evaluated = value.evalf()
            if evaluated.is_real is False:
                complex_val = complex(evaluated)
                return {"real": float(complex_val.real), "imag": float(complex_val.imag)}
            return float(evaluated)
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            data = value.tolist()
        except Exception:
            data = list(value)
        return [_serialize_output(item) for item in data]
    if isinstance(value, (list, tuple)):
        return [_serialize_output(item) for item in value]
    try:
        return float(value)
    except Exception:
        return repr(value)


def to_numpy_fn(expr: sp.Expr, symbols: Sequence[sp.Symbol]) -> Tuple[Callable, str]:
    from sympy.utilities.lambdify import lambdify

    symbol_names = [sp.Symbol(str(sym)) for sym in symbols]
    numpy_fn = lambdify(symbol_names, expr, modules="numpy")
    def_lines = [f"def f({', '.join(_symbol_names(symbols))}):"]
    def_lines.append(f"    return {sp.pycode(expr)}")
    source = "\n".join(def_lines)
    return numpy_fn, source


def _octave_header(name: str, symbols: Sequence[sp.Symbol]) -> List[str]:
    params = ", ".join(_symbol_names(symbols)) or "x"
    header = [f"function y = {name}({params})"]
    return header


def to_octave(expr: sp.Expr, symbols: Sequence[sp.Symbol], *, name: str = "f") -> str:
    from sympy.printing.octave import octave_code

    lines = _octave_header(name, symbols)
    body = octave_code(expr)
    lines.append(f"  y = {body};")
    lines.append("end")
    return "\n".join(lines)


def to_matlab_stub(expr: sp.Expr, symbols: Sequence[sp.Symbol], *, name: str = "f") -> str:
    # Octave and MATLAB share syntax at this level.
    return to_octave(expr, symbols, name=name)


def to_c_stub(expr: sp.Expr, symbols: Sequence[sp.Symbol], *, name: str = "f") -> str:
    params = [f"double {symbol}" for symbol in _symbol_names(symbols)] or ["double x"]
    from sympy.printing.c import ccode

    body = ccode(expr)
    lines = [f"double {name}({', '.join(params)}) {{"]
    lines.append(f"    return {body};")
    lines.append("}")
    return "\n".join(lines)


def generate_artifacts(expr: sp.Expr, symbols: Sequence[sp.Symbol], *, emit_c: bool = True) -> CodegenArtifacts:
    _, numpy_source = to_numpy_fn(expr, symbols)
    octave_stub = to_octave(expr, symbols)
    matlab_stub = to_matlab_stub(expr, symbols)
    c_stub = to_c_stub(expr, symbols) if emit_c else None
    return CodegenArtifacts(
        numpy_source=numpy_source,
        octave_stub=octave_stub,
        matlab_stub=matlab_stub,
        c_stub=c_stub,
    )


def sample_numpy_grid(fn: Callable, symbols: Sequence[sp.Symbol], domains: Dict[str, Optional[Tuple[float, float]]], *, samples: int = 3) -> List[Dict[str, object]]:
    import numpy as np

    symbol_names = _symbol_names(symbols)
    grid: List[Dict[str, object]] = []
    if not symbol_names:
        value = _serialize_output(fn())
        grid.append({"y": value})
        return grid
    rng = np.random.default_rng(seed=123)
    for _ in range(samples):
        inputs: List[float] = []
        entry: Dict[str, object] = {}
        for name in symbol_names:
            domain = domains.get(name)
            if domain:
                low, high = domain
            else:
                low, high = 0.5, 2.0
            value = float(rng.uniform(low, high))
            inputs.append(value)
            entry[name] = value
        result = _serialize_output(fn(*inputs))
        entry["y"] = result
        grid.append(entry)
    return grid