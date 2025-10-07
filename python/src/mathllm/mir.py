from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional

import sympy as sp

__all__ = ["Objective", "MIRExpr", "MIRProblem", "from_sympy", "expr_to_mathcore_string"]


class Objective(str, Enum):
    INTEGRATE = "integrate"
    DIFFERENTIATE = "diff"
    SOLVE = "solve"
    PROVE = "prove"


@dataclass(frozen=True)
class MIRExpr:
    sympy_expr: sp.Expr
    assumptions: Dict[str, str] = field(default_factory=dict)

    @property
    def free_symbols(self) -> List[sp.Symbol]:
        return sorted(self.sympy_expr.free_symbols, key=lambda s: s.name)

    def to_mathcore_string(self) -> str:
        return expr_to_mathcore_string(self.sympy_expr)


@dataclass(frozen=True)
class MIRProblem:
    objective: Objective
    expr: MIRExpr
    variables: List[sp.Symbol]
    constraints: Optional[List[sp.Expr]] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "objective": self.objective.value,
            "expr": str(self.expr.sympy_expr),
            "variables": [str(symbol) for symbol in self.variables],
            "constraints": [str(expr) for expr in self.constraints] if self.constraints else None,
            "assumptions": self.expr.assumptions,
        }


def expr_to_mathcore_string(expr: sp.Expr) -> str:
    return str(sp.simplify(expr))


def _ensure_iterable_symbols(symbols: Iterable[sp.Symbol]) -> List[sp.Symbol]:
    return sorted(set(symbols), key=lambda s: s.name)


def from_sympy(expr: sp.Expr, *, objective: Objective, assumptions: Optional[Dict[str, str]] = None,
               variables: Optional[Iterable[sp.Symbol]] = None,
               constraints: Optional[Iterable[sp.Expr]] = None) -> MIRProblem:
    mir_expr = MIRExpr(sympy_expr=expr, assumptions=assumptions or {})
    inferred_variables = _ensure_iterable_symbols(variables if variables is not None else expr.free_symbols)
    constraint_list = list(constraints) if constraints is not None else None
    return MIRProblem(objective=objective, expr=mir_expr, variables=inferred_variables, constraints=constraint_list)
