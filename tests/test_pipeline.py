from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CPP_BUILD = ROOT / "cpp" / "build"
if CPP_BUILD.exists() and str(CPP_BUILD) not in sys.path:
    sys.path.insert(0, str(CPP_BUILD))

from mathllm.router import MathRouter, RouterRequest

ROUTER = MathRouter()


@pytest.mark.parametrize(
    "latex_expr, objective, expected_contains",
    [
        ("\\int x^{2}\\,dx", "integrate", "x**3/3"),
        ("\\frac{d}{dx}\\sin x", "diff", "cos"),
        ("x^{2}=4", "solve", "[2, -2]"),
    ],
)
def test_router_basic(latex_expr: str, objective: str, expected_contains: str) -> None:
    response = ROUTER.route(RouterRequest(latex=latex_expr, objective=objective, mode="academic"))
    assert response.ok
    assert expected_contains.replace(" ", "") in response.sympy_out.replace(" ", "")


@pytest.mark.parametrize(
    "latex_expr, objective",
    [
        ("\\int (x^{2}+\\sin x)\\,dx", "integrate"),
        ("\\frac{d}{dx}(x^{3}+\\cos x)", "diff"),
        ("x^{2}+x-6=0", "solve"),
    ],
)
def test_router_verification_pass(latex_expr: str, objective: str) -> None:
    response = ROUTER.route(RouterRequest(latex=latex_expr, objective=objective, mode="academic"))
    assert response.ok
    assert response.checks["symbolic"]


def test_dataset_coverage() -> None:
    dataset_path = ROOT / "data" / "examples" / "easy.jsonl"
    payloads = [json.loads(line) for line in dataset_path.read_text().splitlines() if line.strip()]
    successes = 0
    for payload in payloads:
        request = RouterRequest(latex=payload["latex"], objective=payload.get("objective"), mode="academic")
        response = ROUTER.route(request)
        if response.ok:
            successes += 1
    assert len(payloads) == 50
    assert successes >= 45
