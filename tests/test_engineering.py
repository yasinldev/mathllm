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

DATASET_PATH = ROOT / "data" / "eng_examples" / "engineering.jsonl"
ROUTER = MathRouter()


def _load_engineering_payloads() -> list[dict[str, object]]:
    if not DATASET_PATH.exists():
        raise AssertionError("engineering dataset missing")
    lines = DATASET_PATH.read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


@pytest.mark.parametrize("payload", _load_engineering_payloads())
def test_engineering_mode_payload(payload: dict[str, object]) -> None:
    request = RouterRequest(
        latex=str(payload["latex"]),
        objective=payload.get("objective"),
        mode="eng",
        assumptions=payload.get("assumptions"),
        sample_points=int(payload.get("sample_points", 3)),
        emit_c_stub=bool(payload.get("emit_c_stub", True)),
    )
    response = ROUTER.route(request)
    assert response.ok, f"router failed for {payload['latex']}"
    assert response.eng is not None, "engineering payload missing"
    eng_payload = response.eng
    assert eng_payload.get("numpy_fn_preview"), "missing numpy preview"
    assert eng_payload.get("octave_stub"), "missing octave stub"
    assert eng_payload.get("matlab_stub"), "missing matlab stub"
    if request.emit_c_stub:
        assert "c_stub" in eng_payload and eng_payload["c_stub"], "missing c stub"
    samples = eng_payload.get("sample_eval")
    assert isinstance(samples, list) and samples, "sample evaluation missing"
    unit_status = eng_payload.get("unit_status") or response.metadata.get("unit_status")
    assert unit_status, "unit status missing"
    assert unit_status.get("status") in {"ok", "warning"}


def test_engineering_dataset_size() -> None:
    payloads = _load_engineering_payloads()
    assert len(payloads) >= 20
