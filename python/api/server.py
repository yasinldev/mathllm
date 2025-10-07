from __future__ import annotations

import time
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..src.mathllm.router import MathRouter, RouterError, RouterRequest

app = FastAPI(title="MathLLM Core", version="0.2.0")
router = MathRouter()


class SolveRequest(BaseModel):
    latex: str = Field(..., description="Input LaTeX or textual expression")
    mode: str = Field("academic", pattern="^(academic|eng)$")
    objective: Optional[str] = Field(None, description="Optional objective override")
    assumptions: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional symbol assumptions including units and domains",
    )
    emit_c_stub: bool = Field(True, description="Emit C stub in engineering payload")
    sample_points: int = Field(
        4,
        ge=1,
        le=16,
        description="Number of sample points for engineering previews",
    )
    mode_params: Optional[Dict[str, Any]] = Field(
        None,
        description="Experimental mode parameters (e.g., planner toggles)",
    )
    concise: bool = Field(True, description="Return concise response payload")
    verbose: bool = Field(False, description="Include verbose planner artifacts")
    concise_max_chars: Optional[int] = Field(
        None,
        ge=64,
        le=1024,
        description="Override default concise character budget",
    )
    explain: bool = Field(True, description="Generate human-friendly explanation")
    style: str = Field("academic", description="Explanation style: academic, friendly, oral_exam, one_minute")


class SolveResponse(BaseModel):
    ok: bool
    objective: str
    latex_out: str
    sympy_out: str
    verified: bool
    checks: Dict[str, Any]
    timings_ms: Dict[str, float]
    metadata: Dict[str, Any]
    eng: Optional[Dict[str, Any]]
    planner: Optional[Dict[str, Any]]
    concise: Optional[Dict[str, Any]]
    explanation: Optional[Dict[str, Any]]


@app.post("/solve", response_model=SolveResponse)
async def solve(payload: SolveRequest) -> SolveResponse:
    request = RouterRequest(
        latex=payload.latex,
        mode=payload.mode,
        objective=payload.objective,
        assumptions=payload.assumptions,
        emit_c_stub=payload.emit_c_stub,
        sample_points=payload.sample_points,
        mode_params=payload.mode_params,
        concise=payload.concise,
        verbose=payload.verbose,
        concise_max_chars=payload.concise_max_chars,
        explain=payload.explain,
        style=payload.style,
    )
    try:
        start = time.perf_counter()
        response = router.route(request)
        total_ms = (time.perf_counter() - start) * 1000
    except RouterError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    timings = dict(response.timings_ms)
    timings.setdefault("total", total_ms)
    return SolveResponse(
        ok=response.ok,
        objective=response.objective.value,
        latex_out=response.latex_out,
        sympy_out=response.sympy_out,
        verified=response.ok,
        checks=response.checks,
        timings_ms=timings,
        metadata=response.metadata,
        eng=response.eng,
        planner=response.planner,
        concise=response.concise,
        explanation=response.explanation,
    )
