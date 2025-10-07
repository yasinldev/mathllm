from __future__ import annotations

import json
from typing import Any, Dict, Optional

import matplotlib
matplotlib.use('Agg')

import gradio as gr

from mathllm.router import MathRouter, RouterError, RouterRequest

router = MathRouter()

OBJECTIVE_CHOICES = ["auto", "integrate", "diff", "solve"]
MODE_CHOICES = ["academic", "eng"]
STYLE_CHOICES = ["academic", "friendly", "oral_exam", "one_minute"]


def _parse_assumptions(raw: str) -> Optional[Dict[str, Any]]:
    if not raw or not raw.strip():
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Assumptions must be a JSON object")
    return parsed


def solve_pipeline(
    latex: str,
    objective: str,
    mode: str,
    assumptions_json: str,
    sample_points: int,
    emit_c_stub: bool,
    concise: bool,
    verbose: bool,
    explain: bool,
    style: str,
):
    objective_value = None if objective == "auto" else objective
    try:
        assumptions = _parse_assumptions(assumptions_json)
    except ValueError as exc:
        message = f"❌ {exc}"
        empty_payload: tuple[Optional[str], Optional[str], Optional[str], Optional[Dict[str, Any]], str, str, str, str, Optional[Any], Optional[str], Optional[str], Optional[str]]
        empty_payload = (message, "", "", None, "", "", "", "", None, "", "", "")
        return empty_payload
    try:
        response = router.route(
            RouterRequest(
                latex=latex,
                mode=mode,
                objective=objective_value,
                assumptions=assumptions,
                sample_points=sample_points,
                emit_c_stub=emit_c_stub,
                concise=concise,
                verbose=verbose,
                explain=explain,
                style=style,
            )
        )
        status = "✅ Verified" if response.ok else "⚠️ Verification issues"
        engine = response.metadata.get("engine")
        full_status = f"{status} | objective={response.objective.value}"
        if engine:
            full_status += f" | engine={engine}"
        
        concise_text = ""
        if response.concise:
            concise_payload = response.concise
            concise_text = f"## Concise Answer\n\n**Result:** ${concise_payload.get('result_latex', '')}$\n\n{concise_payload.get('explanation', '')}\n\n"
            checks = concise_payload.get("checks", {})
            timings = concise_payload.get("timings_ms", {})
            concise_text += f"**Checks:** symbolic={checks.get('symbolic')}, numeric={checks.get('numeric')}, units={checks.get('units')}\n\n"
            concise_text += f"**Timings:** planner={timings.get('planner_total')}ms, attempts={timings.get('policy_attempts')}\n"
        
        planner_text = ""
        if verbose and response.planner:
            planner_data = json.dumps(response.planner, indent=2)
            planner_text = f"## Planner Details\n```json\n{planner_data}\n```"
        
        explanation_text = ""
        if response.explanation:
            exp_payload = response.explanation
            explanation_text = f"## Explanation ({exp_payload.get('style')})\n\n{exp_payload.get('text')}\n\n"
            guard_info = exp_payload.get('guard', {})
            explanation_text += f"**Guard:** changed={guard_info.get('changed')}, redrafts={guard_info.get('redrafts')}\n"
            if exp_payload.get('cached'):
                explanation_text += "**Cached:** yes\n"
        
        latex_render = f"$$ {response.latex_out} $$"
        sympy_text = response.sympy_out
        eng_payload = response.eng or {}
        unit_status = eng_payload.get("unit_status") or response.metadata.get("unit_status")
        numpy_preview = eng_payload.get("numpy_fn_preview", "")
        octave_stub = eng_payload.get("octave_stub", "")
        matlab_stub = eng_payload.get("matlab_stub", "")
        c_stub_text = eng_payload.get("c_stub", "") or ""
        sample_eval = eng_payload.get("sample_eval")
        return (
            full_status,
            latex_render,
            sympy_text,
            unit_status,
            numpy_preview,
            octave_stub,
            matlab_stub,
            c_stub_text,
            sample_eval,
            concise_text,
            planner_text,
            explanation_text,
        )
    except RouterError as exc:
        return f"❌ {exc}", "", "", None, "", "", "", "", None, "", "", ""


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="MathLLM Demo") as demo:
        gr.Markdown("# MathLLM Core Demo\nProvide a LaTeX expression and choose an objective.")
        with gr.Row():
            latex_input = gr.Textbox(label="Input LaTeX", placeholder="\\int x^2\\sin x\\,dx")
        with gr.Row():
            objective_dropdown = gr.Dropdown(choices=OBJECTIVE_CHOICES, value="auto", label="Objective")
            mode_radio = gr.Radio(choices=MODE_CHOICES, value="academic", label="Mode")
        with gr.Row():
            assumptions_input = gr.Textbox(
                label="Assumptions (JSON)",
                placeholder='{"x": {"unit": "m", "domain": [0.5, 1.0]}}',
                lines=3,
            )
        with gr.Row():
            sample_slider = gr.Slider(
                minimum=1,
                maximum=10,
                step=1,
                value=3,
                label="Sample Points",
            )
            emit_c_checkbox = gr.Checkbox(value=True, label="Emit C Stub")
        with gr.Row():
            concise_checkbox = gr.Checkbox(value=True, label="Concise Answer Mode")
            verbose_checkbox = gr.Checkbox(value=False, label="Verbose Planner Debug")
        with gr.Row():
            explain_checkbox = gr.Checkbox(value=True, label="Generate Explanation")
            style_dropdown = gr.Dropdown(choices=STYLE_CHOICES, value="academic", label="Explanation Style")
        submit = gr.Button("Solve")
        status = gr.Textbox(label="Status")
        explanation_output = gr.Markdown(label="Explanation", visible=True)
        concise_output = gr.Markdown(label="Concise Answer", visible=True)
        planner_output = gr.Markdown(label="Planner Details", visible=True)
        latex_output = gr.Markdown(label="LaTeX Output")
        sympy_output = gr.Textbox(label="SymPy Output")
        unit_output = gr.JSON(label="Unit Status")
        numpy_preview = gr.Code(label="NumPy Preview", language="python")
        octave_preview = gr.Code(label="Octave Stub", language=None)
        matlab_preview = gr.Code(label="MATLAB Stub", language=None)
        c_preview = gr.Code(label="C Stub", language="c")
        sample_json = gr.JSON(label="Sample Evaluations")
        submit.click(
            solve_pipeline,
            inputs=[
                latex_input,
                objective_dropdown,
                mode_radio,
                assumptions_input,
                sample_slider,
                emit_c_checkbox,
                concise_checkbox,
                verbose_checkbox,
                explain_checkbox,
                style_dropdown,
            ],
            outputs=[
                status,
                latex_output,
                sympy_output,
                unit_output,
                numpy_preview,
                octave_preview,
                matlab_preview,
                c_preview,
                sample_json,
                concise_output,
                planner_output,
                explanation_output,
            ],
        )
    return demo


def main() -> None:
    demo = build_interface()
    demo.launch()


if __name__ == "__main__":
    main()
