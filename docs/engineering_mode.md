# Engineering Mode

Engineering mode augments MathLLM's symbolic answers with ready-to-run numerical artifacts. When `mode="eng"`, the router:

- Enforces physical units using Pint and MIR assumptions.
- Generates previews for NumPy, Octave/MATLAB, and optional C99 code.
- Samples the generated NumPy function across the declared symbol domains.
- Returns status metadata so API clients and UIs can visualize unit checks, engines, and execution timings.

This document describes how to submit engineering requests, interpret responses, and extend the dataset/test harness.

## Request anatomy

The router accepts the standard fields (`latex`, optional `objective`) plus engineering-specific controls:

| Field | Type | Description |
| --- | --- | --- |
| `mode` | string | Set to `"eng"` to enable engineering payloads. Case-insensitive aliases like `"engineering"` are also accepted. |
| `assumptions` | object | Optional mapping of symbol names to unit/domain metadata. Either a unit string (`"meter"`) or an object like `{ "unit": "meter", "domain": [0.2, 1.0] }`. Domains bound numeric sampling. |
| `emit_c_stub` | bool | Default `true`. When `false`, C stubs are omitted from the response. |
| `sample_points` | int | Number of random evaluations (minimum 1, default 4). |

Assumptions are interpreted via Pint; all units must be valid members of the SI registry. Trigonometric/exponential arguments must be dimensionless—failures surface via the `units` check.

## Example API usage

```bash
curl -X POST http://127.0.0.1:8000/solve \
  -H "Content-Type: application/json" \
  -d '{
        "latex": "\\int x^{2}\\,dx",
        "mode": "eng",
        "objective": "integrate",
        "assumptions": {"x": {"unit": "meter", "domain": [0.4, 1.2]}},
        "sample_points": 3,
        "emit_c_stub": true
      }'
```

Successful responses include the standard academic payload plus an `eng` object:

```json
{
  "ok": true,
  "objective": "integrate",
  "latex_out": "\\frac{x^{3}}{3}",
  "checks": {"symbolic": true, "numeric": true, "units": "ok"},
  "metadata": {
    "engine": "mathcore",
    "unit_status": {"status": "ok", "dimensionality": "[length] ** 3"}
  },
  "eng": {
    "numpy_fn_preview": "def f(x):\n    return x**3/3",
    "octave_stub": "function y = f(x)\n  y = (x.^3) ./ 3;\nend",
    "matlab_stub": "function y = f(x)\n  y = (x.^3) ./ 3;\nend",
    "c_stub": "double f(double x) {\n  return pow(x, 3.0) / 3.0;\n}",
    "sample_eval": [
      {"x": 0.94, "y": 0.28},
      {"x": 0.52, "y": 0.05}
    ],
    "unit_status": {"status": "ok", "warnings": []},
    "symbols": ["x"],
    "engine": "mathcore"
  }
}
```

## UI expectations

The Gradio demo now exposes engineering inputs (assumptions JSON, sampling controls, optional C stub toggle) and renders the resulting code snippets, unit summaries, and sampled evaluations. Switch the mode radio to `eng` to reveal the extra fields.

## Dataset & tests

Engineering regression coverage lives in `data/eng_examples/engineering.jsonl`. The dataset currently contains 20 curated cases across integration, differentiation, and solving. Each entry specifies assumptions and sampling metadata ensuring deterministic probes.

The pytest suite includes `tests/test_engineering.py`, which validates:

- Every dataset entry produces a successful router response in engineering mode.
- NumPy/Octave/MATLAB previews are present.
- Optional C stubs obey the `emit_c_stub` flag.
- Sample evaluations and unit statuses are populated.

Run just the engineering suite:

```bash
PYTHONPATH=python/src pytest tests/test_engineering.py
```

Or execute the full pipeline with `pytest` to combine academic and engineering checks.

## Troubleshooting

- **`mathcore` import errors**: build the C++ core (`cmake -S cpp -B cpp/build && cmake --build cpp/build`). The router automatically searches both `cpp/build` roots.
- **Unit violations**: audit assumptions—ensure symbols participating in trig/exp/log are dimensionless, and additive terms share dimensionality.
- **Empty samples**: provide numeric domains in assumptions; otherwise the sampler falls back to `[0.5, 2.0]`.
