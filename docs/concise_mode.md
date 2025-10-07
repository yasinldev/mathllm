# Concise Answer Mode

## Overview

The **Concise Answer Mode** provides a compact JSON payload optimized for interactive clients (web, mobile, voice assistants) that need quick responses without verbose planner artifacts. It includes:

- **Short explanation** (~50-200 chars) summarizing the objective and result
- **LaTeX result** (verified and guard-checked)
- **Verification checks** (symbolic, numeric, units)
- **Execution timings** (planner total, policy attempts)
- **Optional engineering snippet** (NumPy/Octave preview for integrations or derivatives)
- **Guard validation** to prevent LaTeX tampering or rendering errors

---

## When to Use

- **Client UIs** that need minimal latency and compact payloads
- **Mobile apps** with limited bandwidth or screen space
- **Voice assistants** requiring concise spoken answers
- **Production APIs** where response size matters

Use `verbose=true` alongside `concise=false` when debugging planner behavior or analyzing policy decisions.

---

## API Usage

### Request

```json
{
  "latex": "\\int x^2 \\sin x\\,dx",
  "mode": "academic",
  "objective": "integrate",
  "concise": true,
  "verbose": false,
  "concise_max_chars": 400
}
```

### Response (concise payload)

```json
{
  "ok": true,
  "objective": "integrate",
  "result_latex": "\\frac{x^{2}}{2} - \\sin(x)",
  "code_preview": "lambda x: (x**2)/2 - np.sin(x)",
  "explanation": "Verified result: antiderivative simplifies to \\frac{x^{2}}{2} - \\sin(x).",
  "verified": true,
  "checks": {
    "symbolic": true,
    "numeric": true,
    "units": "ok"
  },
  "timings_ms": {
    "planner_total": 142.3,
    "policy_attempts": 1
  },
  "execution_metrics": {
    "total_time_ms": 142.3,
    "tool_use_rate": 1.0,
    "verify_success_rate": 1.0,
    "step_count": 3,
    "tool_count": 2,
    "verify_count": 1
  }
}
```

---

## Guard Mechanism

The concise renderer invokes a **guard** that:

1. **Sanitizes** the rendered LaTeX by removing whitespace and comparing with the reference expression.
2. **Simplifies** both reference and rendered expressions symbolically via SymPy.
3. **Samples** numeric values across the problem's variable space to ensure the rendered expression matches the computed result within a configurable threshold (default `1e-6`).

If the guard detects a mismatch (e.g., LaTeX rendering bug, expression tampering), the renderer raises `ConciseError` and the client receives a standard error response.

### Guard Configuration

- `numeric_samples`: Number of random sample points (default `4`)
- `numeric_threshold`: Absolute difference tolerance (default `1e-6`)
- `sample_min` / `sample_max`: Value range for sampling (default `[-3, 3]`)

---

## UI Integration

The Gradio UI exposes:

- **Concise Answer Mode** checkbox (default `true`)
- **Verbose Planner Debug** checkbox (default `false`)

When concise mode is enabled, the UI displays:

- Compact result card with LaTeX, explanation, checks, timings
- Engineering code preview (if applicable)

When verbose mode is enabled alongside standard mode, planner artifacts (plan steps, execution logs, verification details) appear in a collapsible JSON view.

---

## FastAPI Endpoint

```python
POST /solve
```

**Parameters:**

- `concise` (bool): Enable concise payload (default `true`)
- `verbose` (bool): Include planner artifacts in response (default `false`)
- `concise_max_chars` (int, optional): Override default character budget (64-1024)

**Response Fields:**

- Standard fields: `ok`, `objective`, `latex_out`, `sympy_out`, `verified`, `checks`, `timings_ms`, `metadata`, `eng`, `planner`
- Concise payload: `concise` (dict) containing result, explanation, checks, timings, execution metrics

---

## Configuration Tuning

Adjust `ConciseConfig` in `mathllm/concise.py`:

```python
@dataclass
class ConciseConfig:
    max_characters: int = 400
    numeric_guard_samples: int = 4
    guard_threshold: float = 1e-6
    include_teacher_metadata: bool = False
```

- **max_characters**: Total budget for `result_latex + explanation` (trimmed if exceeded)
- **numeric_guard_samples**: More samples increase guard robustness but add latency (~2-5ms per sample)
- **guard_threshold**: Tighter thresholds reduce false positives but may reject legitimate simplifications
- **include_teacher_metadata**: Expose teacher LLM usage stats (useful for cost/latency analysis)

---

## Error Handling

If concise rendering fails (e.g., guard mismatch, no successful attempt), the API returns:

- Standard `RouterError` with `status_code=422`
- `detail` message: `"ConciseError: guard_failed: expression_mismatch"`

Clients should fall back to standard response fields (`latex_out`, `sympy_out`) when `concise` is null.

---

## Performance Notes

- **Latency impact**: Guard checks add ~5-15ms depending on expression complexity and sample count
- **Payload size**: Concise responses are ~60-80% smaller than verbose payloads (excludes planner artifacts, step logs)
- **Teacher latency**: If teacher LLM is used, `teacher_latency_ms` is included in concise metadata when `include_teacher_metadata=true`

---

## Example Workflows

### 1. Quick Integration

```bash
curl -X POST http://localhost:8000/solve \
  -H "Content-Type: application/json" \
  -d '{
    "latex": "\\frac{d}{dx} (x^3 + 2x)",
    "mode": "academic",
    "objective": "diff",
    "concise": true
  }'
```

Response:

```json
{
  "concise": {
    "ok": true,
    "result_latex": "3x^{2} + 2",
    "explanation": "Verified result: derivative equals 3x^{2} + 2.",
    "verified": true,
    "checks": {"symbolic": true, "numeric": true, "units": "ok"},
    "timings_ms": {"planner_total": 89.2, "policy_attempts": 1}
  }
}
```

### 2. Debugging with Verbose

```bash
curl -X POST http://localhost:8000/solve \
  -H "Content-Type: application/json" \
  -d '{
    "latex": "\\int \\sin(x) \\cos(x)\\,dx",
    "mode": "academic",
    "concise": false,
    "verbose": true
  }'
```

Response includes full `planner` field with plan steps, execution logs, verification details.

---

## Testing

Run concise tests:

```bash
PYTHONPATH=python/src pytest python/tests/test_concise.py
```

Tests cover:

- Payload length enforcement (`<= max_characters`)
- Guard detection of tampered LaTeX
- ConciseError raised on guard failure
- Numeric/symbolic check propagation

---

## Future Enhancements

- **Streaming mode**: Send partial results as planner executes steps
- **Multi-language explanations**: Generate explanations in user's preferred language
- **Adaptive budget**: Adjust `max_characters` based on client viewport size
- **Cache warming**: Pre-compute concise payloads for common problems

---

## References

- `python/src/mathllm/concise.py`: Renderer implementation
- `python/src/mathllm/guard.py`: Guard validation logic
- `python/src/mathllm/router.py`: Router integration
- `python/api/server.py`: FastAPI endpoint
- `python/ui/app.py`: Gradio UI toggle
