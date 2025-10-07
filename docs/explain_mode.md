# Explanation Mode (TALKER Integration)

## Overview

The **Explanation Mode** generates human-friendly explanations for mathematical results using a dedicated LLM (TALKER). The TALKER model produces natural language explanations in multiple styles while preserving mathematical accuracy through guard validation.

**Key Features:**
- Multi-style explanations (academic, conversational, oral exam, one-minute summary)
- Result preservation guard (LaTeX + numeric validation)
- Bilingual support (TR/EN)
- Automatic redraft on guard failures
- Response caching for repeated queries

---

## Quick Start

### 1. Launch TALKER Endpoint

```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --port 8010 \
  --max-model-len 4096 \
  --api-key local
```

### 2. Configure Environment

```bash
cp env/talker.example.env env/talker.env
```

Edit `env/talker.env`:
```bash
TALKER_MODE=api
TALKER_API_BASE=http://localhost:8010/v1
TALKER_API_KEY=local
TALKER_MODEL=meta-llama/Llama-3.1-8B-Instruct
TALKER_TEMPERATURE=0.4
TALKER_MAX_TOKENS=256
```

### 3. Test API

```bash
curl -X POST http://localhost:8000/solve \
  -H "Content-Type: application/json" \
  -d '{
    "latex": "\\int x^2 dx",
    "mode": "academic",
    "explain": true,
    "style": "samimi"
  }'
```

---

## Explanation Styles

### akademik (Academic)
**Target:** Technical documentation, research papers, formal education

**Characteristics:**
- Precise mathematical terminology
- 4-7 sentences
- One brief example if needed
- Formal tone

**Example:**
```
The verified result for this integration problem is $\\frac{x^{3}}{3}$.
This follows from the power rule of integration, where $\\int x^n dx = \\frac{x^{n+1}}{n+1} + C$.
In this case, n=2, yielding the antiderivative shown. The solution has been confirmed
through symbolic verification and numeric sampling across the problem domain.
```

### samimi (Friendly)
**Target:** Tutoring, casual learning, quick reviews

**Characteristics:**
- Conversational tone
- Minimal jargon
- 4-7 sentences
- Practical examples

**Example:**
```
So basically, we integrated x squared and got $\\frac{x^{3}}{3}$ as our answer!
This makes sense because when you integrate, you're finding the area under the curve.
The power rule tells us to add 1 to the exponent (2 becomes 3) and divide by the new
exponent. Pretty straightforward once you get the hang of it!
```

### sözlü_sınav (Oral Exam)
**Target:** Exam preparation, interview practice, teaching demonstrations

**Characteristics:**
- Structured reasoning steps
- 5-7 sentences
- Mentions key concepts
- Shows thought process

**Example:**
```
To solve this integration problem, I applied the power rule for integrals.
Starting with the integrand $x^{2}$, I recognized this as a polynomial term
amenable to direct integration. The power rule states that $\\int x^n dx =
\\frac{x^{n+1}}{n+1} + C$. Applying this with n=2 gives us $\\frac{x^{3}}{3}$.
I verified this result by differentiating back to the original integrand.
This confirms the solution is mathematically sound.
```

### 1dk_özet (One-Minute Summary)
**Target:** Quick lookups, mobile apps, voice assistants

**Characteristics:**
- 2-3 short sentences
- Maximum 260 characters
- Essential information only
- No elaborate examples

**Example:**
```
Integrated $x^{2}$ using the power rule. Result: $\\frac{x^{3}}{3}$. Verified symbolically.
```

---

## Guard Mechanism

The guard ensures explanations don't alter mathematical results.

### Validation Steps

1. **LaTeX Extraction**: Extract all LaTeX expressions from explanation text using regex
2. **Parse & Compare**: Parse LaTeX with SymPy, compare with reference result
3. **Numeric Validation**: Check that numeric constants match (±1e-6 tolerance)
4. **Redraft on Failure**: If guard fails, request redraft (max 2 attempts)

### Guard Configuration

```python
from mathllm.guard import GuardConfig

config = GuardConfig(
    numeric_samples=4,        # number of sample points for validation
    numeric_threshold=1e-6,   # tolerance for numeric comparison
)
```

### Example Guard Scenarios

**✅ PASS: Correct preservation**
```
Result: x^{2} + 2x
Explanation: "The derivative is $x^{2} + 2x$..."
→ Guard extracts $x^{2} + 2x$, parses correctly, matches reference
```

**❌ FAIL: Altered LaTeX**
```
Result: x^{2}
Explanation: "The result is $x^{3}$..."  
→ Guard extracts $x^{3}$, doesn't match $x^{2}$
→ Redraft requested
```

**❌ FAIL: Changed numeric values**
```
Result: x + 5
Explanation: "The constant is 7, so $x + 7$..."
→ Numeric mismatch (5 vs 7)
→ Redraft requested
```

---

## API Usage

### Request Format

```json
{
  "latex": "\\frac{d}{dx}(x^3)",
  "mode": "academic",
  "objective": "diff",
  "explain": true,
  "style": "akademik"
}
```

**Parameters:**
- `explain` (bool, default=true): Enable explanation generation
- `style` (string, default="akademik"): One of akademik, samimi, sözlü_sınav, 1dk_özet

### Response Format

```json
{
  "ok": true,
  "objective": "diff",
  "latex_out": "3x^{2}",
  "sympy_out": "3*x**2",
  "verified": true,
  "explanation": {
    "style": "akademik",
    "text": "The verified result for this differentiation problem is $3x^{2}$. This follows from the power rule...",
    "guard": {
      "changed": false,
      "redrafts": 0
    },
    "cached": false,
    "latency_ms": 142.3
  },
  "timings_ms": {
    "parse": 8.2,
    "execute": 12.5,
    "explain": 142.3,
    "total": 165.8
  }
}
```

**Explanation Fields:**
- `style`: Style used for generation
- `text`: Generated explanation (4-260 chars depending on style)
- `guard.changed`: Whether explanation altered the result (should always be false)
- `guard.redrafts`: Number of redraft attempts (0-2)
- `cached`: Whether result came from cache
- `latency_ms`: Time to generate explanation (excludes cache hits)

---

## UI Integration

### Gradio Interface

The UI exposes:
- **Generate Explanation** checkbox (default: checked)
- **Explanation Style** dropdown (akademik, samimi, sözlü_sınav, 1dk_özet)

**Output Display:**
```
## Explanation (samimi)

So basically, we differentiated x cubed and got $3x^{2}$ as our answer!
This makes sense because the power rule says to bring down the exponent...

**Guard:** changed=false, redrafts=0
**Cached:** no
```

---

## Performance

### Latency Benchmarks

| Style | P50 | P95 | Tokens | Notes |
|-------|-----|-----|--------|-------|
| akademik | ~140ms | ~280ms | 80-120 | Detailed formal text |
| samimi | ~130ms | ~260ms | 70-110 | Conversational tone |
| sözlü_sınav | ~160ms | ~310ms | 90-140 | Longest responses |
| 1dk_özet | ~80ms | ~150ms | 30-50 | Shortest, fastest |

**Environment:** Llama-3.1-8B-Instruct, RTX 3090, vLLM batch size 1

### Guard Overhead

- LaTeX extraction: ~2ms
- SymPy parsing + comparison: ~5-10ms
- Numeric validation: ~3-8ms (depends on expression complexity)
- **Total guard overhead:** ~10-20ms

### Cache Performance

- **Hit rate:** ~40-60% for repeated problems
- **Speedup:** ~10-15x faster on cache hits (only guard validation needed)
- **Storage:** ~200KB per 1000 cached entries

---

## Configuration Tuning

### Temperature Scaling

```bash
# Conservative (more deterministic)
TALKER_TEMPERATURE=0.2

# Balanced (recommended)
TALKER_TEMPERATURE=0.4

# Creative (more varied responses)
TALKER_TEMPERATURE=0.7
```

### Token Budget

```bash
# Minimal (1dk_özet optimized)
TALKER_MAX_TOKENS=128

# Balanced (default)
TALKER_MAX_TOKENS=256

# Verbose (academic/oral exam)
TALKER_MAX_TOKENS=384
```

### Top-P Sampling

```bash
# Focused (fewer alternatives)
TALKER_TOP_P=0.8

# Balanced (recommended)
TALKER_TOP_P=0.9

# Diverse (more alternatives)
TALKER_TOP_P=0.95
```

---

## Error Handling

### Explanation Unavailable

If TALKER fails (endpoint down, timeout, etc.):

```json
{
  "explanation": {
    "text": "Explanation unavailable: Connection refused",
    "style": "akademik",
    "guard": {"changed": false, "redrafts": 0},
    "cached": false,
    "latency_ms": 50.2
  }
}
```

**Fallback behavior:** Standard response fields (latex_out, sympy_out) remain intact.

### Guard Failures

After 2 redraft attempts, if guard still fails:

```json
{
  "explanation": {
    "text": "The result is $x^{2}$...",  // Last attempt text
    "style": "samimi",
    "guard": {"changed": true, "redrafts": 2},
    "cached": false,
    "latency_ms": 420.5
  }
}
```

**Client handling:** Check `guard.changed` field; if true, discard explanation text and display only latex_out.

---

## Testing

### Run Explanation Tests

```bash
PYTHONPATH=python/src pytest python/tests/test_explain.py -v
```

**Test Coverage:**
- Guard validation (correct LaTeX, altered LaTeX, altered numerics)
- Style variations (length, tone, terminology)
- Cache behavior (hits, misses, different styles)
- Redraft logic (guard failures, correction)
- Error scenarios (no LaTeX in text, parse failures)

### Sample Test

```python
from mathllm.explain import TalkerClient, ExplanationStyle
from mathllm.guard import preserve_explanation

client = TalkerClient()
text = client.generate_explanation(
    problem_latex="\\int x dx",
    result_latex="\\frac{x^{2}}{2}",
    style=ExplanationStyle.FRIENDLY,
)

guard_result = preserve_explanation(
    result_latex="\\frac{x^{2}}{2}",
    explanation_text=text,
    symbols=[sp.Symbol("x")],
)

assert guard_result.ok  # Explanation preserves result
assert len(text) <= 400  # Reasonable length
```

---

## Monitoring

### Metrics to Track

1. **Explanation Latency** (P50/P95/P99)
2. **Guard Hit Rate** (% of explanations passing first attempt)
3. **Redraft Rate** (% requiring redrafts)
4. **Cache Hit Rate** (% served from cache)
5. **Error Rate** (% of failed generations)

### Logging

```json
{
  "timestamp": "2025-10-07T14:32:10Z",
  "explain_ms": 142.3,
  "guard_redraft": 0,
  "cached": false,
  "style": "akademik",
  "problem_hash": "a3f5e2c1",
  "guard_passed": true
}
```

---

## Best Practices

### 1. Style Selection

- **Academic papers/documentation:** `akademik`
- **Tutoring/casual learning:** `samimi`
- **Exam prep/teaching:** `sözlü_sınav`
- **Quick lookups/mobile:** `1dk_özet`

### 2. Guard Thresholds

- **Strict (research):** `numeric_threshold=1e-8`
- **Balanced (default):** `numeric_threshold=1e-6`
- **Lenient (approximations):** `numeric_threshold=1e-4`

### 3. Caching Strategy

- Enable for production (significant speedup)
- Invalidate cache when TALKER model changes
- Monitor cache size (trim if >10MB)

### 4. Error Recovery

- Always check `guard.changed` before displaying explanation
- Fall back to `latex_out` if explanation unavailable
- Log guard failures for model fine-tuning

---

## Example Workflows

### 1. Academic Integration

```bash
curl -X POST http://localhost:8000/solve \
  -H "Content-Type: application/json" \
  -d '{
    "latex": "\\int \\sin(x) \\cos(x) dx",
    "mode": "academic",
    "objective": "integrate",
    "explain": true,
    "style": "akademik"
  }'
```

Response:
```json
{
  "latex_out": "\\frac{\\sin^{2}(x)}{2}",
  "explanation": {
    "text": "The verified result for this integration is $\\frac{\\sin^{2}(x)}{2}$. This can be derived using the substitution method with u = sin(x), leading to $\\int u du = \\frac{u^2}{2}$. The solution has been confirmed through symbolic differentiation back to the original integrand.",
    "style": "akademik",
    "guard": {"changed": false, "redrafts": 0}
  }
}
```

### 2. Friendly Differentiation

```bash
curl -X POST http://localhost:8000/solve \
  -H "Content-Type: application/json" \
  -d '{
    "latex": "\\frac{d}{dx}(e^{2x})",
    "mode": "academic",
    "explain": true,
    "style": "samimi"
  }'
```

Response:
```json
{
  "latex_out": "2e^{2x}",
  "explanation": {
    "text": "So we differentiated $e^{2x}$ and got $2e^{2x}$! The chain rule kicks in here because of that 2x in the exponent. Basically, the derivative of $e^u$ is $e^u \\cdot u'$, so we bring down the 2 from 2x. Makes sense, right?",
    "style": "samimi",
    "guard": {"changed": false, "redrafts": 0}
  }
}
```

### 3. One-Minute Summary

```bash
curl -X POST http://localhost:8000/solve \
  -H "Content-Type: application/json" \
  -d '{
    "latex": "x^2 = 9",
    "mode": "academic",
    "objective": "solve",
    "explain": true,
    "style": "1dk_özet"
  }'
```

Response:
```json
{
  "latex_out": "x = -3, x = 3",
  "explanation": {
    "text": "Solved $x^{2} = 9$. Result: $x = \\pm 3$. Square root both sides.",
    "style": "1dk_özet",
    "guard": {"changed": false, "redrafts": 0}
  }
}
```

---

## Troubleshooting

### Issue: Guard failures on correct explanations

**Cause:** Overly strict numeric threshold or LaTeX formatting differences

**Solution:**
```bash
# Relax threshold
TALKER_TEMPERATURE=0.2  # More deterministic
# Or adjust guard config
numeric_threshold=1e-5
```

### Issue: Explanation too long for style

**Cause:** TALKER exceeding token budget

**Solution:**
```bash
TALKER_MAX_TOKENS=192  # Reduce for 1dk_özet
# Or enforce in prompt templates
```

### Issue: TALKER endpoint timeouts

**Cause:** Slow model inference or network latency

**Solution:**
```bash
TALKER_TIMEOUT_SECONDS=60  # Increase timeout
# Or switch to faster model
--model meta-llama/Llama-3.2-1B-Instruct  # Smaller, faster
```

---

## Future Enhancements

1. **Multi-language Support**: Detect problem language (TR/EN/ES/etc.) and generate matching explanation
2. **Streaming Explanations**: Send partial text as TALKER generates tokens
3. **Voice Output**: TTS integration for oral exam style explanations
4. **User Feedback**: Collect ratings to fine-tune TALKER prompts
5. **Adaptive Styles**: Learn user preferences over time

---

## References

- TALKER configuration: `env/talker.example.env`
- Implementation: `python/src/mathllm/explain.py`
- Guard logic: `python/src/mathllm/guard.py`
- Router integration: `python/src/mathllm/router.py`
- API endpoint: `python/api/server.py`
- Tests: `python/tests/test_explain.py`

---

**Version:** 0.2.0 (Sprint 7)  
**Last Updated:** October 7, 2025
