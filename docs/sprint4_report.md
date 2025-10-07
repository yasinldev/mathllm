# Sprint 4 Deliverables

## Summary

Sprint 4 introduces planner-aware routing with a verifier-first policy, tool runtime, and evaluation harness. The current run executed entirely in deterministic stub mode to provide reproducible metrics without loading a large transformer checkpoint.

- **Student LLM**: Stub-enabled wrapper with optional transformer dependencies.
- **Planner & Policy**: JSON plan validation plus verifier-first retries (repair attempts disabled for stub runs).
- **Evaluation Harness**: `python/src/mathllm/evaluation.py` and `eval/scripts/run_bench.py` automate bench execution and reporting.
- **Artifacts**: Evaluation summaries in `eval/runs/`, raw policy logs in `runs/`, sample `/solve` response via router planner mode.

## Evaluation Results

Command:

```bash
STUDENT_MODEL=stub \
/home/yasinldev/Documents/mathllm/.venv/bin/python \
  eval/scripts/run_bench.py \
  --stub \
  --max-repairs 0 \
  --consistency 1 \
  --output eval/runs/stub_run_sr.json
```

Key metrics (`eval/runs/stub_run_sr.json`):

| Bench | Successes | Total | Success Rate | Avg runtime per attempt |
| --- | --- | --- | --- | --- |
| easy | 6 | 10 | 60% | 46.63 ms |
| hard | 2 | 10 | 20% | 38.20 ms |

Observations:

- Integrals of simple polynomials/deterministic derivatives are reliable.
- Multi-solution equations succeed after policy verification fix, but plan-level verify still struggles with matrix binds (logged as execution parsing errors).
- Hard bench failures stem from mathcore limitations (`Unsupported integrand`, `Not a Polynomial`) and strict verify semantics—flagged as risks below.

## Planner Sample (`/solve` with planner mode)

```text
Request: RouterRequest(latex='\\int x^2\\,dx', objective='integrate', mode='academic', mode_params={'planner': True})
STUDENT_MODEL=stub
```

Response excerpt:

```json
{
  "ok": true,
  "objective": "integrate",
  "sympy_out": "x**3/3",
  "planner": {
    "plan": {
      "steps": [
        {"type": "tool_call", "tool": "integrate", "args": {"expr": "x**2", "var": "x"}, "bind": "I1"},
        {"type": "verify", "lhs": "diff(I1, x)", "rhs": "x**2"},
        {"type": "final", "result": "I1"}
      ]
    },
    "execution": {
      "metrics": {"total_time_ms": 5.964, "tool_use_rate": 1.0, "verify_success_rate": 1.0},
      "steps": [
        {"index": 0, "type": "tool_call", "output": {"bind": "I1", "expr": "x**3/3"}},
        {"index": 1, "type": "verify", "verify_flag": true},
        {"index": 2, "type": "final", "output": "I1"}
      ]
    },
    "log_path": "runs/plan_20251007T182817Z.jsonl"
  }
}
```

## Logs & Artifacts

- **Evaluation summaries**: `eval/runs/stub_run_sr.json` (primary), `eval/runs/stub_run.json` (default config, higher retries).
- **Planner logs**: Latest runs saved under `runs/plan_*.jsonl` (e.g., `runs/plan_20251007T182615Z.jsonl`).
- **Evaluation harness**: CLI runner in `eval/scripts/run_bench.py`; library utilities in `python/src/mathllm/evaluation.py`.

## Risks & Follow-ups

1. **MathCore coverage** – The C++ kernel rejects several hard-bench integrals and transcendental solves (`Unsupported integrand`, `Not a Polynomial`). Requires extending mathcore ops or adding SymPy fallbacks inside `ToolRuntime`.
2. **Matrix verify semantics** – Plan-level verifies for multi-root equations try to substitute entire matrices, causing parsing failures. Options: adjust stub plan generation to index solutions, or enhance `ToolRuntime._execute_verify` to iterate over matrix entries.
3. **Real-model validation** – Stub mode proves the pipeline, but transformer evaluations (with adapters) remain outstanding. Allocate GPU time and revisit success thresholds once a fine-tuned student model is available.

## Reproduction Checklist

1. Build mathcore (`cmake -S cpp -B cpp/build && cmake --build cpp/build`).
2. Install Python requirements (`pip install -r python/requirements.txt`).
3. Export `STUDENT_MODEL=stub` for deterministic smoke runs.
4. Execute the evaluation runner (command above) and inspect `eval/runs/*.json` plus `runs/*.jsonl`.

This report consolidates Sprint 4 deliverables, providing concrete evaluation evidence, sample router behaviour, and a focused risk list for the next sprint.
