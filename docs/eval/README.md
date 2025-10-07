# Evaluation Framework

Sprint 4 introduces a planner-specific evaluation harness that measures success rates and runtime metrics across curated LaTeX benches.

## Components

- `python/src/mathllm/evaluation.py` – utilities for loading JSONL benches, preparing MIR problems, running the verifier-first policy, and aggregating metrics.
- `eval/scripts/run_bench.py` – CLI runner that executes the policy on one or more benches and emits a structured summary JSON file.
- `eval/benches/*.jsonl` – example benches (easy, hard) used for smoke and nightly checks.

## Running Evaluations

Activate your virtual environment and ensure `mathcore` is built (`cmake -S cpp -B cpp/build && cmake --build cpp/build`). Then execute:

```bash
python eval/scripts/run_bench.py \
  --output eval/runs/stub_latest.json \
  --stub
```

This command uses the deterministic student stub, evaluates both default benches, and writes aggregated metrics to `eval/runs/stub_latest.json`. The output contains:

- Overall metadata (timestamp, policy configuration).
- Per-bench statistics: success count, success rate, average runtime, summaries for each example.

To run against a real model, drop the `--stub` flag and export the student environment variables described in `docs/models_student.md`.

```bash
export STUDENT_MODEL=meta-llama/Llama-2-7b-hf
export STUDENT_ADAPTER=/path/to/adapter
python eval/scripts/run_bench.py \
  --output eval/runs/live_$(date +%Y%m%d).json \
  --log-dir runs/live
```

## Bench Format

Each bench is a JSONL file with one object per line:

```json
{"latex": "\\int x^2\\,dx", "objective": "integrate", "name": "poly_integral"}
```

Fields:

- `latex` (required): LaTeX or plain-text expression.
- `objective` (required): `integrate`, `diff`, or `solve`.
- `name` (optional): Human-readable identifier for reports.
- `assumptions` (optional): Passed to MIR construction for unit reasoning.
- `context` (optional): Extra prompt context forwarded to the planner.
- Additional fields are preserved in the per-record metadata.

## Logs and Artifacts

`VerifierFirstPolicy` writes attempt logs to `runs/` by default. The evaluation summary includes the final log path so you can inspect planner generations and verifier traces.

Successful evaluations should archive:

1. Summary JSON (`eval/runs/*.json`).
2. Raw planner logs (`runs/*.jsonl`).
3. A short report (see `docs/sprint4_report.md`).

## Extending Benches

- Keep objective coverage balanced: include integrate, differentiate, and solve examples.
- Provide deterministic tool outputs when possible (avoid random coefficients).
- For challenging integrals/equations, supply descriptive `name` fields for easier triage.

After editing bench files, re-run `python eval/scripts/run_bench.py --stub --output eval/runs/smoke.json` to confirm the stub pipeline still succeeds.
