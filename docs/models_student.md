# Student Model Wrapper

The student planner model is loaded via `mathllm.llm_student.StudentLLM`. It provides light orchestration around Hugging Face transformers and exposes deterministic stub behaviour for tests and evaluations.

## Configuration

`StudentLLM` is configured with the `StudentConfig` dataclass. Values are typically sourced from environment variables via `StudentConfig.from_env()`.

| Environment variable | Default | Description |
| --- | --- | --- |
| `STUDENT_MODEL` | `meta-llama/Llama-2-7b-hf` | Base model checkpoint. Use `stub` to enable the deterministic planner stub. |
| `STUDENT_ADAPTER` | _unset_ | Optional PEFT adapter to merge after the base model loads. |
| `STUDENT_DTYPE` | `bfloat16` | Torch dtype (supports `float16`, `bfloat16`, `float32`). |
| `STUDENT_MAX_NEW_TOKENS` | `512` | Maximum generated tokens per call. |
| `STUDENT_TEMPERATURE` | `0.2` | Sampling temperature used by `generate`. |
| `STUDENT_TOP_P` | `0.9` | Nucleus sampling cutoff. |
| `STUDENT_TOP_K` | `50` | Top-K sampling cutoff. |
| `STUDENT_REP_PENALTY` | `1.05` | Repetition penalty applied during sampling. |
| `STUDENT_DEVICE_MAP` | `auto` | Hugging Face device map for model shard placement. |
| `STUDENT_STUB` | `0` | When set to `1`, forces stub mode even if `STUDENT_MODEL` is not `stub`. |

Additional config values are exposed via the dataclass (pad token, EOS token, system prompt, extra generation kwargs).

## Stub Mode

Set `STUDENT_MODEL=stub` or `STUDENT_STUB=1` to bypass transformer loading. In stub mode:

- `StudentLLM.generate_plan` and `repair_plan` synthesize plans using structured heuristics derived from the MIR summary.
- `StudentLLM.generate` raises an exception (raw text sampling is unavailable).
- Plans follow the same schema as the live model, enabling deterministic planner tests.

The evaluation runner (`eval/scripts/run_bench.py`) uses stub mode by default when invoked with `--stub`.

## Planner Interface

`StudentLLM` exposes two high-level entry points:

- `generate_plan(prompt: str)` – builds a plan in response to a planner prompt assembled by `mathllm.planner.Planner`.
- `repair_plan(prompt: str)` – proposes a patched plan from verifier feedback. Temperature is reduced for predictable edits.

Both functions return a `GenerationResult` containing token counts, the decoded text, and raw metadata used by downstream logging.

## Loading a Real Model

To run the full pipeline against a fine-tuned student model:

1. Build the C++ core (see `README.md`) so mathcore is importable.
2. Export Hugging Face credentials if the checkpoint is gated.
3. Configure the environment:

```bash
export STUDENT_MODEL=path-or-hub-id
export STUDENT_ADAPTER=/path/to/adapter  # optional
export STUDENT_DTYPE=bfloat16
```

4. Start the planner-aware router or evaluation runner without `--stub`.

The wrapper automatically applies PEFT adapters (if provided), resolves pad/EOS tokens, and exposes the tokenizer via `StudentLLM.tokenizer`.

## Error Handling

- Missing PEFT adapter raises `FileNotFoundError` before model load.
- Device map and dtype mismatches bubble up from `transformers` with clear error messages.
- Stub mode safeguards guard against tokenizer/model access to prevent unexpected usage.

Refer to `python/src/mathllm/llm_student.py` for full implementation details.
