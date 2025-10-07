from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
PYTHON_SRC = ROOT / "python" / "src"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

from mathllm.evaluation import EvaluationSummary, load_bench, run_bench  # noqa: E402
from mathllm.llm_student import StudentConfig, StudentLLM  # noqa: E402
from mathllm.policy import PolicyConfig, VerifierFirstPolicy  # noqa: E402


def _build_student(use_stub: bool) -> StudentLLM:
    if use_stub:
        config = StudentConfig(
            model_name_or_path="stub",
            adapter_path=None,
            max_new_tokens=256,
            temperature=0.2,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.05,
            device_map="cpu",
            torch_dtype=None,
            enable_adapter=False,
            use_cache=False,
            eos_token=None,
            pad_token=None,
            system_prompt="You are MathLLM's student planner.",
            generation_kwargs={},
            use_stub=True,
        )
    else:
        config = StudentConfig.from_env()
    return StudentLLM(config)


def _build_policy(use_stub: bool, log_dir: Path, max_repairs: int, consistency: int) -> VerifierFirstPolicy:
    student = _build_student(use_stub)
    policy_config = PolicyConfig(
        max_repair_attempts=max_repairs,
        self_consistency=consistency,
        log_dir=str(log_dir),
        enable_logging=True,
    )
    return VerifierFirstPolicy(student, config=policy_config)


def _run_single(policy: VerifierFirstPolicy, bench_path: Path) -> EvaluationSummary:
    entries = load_bench(bench_path)
    bench_name = bench_path.stem
    return run_bench(policy, entries, bench_name=bench_name)


def _summaries_to_payload(summaries: List[EvaluationSummary], *, metadata: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "metadata": metadata,
        "benches": [summary.to_json() for summary in summaries],
    }


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MathLLM planner policy on evaluation benches")
    parser.add_argument(
        "--bench",
        action="append",
        dest="benches",
        type=Path,
        help="Path to a bench JSONL file. Provide multiple times to evaluate several benches.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the evaluation summary JSON.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("runs"),
        help="Directory for policy attempt logs (default: runs).",
    )
    parser.add_argument(
        "--max-repairs",
        type=int,
        default=2,
        help="Maximum number of repair attempts per plan (default: 2).",
    )
    parser.add_argument(
        "--consistency",
        type=int,
        default=3,
        help="Number of self-consistency attempts (default: 3).",
    )
    parser.add_argument(
        "--stub",
        action="store_true",
        help="Run using the deterministic student stub instead of loading a transformer.",
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    benches = args.benches or [Path("eval/benches/easy.jsonl"), Path("eval/benches/hard.jsonl")]
    summaries: List[EvaluationSummary] = []
    policy = _build_policy(args.stub, args.log_dir, args.max_repairs, args.consistency)

    for bench_path in benches:
        bench_path = bench_path if bench_path.is_absolute() else (ROOT / bench_path)
        if not bench_path.exists():
            raise FileNotFoundError(f"Bench file not found: {bench_path}")
        summary = _run_single(policy, bench_path)
        summaries.append(summary)

    metadata = {
        "stub": args.stub,
        "max_repairs": args.max_repairs,
        "self_consistency": args.consistency,
        "log_dir": str(args.log_dir),
        "student_config": "env" if not args.stub else "stub",
        "env_student_model": os.environ.get("STUDENT_MODEL"),
        "policy_teacher_stats": policy.teacher_stats(),
    }

    payload = _summaries_to_payload(summaries, metadata=metadata)
    output_path = args.output if args.output.is_absolute() else (ROOT / args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote evaluation summary to {output_path}")
    for summary in summaries:
        success_rate = summary.success_rate() * 100
        max_attempts = summary.max_attempts()
        pass1 = summary.pass_at_k(1) * 100 if max_attempts else 0.0
        passk = summary.pass_at_k(max_attempts) * 100 if max_attempts else 0.0
        teacher_attempt = summary.teacher_attempt_rate() * 100
        teacher_use = summary.teacher_use_rate() * 100
        verify_rate = summary.verify_success_rate() * 100
        teacher_latency = summary.average_teacher_latency()
        print(
            (
                f"Bench {summary.bench_name}: {summary.successes()}/{summary.total()} success ({success_rate:.1f}%); "
                f"avg runtime {summary.average_runtime():.1f} ms; pass@1 {pass1:.1f}%"
                + (f", pass@{max_attempts} {passk:.1f}%" if max_attempts > 1 else "")
                + f"; verify rate {verify_rate:.1f}%; teacher attempt {teacher_attempt:.1f}%; teacher use {teacher_use:.1f}%"
                + (f"; avg teacher latency {teacher_latency:.1f} ms" if teacher_latency else "")
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
