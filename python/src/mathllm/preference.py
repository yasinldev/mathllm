from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence


@dataclass
class PlanAttempt:
    prompt: str
    response: str
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

    def normalized_prompt(self) -> str:
        return self.prompt.strip()


@dataclass
class PreferenceExample:
    prompt: str
    chosen: str
    rejected: Optional[str] = None
    score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "score": self.score,
            "metadata": self.metadata,
        }


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_plan_attempts(log_path: Path) -> List[PlanAttempt]:
    attempts: List[PlanAttempt] = []
    for entry in _iter_jsonl(log_path):
        plan = entry.get("plan", {})
        generation = plan.get("generation", {})
        response = plan.get("text") or generation.get("completion") or ""
        if not isinstance(response, str):
            response = json.dumps(response, ensure_ascii=False)
        attempts.append(
            PlanAttempt(
                prompt=generation.get("prompt", ""),
                response=response,
                success=bool(entry.get("success")),
                metadata={
                    "attempt": entry.get("attempt"),
                    "repair_round": entry.get("repair_round"),
                    "log_path": str(log_path),
                    "execution": entry.get("execution"),
                    "verification": entry.get("verification"),
                },
            )
        )
    return attempts


def load_plan_attempts_from_many(paths: Sequence[Path]) -> List[PlanAttempt]:
    attempts: List[PlanAttempt] = []
    for path in paths:
        attempts.extend(load_plan_attempts(path))
    return attempts


def build_preference_dataset(attempts: Iterable[PlanAttempt]) -> List[PreferenceExample]:
    grouped: Dict[str, List[PlanAttempt]] = defaultdict(list)
    for attempt in attempts:
        grouped[attempt.normalized_prompt()].append(attempt)

    preferences: List[PreferenceExample] = []
    for prompt, group in grouped.items():
        successes = [a for a in group if a.success and a.response]
        failures = [a for a in group if not a.success and a.response]
        if successes:
            chosen = _select_best_attempt(successes)
            if failures:
                for failed in failures:
                    preferences.append(
                        PreferenceExample(
                            prompt=prompt,
                            chosen=chosen.response,
                            rejected=failed.response,
                            score=1.0,
                            metadata={
                                "winner": chosen.metadata,
                                "loser": failed.metadata,
                            },
                        )
                    )
            else:
                preferences.append(
                    PreferenceExample(
                        prompt=prompt,
                        chosen=chosen.response,
                        rejected=None,
                        score=1.0,
                        metadata={"winner": chosen.metadata},
                    )
                )
    return preferences


def _select_best_attempt(attempts: Sequence[PlanAttempt]) -> PlanAttempt:
    """Select the "best" attempt given metadata (prefers lowest repair_round)."""

    return sorted(
        attempts,
        key=lambda a: (
            a.metadata.get("repair_round", 0) or 0,
            a.metadata.get("attempt", 0) or 0,
        ),
    )[0]


def save_preferences_jsonl(examples: Sequence[PreferenceExample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")


def load_preferences_jsonl(path: Path) -> List[PreferenceExample]:
    examples: List[PreferenceExample] = []
    for entry in _iter_jsonl(path):
        examples.append(
            PreferenceExample(
                prompt=entry["prompt"],
                chosen=entry["chosen"],
                rejected=entry.get("rejected"),
                score=float(entry.get("score", 1.0)),
                metadata=entry.get("metadata", {}),
            )
        )
    return examples


def preference_examples_to_pairs(examples: Sequence[PreferenceExample]) -> List[Dict[str, Any]]:
    """Return only examples containing both chosen and rejected responses."""

    pairs: List[Dict[str, Any]] = []
    for example in examples:
        if not example.chosen or not example.rejected:
            continue
        pairs.append(
            {
                "prompt": example.prompt,
                "chosen": example.chosen,
                "rejected": example.rejected,
                "score": example.score,
                "metadata": example.metadata,
            }
        )
    return pairs


def preference_examples_to_hf_dataset(examples: Sequence[PreferenceExample]):
    """Convert preference examples to a Hugging Face Dataset for DPO."""

    pairs = preference_examples_to_pairs(examples)
    if not pairs:
        raise ValueError("No preference pairs with both chosen and rejected responses")
    try:
        from datasets import Dataset  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("datasets library is required for DPO training") from exc
    records = [
        {"prompt": p["prompt"], "chosen": p["chosen"], "rejected": p["rejected"]}
        for p in pairs
    ]
    return Dataset.from_list(records)


def extract_preferences_from_eval_run(run_path: Path) -> List[PreferenceExample]:
    data = json.loads(run_path.read_text(encoding="utf-8"))
    log_paths = set()
    resolved = run_path.resolve()
    repo_root = resolved.parents[2] if len(resolved.parents) >= 3 else resolved.parent
    for bench in data.get("benches", []):
        for record in bench.get("records", []):
            logs_path = record.get("logs_path")
            if logs_path:
                candidate = Path(logs_path)
                if not candidate.is_absolute():
                    root_candidate = (repo_root / logs_path).resolve()
                    alt_candidate = (run_path.parent / logs_path).resolve()
                    candidate = root_candidate if root_candidate.exists() else alt_candidate
                log_paths.add(candidate)
    return build_preference_dataset(load_plan_attempts_from_many(sorted(log_paths)))
