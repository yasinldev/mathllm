import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2] / "python" / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("datasets")

from mathllm.preference import (PreferenceExample, build_preference_dataset, extract_preferences_from_eval_run,
                                load_plan_attempts, load_preferences_jsonl, preference_examples_to_hf_dataset,
                                preference_examples_to_pairs, save_preferences_jsonl)


def _write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_build_preference_dataset_creates_pairs(tmp_path):
    log_path = tmp_path / "runs" / "plan.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        log_path,
        [
            {
                "attempt": 0,
                "repair_round": 0,
                "success": False,
                "plan": {
                    "text": "{\"steps\": [{\"type\": \"tool_call\"}]}",
                    "generation": {
                        "prompt": "prompt A",
                        "completion": "{\"steps\": []}"
                    }
                }
            },
            {
                "attempt": 1,
                "repair_round": 0,
                "success": True,
                "plan": {
                    "text": "{\"steps\": [{\"type\": \"final\"}]}",
                    "generation": {
                        "prompt": "prompt A",
                        "completion": "{\"steps\": [{\"type\": \"final\"}]}"
                    }
                }
            }
        ],
    )

    attempts = load_plan_attempts(log_path)
    prefs = build_preference_dataset(attempts)
    assert len(prefs) == 1
    preference = prefs[0]
    assert preference.prompt == "prompt A"
    assert preference.rejected is not None
    assert "final" in preference.chosen


def test_extract_preferences_from_eval_run(tmp_path):
    repo_root = tmp_path
    (repo_root / "runs").mkdir()
    log_path = repo_root / "runs" / "plan.jsonl"
    _write_jsonl(
        log_path,
        [
            {
                "attempt": 0,
                "repair_round": 0,
                "success": True,
                "plan": {
                    "text": "{\"steps\": [{\"type\": \"final\"}]}",
                    "generation": {
                        "prompt": "prompt B",
                        "completion": "{\"steps\": [{\"type\": \"final\"}]}"
                    }
                }
            }
        ],
    )

    eval_runs_dir = repo_root / "eval" / "runs"
    eval_runs_dir.mkdir(parents=True, exist_ok=True)
    run_path = eval_runs_dir / "stub.json"
    run_payload = {
        "benches": [
            {
                "records": [
                    {
                        "logs_path": "runs/plan.jsonl"
                    }
                ]
            }
        ]
    }
    run_path.write_text(json.dumps(run_payload), encoding="utf-8")

    prefs = extract_preferences_from_eval_run(run_path)
    assert len(prefs) == 1
    assert prefs[0].prompt == "prompt B"


def test_save_and_load_preferences(tmp_path):
    prefs = extract_preferences_from_eval_run(
        _make_eval_run_with_entries(tmp_path, prompt="prompt C", response="{\"steps\": []}")
    )
    output_path = tmp_path / "prefs.jsonl"
    save_preferences_jsonl(prefs, output_path)
    loaded = load_preferences_jsonl(output_path)
    assert len(loaded) == len(prefs)
    assert loaded[0].prompt == prefs[0].prompt


def test_preference_examples_to_pairs_filters_missing(tmp_path):
    prefs = extract_preferences_from_eval_run(
        _make_eval_run_with_entries(tmp_path, prompt="prompt D", response="{\"steps\": []}")
    )
    prefs.append(PreferenceExample(prompt="prompt D", chosen="good", rejected=None))
    pairs = preference_examples_to_pairs(prefs)
    assert all("rejected" in pair and pair["rejected"] for pair in pairs)


def test_preference_examples_to_hf_dataset_requires_dependency(tmp_path):
    prefs = _make_preference_pairs_with_success_and_failure(tmp_path)
    dataset = preference_examples_to_hf_dataset(prefs)
    assert dataset.features.keys() >= {"prompt", "chosen", "rejected"}


def _make_eval_run_with_entries(tmp_path, prompt: str, response: str) -> Path:
    repo_root = tmp_path
    (repo_root / "runs").mkdir(exist_ok=True)
    log_path = repo_root / "runs" / "plan.jsonl"
    _write_jsonl(
        log_path,
        [
            {
                "attempt": 0,
                "repair_round": 0,
                "success": True,
                "plan": {
                    "text": response,
                    "generation": {
                        "prompt": prompt,
                        "completion": response
                    }
                }
            }
        ],
    )
    eval_runs_dir = repo_root / "eval" / "runs"
    eval_runs_dir.mkdir(parents=True, exist_ok=True)
    run_path = eval_runs_dir / "stub.json"
    run_payload = {
        "benches": [
            {
                "records": [
                    {
                        "logs_path": "runs/plan.jsonl"
                    }
                ]
            }
        ]
    }
    run_path.write_text(json.dumps(run_payload), encoding="utf-8")
    return run_path


def _make_preference_pairs_with_success_and_failure(tmp_path) -> list:
    """Create preference examples with both successful and failed attempts."""
    repo_root = tmp_path
    (repo_root / "runs").mkdir(exist_ok=True)
    log_path = repo_root / "runs" / "plan.jsonl"
    _write_jsonl(
        log_path,
        [
            {
                "attempt": 0,
                "repair_round": 0,
                "success": False,
                "plan": {
                    "text": "{\"steps\": [{\"type\": \"invalid\"}]}",
                    "generation": {
                        "prompt": "prompt E",
                        "completion": "{\"steps\": [{\"type\": \"invalid\"}]}"
                    }
                }
            },
            {
                "attempt": 1,
                "repair_round": 0,
                "success": True,
                "plan": {
                    "text": "{\"steps\": [{\"type\": \"final\"}]}",
                    "generation": {
                        "prompt": "prompt E",
                        "completion": "{\"steps\": [{\"type\": \"final\"}]}"
                    }
                }
            }
        ],
    )
    eval_runs_dir = repo_root / "eval" / "runs"
    eval_runs_dir.mkdir(parents=True, exist_ok=True)
    run_path = eval_runs_dir / "stub.json"
    run_payload = {
        "benches": [
            {
                "records": [
                    {
                        "logs_path": "runs/plan.jsonl"
                    }
                ]
            }
        ]
    }
    run_path.write_text(json.dumps(run_payload), encoding="utf-8")
    return extract_preferences_from_eval_run(run_path)

