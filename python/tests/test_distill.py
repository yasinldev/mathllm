import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2] / "python" / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch")

from mathllm.distill import (DistillationDataset, DistillationExample, apply_lora_if_available,
                              examples_from_teacher_cache, load_distillation_examples,
                              save_distillation_examples)


class _FakeTokenizer:
    def __init__(self):
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"

    def __call__(self, text, truncation, max_length, padding, return_tensors):
        tokens = [min(ord(ch), 255) for ch in text][:max_length]
        attention = [1] * len(tokens)
        if padding == "max_length" and len(tokens) < max_length:
            pad_count = max_length - len(tokens)
            tokens += [0] * pad_count
            attention += [0] * pad_count
        return {
            "input_ids": torch.tensor([tokens], dtype=torch.long),
            "attention_mask": torch.tensor([attention], dtype=torch.long),
        }


def _write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_load_distillation_examples(tmp_path):
    path = tmp_path / "distill.jsonl"
    _write_jsonl(
        path,
        [
            {"prompt": "Solve x", "completion": "x"},
            {"input": "Integrate x", "response": "x**2/2"},
        ],
    )
    examples = load_distillation_examples(path)
    assert len(examples) == 2
    assert examples[0].prompt == "Solve x"
    assert examples[1].completion == "x**2/2"


def test_distillation_dataset_renders_eos(tmp_path):
    examples = [
        DistillationExample(prompt="Question", completion="Answer"),
    ]
    tokenizer = _FakeTokenizer()
    dataset = DistillationDataset(examples, tokenizer, max_length=16, pad_to_max_length=True, add_eos=True)
    sample = dataset[0]
    assert sample["input_ids"].shape == (16,)
    assert torch.equal(sample["input_ids"], sample["labels"])
    assert sample["attention_mask"].sum() >= 2


def test_examples_from_teacher_cache(tmp_path):
    cache_path = tmp_path / "teacher_cache.jsonl"
    _write_jsonl(
        cache_path,
        [
            {"prompt": "P", "response": "R"},
            {"input": "P2", "completion": "R2"},
            {"foo": "bar"},
        ],
    )
    examples = examples_from_teacher_cache(cache_path)
    assert len(examples) == 2
    assert examples[0].metadata["source"] == "teacher_cache"


def test_save_and_reload_distillation_examples(tmp_path):
    path = tmp_path / "distill.jsonl"
    examples = [
        DistillationExample(prompt="prompt", completion="completion", metadata={"score": 1.0})
    ]
    save_distillation_examples(examples, path)
    loaded = load_distillation_examples(path)
    assert loaded[0].metadata["score"] == 1.0


def test_apply_lora_without_dependency(monkeypatch):
    model = object()
    import mathllm.distill as distill_module
    monkeypatch.setattr(distill_module, "LoraConfig", None)
    monkeypatch.setattr(distill_module, "get_peft_model", None)

    assert apply_lora_if_available(model, None) is model
    with pytest.raises(RuntimeError):
        apply_lora_if_available(model, {"r": 8, "lora_alpha": 16, "target_modules": ["q_proj"]})
