from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None

    class Dataset:  # type: ignore[misc]
        """Fallback Dataset to keep type checkers satisfied when torch is absent."""
        pass

LOGGER = logging.getLogger(__name__)

try:
    from peft import LoraConfig, get_peft_model
except ImportError:  # pragma: no cover - optional dependency
    LoraConfig = get_peft_model = None


@dataclass
class DistillationExample:
    prompt: str
    completion: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "completion": self.completion,
            "metadata": self.metadata,
        }


def load_distillation_examples(path: Path) -> List[DistillationExample]:
    examples: List[DistillationExample] = []
    for entry in _iter_jsonl(path):
        prompt = entry.get("prompt") or entry.get("input")
        completion = entry.get("completion") or entry.get("response")
        if prompt is None or completion is None:
            raise ValueError(f"Malformed distillation entry: {entry}")
        examples.append(
            DistillationExample(
                prompt=str(prompt),
                completion=str(completion),
                metadata=entry.get("metadata", {}),
            )
        )
    return examples


def save_distillation_examples(examples: Sequence[DistillationExample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")


class DistillationDataset(Dataset):
    def __init__(
        self,
        examples: Sequence[DistillationExample],
        tokenizer,
        *,
        max_length: int = 2048,
        pad_to_max_length: bool = False,
        add_eos: bool = True,
        instruction_template: Optional[str] = None,
    ) -> None:
        if torch is None:
            raise RuntimeError("torch is required to instantiate DistillationDataset")
        if not examples:
            raise ValueError("DistillationDataset requires at least one example")
        self.examples = list(examples)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_max_length = pad_to_max_length
        self.add_eos = add_eos
        self.instruction_template = instruction_template or "{prompt}\n\n### Response:\n{completion}"
        self.eos = getattr(tokenizer, "eos_token", None)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if torch is None:
            raise RuntimeError("torch is required to fetch items from DistillationDataset")
        example = self.examples[index]
        text = self._render_example(example)
        tokenised = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length" if self.pad_to_max_length else False,
            return_tensors="pt",
        )
        input_ids = tokenised["input_ids"].squeeze(0)
        attention_mask = tokenised["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _render_example(self, example: DistillationExample) -> str:
        text = self.instruction_template.format(prompt=example.prompt.strip(), completion=example.completion.strip())
        if self.add_eos and self.eos and not text.endswith(self.eos):
            text = f"{text}{self.eos}"
        return text


def apply_lora_if_available(model, lora_config: Optional[Dict[str, Any]]) -> Any:
    if not lora_config:
        return model
    if LoraConfig is None or get_peft_model is None:
        raise RuntimeError("peft is required for LoRA training but is not installed")
    config = LoraConfig(**lora_config)
    LOGGER.info("Applying LoRA configuration: %s", config)
    return get_peft_model(model, config)


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def examples_from_teacher_cache(cache_path: Path) -> List[DistillationExample]:
    examples: List[DistillationExample] = []
    for entry in _iter_jsonl(cache_path):
        prompt = entry.get("prompt") or entry.get("input")
        response = entry.get("response") or entry.get("completion")
        if prompt is None or response is None:
            continue
        examples.append(
            DistillationExample(
                prompt=str(prompt),
                completion=str(response),
                metadata={"source": "teacher_cache"},
            )
        )
    return examples
