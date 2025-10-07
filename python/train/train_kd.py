from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml
from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments, set_seed)

from mathllm.distill import (DistillationDataset, apply_lora_if_available,
                              load_distillation_examples, save_distillation_examples)

LOGGER = logging.getLogger("mathllm.train.kd")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Knowledge distillation training for MathLLM student")
    parser.add_argument("--config", type=Path, required=True, help="Path to KD YAML configuration")
    parser.add_argument("--output_dir", type=Path, default=None, help="Override output directory")
    parser.add_argument("--save_processed", type=Path, default=None,
                        help="Optional path to export the processed KD dataset as JSONL")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("KD config must be a mapping")
    return raw


def prepare_dataset(config: Dict[str, Any], tokenizer) -> DistillationDataset:
    dataset_path = Path(config["dataset_path"]).expanduser().resolve()
    examples = load_distillation_examples(dataset_path)
    if config.get("subset_size"):
        examples = examples[: int(config["subset_size"])]
    instruction_template = config.get("instruction_template")
    pad_to_max = bool(config.get("pad_to_max_length", False))
    dataset = DistillationDataset(
        examples,
        tokenizer,
        max_length=int(config.get("max_seq_length", 2048)),
        pad_to_max_length=pad_to_max,
        add_eos=bool(config.get("add_eos", True)),
        instruction_template=instruction_template,
    )
    return dataset


def maybe_load_eval_dataset(config: Dict[str, Any], tokenizer) -> Optional[DistillationDataset]:
    eval_path = config.get("eval_dataset_path")
    if not eval_path:
        return None
    path = Path(eval_path).expanduser().resolve()
    if not path.exists():
        LOGGER.warning("Eval dataset %s not found; skipping", path)
        return None
    return DistillationDataset(
        load_distillation_examples(path),
        tokenizer,
        max_length=int(config.get("max_seq_length", 2048)),
        pad_to_max_length=bool(config.get("pad_to_max_length", False)),
        add_eos=bool(config.get("add_eos", True)),
        instruction_template=config.get("instruction_template"),
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    config = load_config(args.config)
    set_seed(args.seed)
    output_dir = (args.output_dir or config.get("output_dir") or Path("runs/kd")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading tokenizer %s", config["student_model"])
    tokenizer = AutoTokenizer.from_pretrained(config["student_model"], use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    LOGGER.info("Loading student model %s", config["student_model"])
    model = AutoModelForCausalLM.from_pretrained(
        config["student_model"],
        torch_dtype=_maybe_get_dtype(config.get("torch_dtype")),
        device_map=config.get("device_map", "auto"),
    )

    lora_config = config.get("lora")
    if lora_config:
        model = apply_lora_if_available(model, lora_config)

    dataset = prepare_dataset(config, tokenizer)
    eval_dataset = maybe_load_eval_dataset(config, tokenizer)

    if args.save_processed:
        save_distillation_examples(dataset.examples, args.save_processed)

    training_args = _build_training_args(config, output_dir)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    train_result = trainer.train(resume_from_checkpoint=config.get("resume_from_checkpoint"))
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics = {
        "train": train_result.metrics,
        "state": trainer.state.log_history,
    }
    (output_dir / "trainer_state.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    LOGGER.info("Training complete; artifacts saved to %s", output_dir)


def _build_training_args(config: Dict[str, Any], output_dir: Path) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(config.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(config.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 1)),
        learning_rate=float(config.get("learning_rate", 5e-5)),
        num_train_epochs=float(config.get("num_train_epochs", 1)),
        warmup_steps=int(config.get("warmup_steps", 0)),
        weight_decay=float(config.get("weight_decay", 0.0)),
        logging_steps=int(config.get("logging_steps", 50)),
        save_steps=int(config.get("save_steps", 500)),
        evaluation_strategy=config.get("evaluation_strategy", "steps" if config.get("eval_dataset_path") else "no"),
        bf16=bool(config.get("bf16", False)),
        fp16=bool(config.get("fp16", False)),
        gradient_checkpointing=bool(config.get("gradient_checkpointing", False)),
        max_steps=int(config.get("max_steps", -1)),
        save_total_limit=int(config.get("save_total_limit", 2)),
        report_to=config.get("report_to") or None,
        seed=int(config.get("seed", 42)),
    )


def _maybe_get_dtype(dtype_name: Optional[str]):
    if dtype_name is None:
        return None
    norm = dtype_name.lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat32": torch.bfloat16,
    }
    if norm not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return mapping[norm]


if __name__ == "__main__":  # pragma: no cover
    main()
