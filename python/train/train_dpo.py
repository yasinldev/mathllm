from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from transformers import (AutoModelForCausalLM, AutoTokenizer, TrainingArguments,
                          set_seed)

from mathllm.distill import apply_lora_if_available
from mathllm.preference import (load_preferences_jsonl, preference_examples_to_hf_dataset,
                                preference_examples_to_pairs)

LOGGER = logging.getLogger("mathllm.train.dpo")

try:
    from trl import DPOTrainer
except ImportError as exc:
    raise RuntimeError("trl library is required for DPO training") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct Preference Optimization training for MathLLM")
    parser.add_argument("--config", type=Path, required=True, help="Path to DPO YAML configuration")
    parser.add_argument("--output_dir", type=Path, default=None, help="Override output directory")
    parser.add_argument("--save_filtered", type=Path, default=None,
                        help="Optional JSONL export of filtered DPO pairs")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("DPO config must be a mapping")
    return raw


def prepare_dataset(config: Dict[str, Any]):
    dataset_path = Path(config["dataset_path"]).expanduser().resolve()
    examples = load_preferences_jsonl(dataset_path)
    if config.get("subset_size"):
        examples = examples[: int(config["subset_size"])]
    if config.get("filter_max_prompt_length"):
        max_len = int(config["filter_max_prompt_length"])
        examples = [ex for ex in examples if len(ex.prompt) <= max_len]
    dataset = preference_examples_to_hf_dataset(examples)
    pairs = preference_examples_to_pairs(examples)
    return dataset, pairs


def maybe_save_pairs(pairs, path: Optional[Path]) -> None:
    if path is None:
        return
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    config = load_config(args.config)
    set_seed(args.seed)

    output_dir = (args.output_dir or config.get("output_dir") or Path("runs/dpo")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Preparing DPO dataset from %s", config["dataset_path"])
    train_dataset, pairs = prepare_dataset(config)
    maybe_save_pairs(pairs, args.save_filtered)

    eval_dataset = None
    eval_path = config.get("eval_dataset_path")
    if eval_path:
        eval_dataset, _ = prepare_dataset({**config, "dataset_path": eval_path})

    LOGGER.info("Loading tokenizer %s", config["student_model"])
    tokenizer = AutoTokenizer.from_pretrained(config["student_model"], use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    LOGGER.info("Loading student (policy) model %s", config["student_model"])
    policy_model = AutoModelForCausalLM.from_pretrained(
        config["student_model"],
        device_map=config.get("device_map", "auto"),
        torch_dtype=_maybe_get_dtype(config.get("torch_dtype")),
    )

    lora_config = config.get("lora")
    if lora_config:
        policy_model = apply_lora_if_available(policy_model, lora_config)

    ref_model_path = config.get("reference_model") or config["student_model"]
    LOGGER.info("Loading reference model %s", ref_model_path)
    reference_model = AutoModelForCausalLM.from_pretrained(
        ref_model_path,
        device_map=config.get("reference_device_map") or config.get("device_map", "auto"),
        torch_dtype=_maybe_get_dtype(config.get("reference_torch_dtype") or config.get("torch_dtype")),
    )

    beta = float(config.get("beta", 0.1))
    training_args = _build_training_args(config, output_dir)

    trainer = DPOTrainer(
        model=policy_model,
        ref_model=reference_model,
        beta=beta,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    train_result = trainer.train(resume_from_checkpoint=config.get("resume_from_checkpoint"))
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics = {
        "train": train_result.metrics,
        "state": trainer.state.log_history,
    }
    (output_dir / "trainer_state.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    LOGGER.info("DPO training complete; artifacts saved to %s", output_dir)


def _build_training_args(config: Dict[str, Any], output_dir: Path) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(config.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(config.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 1)),
        learning_rate=float(config.get("learning_rate", 1e-5)),
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
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("torch is required for DPO training") from exc
    norm = dtype_name.lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if norm not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return mapping[norm]


if __name__ == "__main__":  # pragma: no cover
    main()
