# train_sft_baseline.py
"""
Baseline instruction SFT for Polyglot-Ko style causal LM.

Uses:
  - sft_dataset.py
  - answer-only labels: prompt tokens are masked as -100
  - no jamo auxiliary loss

Example single GPU:
  CUDA_VISIBLE_DEVICES=1 python train_sft_baseline.py \
    --train_jsonl /home/irteam/data-vol1/minsu/kolm/data/processed/instruction/instruction_train.balanced_250k.jsonl \
    --eval_jsonl /home/irteam/data-vol1/minsu/kolm/data/processed/instruction/instruction_valid.jsonl \
    --max_length 1024 \
    --max_steps 2000 \
    --bf16 \
    --output_dir ./outputs/sft_baseline_2k

Example 4 GPU:
  CUDA_VISIBLE_DEVICES=1,2,3,4 torchrun --nproc_per_node=4 train_sft_baseline.py \
    --train_jsonl /home/irteam/data-vol1/minsu/kolm/data/processed/instruction/instruction_train.balanced_250k.jsonl \
    --eval_jsonl /home/irteam/data-vol1/minsu/kolm/data/processed/instruction/instruction_valid.jsonl \
    --max_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_steps 2000 \
    --bf16 \
    --output_dir ./outputs/sft_baseline_2k
"""

from __future__ import annotations

import argparse
import inspect
import math
import os
import random
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    set_seed,
)

from sft_dataset import SFTJsonlDataset, SFTDataCollator


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--base_model", type=str, default="EleutherAI/polyglot-ko-1.3b")
    p.add_argument(
        "--train_jsonl",
        type=str,
        default="/home/irteam/data-vol1/minsu/kolm/data/processed/instruction/instruction_train.balanced_250k.jsonl",
    )
    p.add_argument(
        "--eval_jsonl",
        type=str,
        default="/home/irteam/data-vol1/minsu/kolm/data/processed/instruction/instruction_valid.jsonl",
    )
    p.add_argument("--output_dir", type=str, default="./outputs/sft_baseline")

    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--max_train_examples", type=int, default=None)
    p.add_argument("--max_eval_examples", type=int, default=None)
    p.add_argument("--overwrite_cache", action="store_true")
    p.add_argument(
        "--bpe_dropout",
        type=float,
        default=0.0,
        help="Apply BPE dropout during SFT dataset tokenization. Use 0.0 to disable.",
    )

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--learning_rate", type=float, default=5e-6)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.03)

    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)

    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--eval_steps", type=int, default=250)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--save_total_limit", type=int, default=3)

    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true", default=False)

    p.add_argument("--report_to", type=str, default="none", choices=["none", "wandb", "tensorboard"])
    p.add_argument("--run_name", type=str, default="sft_baseline")
    p.add_argument("--template_name", type=str, default="auto", choices=["auto", "simple", "qwen"])

    return p.parse_args()


def make_training_args(**kwargs):
    """
    Keep compatibility across transformers versions by dropping unsupported args.
    """
    sig = inspect.signature(TrainingArguments.__init__)
    allowed = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    dropped = sorted(set(kwargs.keys()) - set(filtered.keys()))
    if dropped:
        print("[WARN] Dropped unsupported TrainingArguments keys:", dropped)
    return TrainingArguments(**filtered)


class LossPrinterCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        keys = ["loss", "eval_loss", "learning_rate", "epoch"]
        msg = {k: logs[k] for k in keys if k in logs}
        if msg:
            print("[LOG]", msg)


def get_cache_path(
    jsonl_path: str,
    split: str,
    max_length: int,
    max_examples,
    template_name: str,
    model_name: str,
    bpe_dropout: float = 0.0,
):
    path = Path(jsonl_path)
    tag = "full" if max_examples is None else f"max{max_examples}"
    name = path.stem.replace(".", "_")
    model_tag = model_name.replace("/", "_").replace(".", "_")

    if bpe_dropout and bpe_dropout > 0:
        drop_tag = f"bpedrop{str(bpe_dropout).replace('.', 'p')}"
    else:
        drop_tag = "bpedrop0"

    return path.parent / f"{name}.{split}.sft.{model_tag}.{template_name}.{drop_tag}.max{max_length}.{tag}.pt"


def compute_metrics_from_loss(eval_pred):
    return {}


def set_bpe_dropout(tokenizer, dropout: float):
    """Enable BPE dropout for tokenization-time subword regularization."""
    if not dropout or dropout <= 0:
        print("[BPE_DROPOUT] disabled")
        return False

    backend = getattr(tokenizer, "backend_tokenizer", None)
    model = getattr(backend, "model", None)

    if model is None or not hasattr(model, "dropout"):
        raise RuntimeError(
            f"Tokenizer backend does not support BPE dropout: backend_model={type(model)}"
        )

    model.dropout = float(dropout)
    print(f"[BPE_DROPOUT] enabled tokenizer.backend_tokenizer.model.dropout={model.dropout}")
    return True


def clear_bpe_dropout(tokenizer):
    """Disable BPE dropout for deterministic eval tokenization and saved tokenizer."""
    backend = getattr(tokenizer, "backend_tokenizer", None)
    model = getattr(backend, "model", None)
    if model is not None and hasattr(model, "dropout"):
        model.dropout = None
        print("[BPE_DROPOUT] cleared tokenizer.backend_tokenizer.model.dropout=None")


def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    set_bpe_dropout(tokenizer, args.bpe_dropout)

    template_name = args.template_name
    if template_name == "auto":
        model_l = args.base_model.lower()
        if "qwen" in model_l:
            template_name = "qwen"
        else:
            template_name = "simple"

    print("Prompt template:", template_name)

    print("Loading datasets...")
    train_ds = SFTJsonlDataset(
        jsonl_path=args.train_jsonl,
        base_tokenizer=tokenizer,
        jamo_tokenizer=None,
        max_length=args.max_length,
        max_examples=args.max_train_examples,
        cache_path=get_cache_path(
            args.train_jsonl,
            "train",
            args.max_length,
            args.max_train_examples,
            template_name,
            args.base_model,
            args.bpe_dropout,
        ),
        overwrite_cache=args.overwrite_cache,
        template_name=template_name,
    )
    # Training examples were already tokenized above.
    # Use deterministic tokenization for eval cache and saved tokenizer.
    clear_bpe_dropout(tokenizer)

    eval_ds = SFTJsonlDataset(
        jsonl_path=args.eval_jsonl,
        base_tokenizer=tokenizer,
        jamo_tokenizer=None,
        max_length=args.max_length,
        max_examples=args.max_eval_examples,
        cache_path=get_cache_path(
            args.eval_jsonl,
            "eval",
            args.max_length,
            args.max_eval_examples,
            template_name,
            args.base_model,
            0.0,
        ),
        overwrite_cache=args.overwrite_cache,
        template_name=template_name,
    )

    collator = SFTDataCollator(
        pad_token_id=tokenizer.pad_token_id,
        pad_to_multiple_of=8,
    )

    print("Loading model...")
    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)

    model_kwargs: Dict[str, Any] = {}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        **model_kwargs,
    )

    model.config.use_cache = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    report_to = [] if args.report_to == "none" else [args.report_to]

    training_args = make_training_args(
        output_dir=args.output_dir,
        do_train=True,
        do_eval=True,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to=report_to,
        run_name=args.run_name,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=False,
        logging_first_step=True,
        save_safetensors=False,
        eval_strategy="steps",
        evaluation_strategy="steps",
        save_strategy="steps",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        callbacks=[LossPrinterCallback()],
    )

    print("Starting SFT baseline training...")
    train_result = trainer.train()

    print("Saving final model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    print("Final eval...")
    eval_metrics = trainer.evaluate()
    if "eval_loss" in eval_metrics:
        try:
            eval_metrics["eval_ppl"] = math.exp(eval_metrics["eval_loss"])
        except OverflowError:
            eval_metrics["eval_ppl"] = float("inf")

    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    print("Done.")
    print("Output:", args.output_dir)


if __name__ == "__main__":
    main()
