"""
Cleaned SFT trainer for Robust Korean Language Modeling (+Jamo Extension).
Optimizes standard causal language modeling loss on noisy instruction data.
"""

from __future__ import annotations

import argparse
import inspect
import math
import os
from pathlib import Path
from typing import Any, Dict

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

    p.add_argument("--base_model", type=str, required=True, help="Path to the CPT model directory")
    p.add_argument("--jamo_tokenizer_path", type=str, default="", help="Path to Jamo tokenizer (if needed by dataset)")
    p.add_argument("--train_jsonl", type=str, required=True)
    p.add_argument("--eval_jsonl", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./outputs/sft_final")

    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--max_train_examples", type=int, default=None)
    p.add_argument("--max_eval_examples", type=int, default=None)
    p.add_argument("--overwrite_cache", action="store_true")

    p.add_argument("--template_name", type=str, default="auto")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_steps", type=int, default=1000)
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
    p.add_argument("--save_total_limit", type=int, default=2)

    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true", default=False)

    p.add_argument("--report_to", type=str, default="none", choices=["none", "wandb", "tensorboard"])
    p.add_argument("--run_name", type=str, default="sft_final")

    return p.parse_args()


def infer_template_name(template_name: str, base_model: str) -> str:
    if template_name != "auto":
        return template_name
    return "qwen" if "qwen" in base_model.lower() else "simple"


def make_training_args(**kwargs):
    sig = inspect.signature(TrainingArguments.__init__)
    allowed = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    return TrainingArguments(**filtered)


def sanitize_model_tag(model_name: str) -> str:
    return model_name.replace("/", "_").replace(".", "_").replace(":", "_")


def get_cache_path(jsonl_path: str, split: str, max_length: int, max_examples, template_name: str, model_name: str):
    path = Path(jsonl_path)
    tag = "full" if max_examples is None else f"max{max_examples}"
    name = path.stem.replace(".", "_")
    model_tag = sanitize_model_tag(model_name)
    return path.parent / f"{name}.{split}.sft.standard.{model_tag}.{template_name}.max{max_length}.{tag}.pt"


class LossPrinterCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or not state.is_world_process_zero:
            return
        msg = {k: logs[k] for k in ["loss", "eval_loss", "learning_rate", "epoch", "grad_norm"] if k in logs}
        if msg:
            print("[LOG]", msg, flush=True)


def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    template_name = infer_template_name(args.template_name, args.base_model)
    print("Prompt template:", template_name)

    print("Loading tokenizers...")
    base_tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    
    jamo_tokenizer = None
    if args.jamo_tokenizer_path:
        jamo_tokenizer = AutoTokenizer.from_pretrained(args.jamo_tokenizer_path, trust_remote_code=True)

    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token

    print("Loading datasets...")
    train_ds = SFTJsonlDataset(
        jsonl_path=args.train_jsonl,
        base_tokenizer=base_tokenizer,
        jamo_tokenizer=jamo_tokenizer,
        max_length=args.max_length,
        max_examples=args.max_train_examples,
        cache_path=get_cache_path(args.train_jsonl, "train", args.max_length, args.max_train_examples, template_name, args.base_model),
        overwrite_cache=args.overwrite_cache,
        template_name=template_name,
    )

    eval_ds = SFTJsonlDataset(
        jsonl_path=args.eval_jsonl,
        base_tokenizer=base_tokenizer,
        jamo_tokenizer=jamo_tokenizer,
        max_length=args.max_length,
        max_examples=args.max_eval_examples,
        cache_path=get_cache_path(args.eval_jsonl, "eval", args.max_length, args.max_eval_examples, template_name, args.base_model),
        overwrite_cache=args.overwrite_cache,
        template_name=template_name,
    )

    collator = SFTDataCollator(pad_token_id=base_tokenizer.pad_token_id, pad_to_multiple_of=8)

    print("Loading base model...")
    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)
    model_kwargs = {"torch_dtype": dtype} if dtype else {}

    model = AutoModelForCausalLM.from_pretrained(args.base_model, trust_remote_code=True, **model_kwargs)
    model.config.use_cache = False

    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
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
        disable_tqdm=True,
        logging_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to=report_to,
        run_name=args.run_name,
        remove_unused_columns=True, 
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=False,
        logging_first_step=True,
        save_safetensors=False,
        eval_strategy="steps",
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

    print("Starting Standard SFT training...")
    train_result = trainer.train()

    print("Saving final model...")
    trainer.save_model(args.output_dir)
    base_tokenizer.save_pretrained(args.output_dir)

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

    print("Done. Model saved to:", args.output_dir)


if __name__ == "__main__":
    main()