# train.py
"""
Train Polyglot-Ko 1.3B with Korean Jamo Auxiliary Loss.

Expected files in same project:
    - token_model.py      # PolyglotKoWithJamoAux
    - dataset.py          # JamoPackedTextDataset
    - collator.py         # JamoCausalLMCollator
    - utils.py            # jamo utilities

Example sanity run:
    accelerate launch train.py \
      --max_train_lines 100000 \
      --max_eval_lines 10000 \
      --block_size 1024 \
      --per_device_train_batch_size 1 \
      --gradient_accumulation_steps 16 \
      --max_steps 500 \
      --output_dir ./outputs/jamo_aux_debug

Example fuller run:
    accelerate launch train.py \
      --block_size 2048 \
      --per_device_train_batch_size 1 \
      --gradient_accumulation_steps 32 \
      --max_steps 10000 \
      --output_dir ./outputs/jamo_aux_aihub_10k
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Any

import torch
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    set_seed,
)

from token_model import PolyglotKoWithJamoAux
from dataset import JamoPackedTextDataset
from collator import JamoCausalLMCollator
import hashlib

def get_cache_name(text_path: str, split: str, block_size: int, max_tag: str):
    path = Path(text_path)
    stem = path.stem.replace(".", "_")
    digest = hashlib.md5(str(path).encode("utf-8")).hexdigest()[:8]
    return f"{stem}_{split}.block{block_size}.{max_tag}.{digest}.pt"

class LossLoggingCallback(TrainerCallback):
    """
    Logs auxiliary losses returned by the model.
    HF Trainer only logs 'loss' by default, so we manually print recent values.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        if not state.is_world_process_zero:
            return

        msg = {
            k: logs[k]
            for k in ["loss", "eval_loss", "learning_rate", "epoch", "grad_norm"]
            if k in logs
        }
        if msg:
            print("[LOG]", msg, flush=True)


class JamoTrainer(Trainer):
    """
    Custom Trainer to:
      1. accept dict output from PolyglotKoWithJamoAux
      2. log lm_loss / cho_loss / jung_loss / jong_loss
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs["loss"]

        # Store detached losses for logging.
        for k in ["lm_loss", "cho_loss", "jung_loss", "jong_loss"]:
            v = outputs.get(k)
            if v is not None:
                self._last_aux_losses = getattr(self, "_last_aux_losses", {})
                self._last_aux_losses[k] = float(v.detach().float().cpu())

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: Dict[str, float], start_time=None) -> None:
        aux = getattr(self, "_last_aux_losses", None)
        if aux:
            logs.update(aux)
        try:
            super().log(logs, start_time=start_time)
        except TypeError:
            super().log(logs)


def parse_args():
    parser = argparse.ArgumentParser()

    # Paths
    parser.add_argument("--base_model", type=str, default="EleutherAI/polyglot-ko-1.3b")
    parser.add_argument("--jamo_tokenizer_path", type=str, default="./polyglot-ko-1.3b-jamo-tokenizer")

    parser.add_argument(
        "--train_text",
        type=str,
        default="path_to_data.txt",
    )
    parser.add_argument(
        "--eval_text",
        type=str,
        default="path_to_data.txt",
    )
    parser.add_argument("--cache_dir", type=str, default="path_to_cache")
    parser.add_argument("--output_dir", type=str, default="./outputs/polyglot-ko-jamo-aux")

    # Dataset
    parser.add_argument("--block_size", type=int, default=2048)
    parser.add_argument("--max_train_lines", type=int, default=None)
    parser.add_argument("--max_eval_lines", type=int, default=None)
    parser.add_argument("--overwrite_cache", action="store_true")
    parser.add_argument("--min_tokens_per_block", type=int, default=16)

    # Training
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)

    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)

    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=3)

    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)

    # Auxiliary loss weights
    parser.add_argument("--cho_loss_weight", type=float, default=0.1)
    parser.add_argument("--jung_loss_weight", type=float, default=0.1)
    parser.add_argument("--jong_loss_weight", type=float, default=0.1)

    # Logging
    parser.add_argument("--report_to", type=str, default="none", choices=["none", "wandb", "tensorboard"])
    parser.add_argument("--run_name", type=str, default="polyglot-ko-jamo-aux")

    parser.add_argument("--add_jamo_tokens", action="store_true")
    parser.add_argument("--save_extended_tokenizer", action="store_true")

    return parser.parse_args()


def maybe_set_pad_token(tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def get_compat_jamo_tokens():
    return [
        # Compatibility choseong/consonants
        "ㄱ","ㄲ","ㄳ","ㄴ","ㄵ","ㄶ","ㄷ","ㄸ","ㄹ","ㄺ",
        "ㄻ","ㄼ","ㄽ","ㄾ","ㄿ","ㅀ","ㅁ","ㅂ","ㅃ","ㅄ",
        "ㅅ","ㅆ","ㅇ","ㅈ","ㅉ","ㅊ","ㅋ","ㅌ","ㅍ","ㅎ",

        # Compatibility jungseong/vowels
        "ㅏ","ㅐ","ㅑ","ㅒ","ㅓ","ㅔ","ㅕ","ㅖ","ㅗ","ㅘ",
        "ㅙ","ㅚ","ㅛ","ㅜ","ㅝ","ㅞ","ㅟ","ㅠ","ㅡ","ㅢ","ㅣ",
    ]

def add_jamo_tokens_to_tokenizer(tokenizer):
    tokens = get_compat_jamo_tokens()
    added = tokenizer.add_tokens(tokens, special_tokens=False)
    print(f"Added jamo tokens: {added}")
    print(f"New BASE vocab: {len(tokenizer)}")
    return tokenizer, added

def build_dataset(
    split: str,
    text_path: str,
    base_tokenizer,
    jamo_tokenizer,
    args,
):
    

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    max_lines = args.max_train_lines if split == "train" else args.max_eval_lines

    max_tag = "full" if max_lines is None else f"max{max_lines}"
    cache_filename = get_cache_name(
            text_path=text_path,
            split=split,
            block_size=args.block_size,
            max_tag=max_tag,
        )
    cache_path = cache_dir / cache_filename

    dataset = JamoPackedTextDataset(
        text_path=text_path,
        base_tokenizer=base_tokenizer,
        jamo_tokenizer=jamo_tokenizer,
        block_size=args.block_size,
        add_eos_between_lines=True,
        min_tokens_per_block=args.min_tokens_per_block,
        max_lines=max_lines,
        cache_path=cache_path,
        overwrite_cache=args.overwrite_cache,
    )

    print(f"[{split}] text_path={text_path}")
    print(f"[{split}] cache_path={cache_path}")
    print(f"[{split}] num_examples={len(dataset)}")

    return dataset


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "train_args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    print("Loading tokenizers...")
    base_tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base_tokenizer = maybe_set_pad_token(base_tokenizer)
    num_added_tokens = 0
    if args.add_jamo_tokens:
        base_tokenizer, num_added_tokens = add_jamo_tokens_to_tokenizer(base_tokenizer)
    jamo_tokenizer = AutoTokenizer.from_pretrained(
        args.jamo_tokenizer_path,
        local_files_only=True,
    )

    print("BASE vocab:", len(base_tokenizer))
    print("JAMO vocab:", len(jamo_tokenizer))

    print("Building datasets...")
    train_dataset = build_dataset(
        "train",
        args.train_text,
        base_tokenizer,
        jamo_tokenizer,
        args,
    )
    eval_dataset = build_dataset(
        "eval",
        args.eval_text,
        base_tokenizer,
        jamo_tokenizer,
        args,
    )

    collator = JamoCausalLMCollator(
        tokenizer=base_tokenizer,
        pad_to_multiple_of=8,
    )

    if args.bf16:
        model_dtype = torch.bfloat16
    elif args.fp16:
        model_dtype = torch.float16
    else:
        model_dtype = torch.float32

    print("Loading model...")
    model = PolyglotKoWithJamoAux(
        model_name_or_path=args.base_model,
        jamo_vocab_size=len(jamo_tokenizer),
        cho_loss_weight=args.cho_loss_weight,
        jung_loss_weight=args.jung_loss_weight,
        jong_loss_weight=args.jong_loss_weight,
        torch_dtype=model_dtype,
    )
    if args.add_jamo_tokens and num_added_tokens > 0:
        print("Resizing base model token embeddings...")
        model.base_model.resize_token_embeddings(len(base_tokenizer))
        model.base_model.config.vocab_size = len(base_tokenizer)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        # Important for gradient checkpointing on causal LM.
        if hasattr(model.base_model.config, "use_cache"):
            model.base_model.config.use_cache = False

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        disable_tqdm=True,
        logging_strategy="steps",
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_safetensors=False,
        fp16=args.fp16 and not args.bf16,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to=[] if args.report_to == "none" else [args.report_to],
        run_name=args.run_name,
        remove_unused_columns=False,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        prediction_loss_only=True,
        load_best_model_at_end=False,
        logging_first_step=True,
    )

    trainer = JamoTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=[LossLoggingCallback()],
    )

    print("Starting training...")
    train_result = trainer.train()

    print("Saving final model...")

    trainer.save_model(str(output_dir / "wrapper"))

    model_to_save = trainer.model
    if hasattr(model_to_save, "module"):
        model_to_save = model_to_save.module

    model_to_save.base_model.save_pretrained(args.output_dir)
    base_tokenizer.save_pretrained(args.output_dir)

    jamo_tokenizer.save_pretrained(str(output_dir / "jamo_tokenizer"))

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    metrics["eval_samples"] = len(eval_dataset)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    print("Running final evaluation...")
    eval_metrics = trainer.evaluate()
    try:
        eval_metrics["eval_ppl"] = math.exp(eval_metrics["eval_loss"])
    except Exception:
        pass

    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    print("Done.")
    print("Output:", args.output_dir)


if __name__ == "__main__":
    main()
