# eval_ppl.py
"""
Evaluate causal LM perplexity on a plain text validation file.

This script evaluates ONLY language-modeling loss/perplexity.
For a jamo-aux trained checkpoint, it loads the saved base_model as a normal
AutoModelForCausalLM checkpoint and ignores auxiliary heads.

Examples:

1) Evaluate original Polyglot-Ko:
    CUDA_VISIBLE_DEVICES=0 python eval_ppl.py \
      --model_name_or_path EleutherAI/polyglot-ko-1.3b \
      --eval_text path_to_dataset.txt \
      --block_size 1024 \
      --max_lines 10000

2) Evaluate jamo-aux checkpoint:
    CUDA_VISIBLE_DEVICES=0 python eval_ppl.py \
      --model_name_or_path ./outputs/jamo_aux_debug \
      --eval_text path_to_dataset.txt \
      --block_size 1024 \
      --max_lines 10000
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument(
        "--eval_text",
        type=str,
        default="path_to_dataset.txt",
    )
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--max_lines", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help="Optional tokenizer path. Defaults to model_name_or_path.",
    )

    return parser.parse_args()


def get_dtype(dtype: str):
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    return torch.float32


def iter_lines(path: str | Path, max_lines: Optional[int] = None):
    path = Path(path)
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            yield line


def build_blocks(
    text_path: str | Path,
    tokenizer,
    block_size: int,
    max_lines: Optional[int] = None,
    add_eos_between_lines: bool = True,
) -> List[List[int]]:
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("tokenizer.eos_token_id is None.")

    blocks: List[List[int]] = []
    buffer: List[int] = []

    for line in tqdm(iter_lines(text_path, max_lines=max_lines), desc="Tokenizing eval text"):
        ids = tokenizer.encode(line, add_special_tokens=False)
        if not ids:
            continue

        buffer.extend(ids)
        if add_eos_between_lines:
            buffer.append(eos_id)

        while len(buffer) >= block_size:
            blocks.append(buffer[:block_size])
            buffer = buffer[block_size:]

    if len(buffer) >= 16:
        blocks.append(buffer)

    return blocks


def collate_blocks(blocks: List[List[int]], pad_id: int):
    max_len = max(len(x) for x in blocks)

    input_ids = []
    attention_mask = []
    labels = []

    for ids in blocks:
        pad_len = max_len - len(ids)
        input_ids.append(ids + [pad_id] * pad_len)
        attention_mask.append([1] * len(ids) + [0] * pad_len)
        labels.append(ids + [-100] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


@torch.no_grad()
def evaluate_ppl(model, blocks, pad_id: int, batch_size: int, device: str):
    model.eval()

    total_nll = 0.0
    total_tokens = 0

    for start in tqdm(range(0, len(blocks), batch_size), desc="Evaluating"):
        batch_blocks = blocks[start : start + batch_size]
        batch = collate_blocks(batch_blocks, pad_id=pad_id)
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=None,
            return_dict=True,
        )

        logits = outputs.logits
        labels = batch["labels"]

        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = labels[:, 1:].contiguous()

        loss_sum = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="sum",
        )

        valid_tokens = (shift_labels != -100).sum().item()

        total_nll += float(loss_sum.detach().cpu())
        total_tokens += int(valid_tokens)

    mean_loss = total_nll / max(total_tokens, 1)
    ppl = math.exp(mean_loss) if mean_loss < 100 else float("inf")

    return {
        "loss": mean_loss,
        "ppl": ppl,
        "num_tokens": total_tokens,
        "num_blocks": len(blocks),
    }


def load_tokenizer(args):
    tok_path = args.tokenizer_name_or_path or args.model_name_or_path

    try:
        tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True )
    except Exception:
        print(f"[WARN] Failed to load tokenizer from {tok_path}. Falling back to EleutherAI/polyglot-ko-1.3b")
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/polyglot-ko-1.3b", trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def main():
    args = parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    dtype = get_dtype(args.dtype)

    print("Loading tokenizer...")
    tokenizer = load_tokenizer(args)
    print("Tokenizer vocab:", len(tokenizer))

    print("Building eval blocks...")
    blocks = build_blocks(
        text_path=args.eval_text,
        tokenizer=tokenizer,
        block_size=args.block_size,
        max_lines=args.max_lines,
        add_eos_between_lines=True,
    )

    print("Eval blocks:", len(blocks))
    if len(blocks) == 0:
        raise ValueError("No eval blocks were built. Check eval_text/max_lines.")

    print("Loading model...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
            dtype=dtype,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
            torch_dtype=dtype,
        )

    model.to(device)

    metrics = evaluate_ppl(
        model=model,
        blocks=blocks,
        pad_id=tokenizer.pad_token_id,
        batch_size=args.batch_size,
        device=device,
    )

    metrics.update(
        {
            "model_name_or_path": args.model_name_or_path,
            "eval_text": args.eval_text,
            "block_size": args.block_size,
            "max_lines": args.max_lines,
            "batch_size": args.batch_size,
            "dtype": args.dtype,
        }
    )

    print("\n***** PPL metrics *****")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print("Saved:", output_path)


if __name__ == "__main__":
    main()
