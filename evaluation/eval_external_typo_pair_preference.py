#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--pairs_jsonl", required=True)
    p.add_argument("--output_json", required=True)
    p.add_argument("--dtype", choices=["bf16","fp16","fp32"], default="bf16")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--max_examples", type=int, default=None)
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--use_fast", action="store_true")
    return p.parse_args()

def dtype(name):
    return torch.bfloat16 if name=="bf16" else torch.float16 if name=="fp16" else torch.float32

def load_pairs(path: Path, max_examples=None):
    rows = []
    for line in path.open("r", encoding="utf-8"):
        if line.strip():
            ex = json.loads(line)
            if ex.get("error_sentence") and ex.get("correct_sentence"):
                rows.append(ex)
            if max_examples and len(rows) >= max_examples:
                break
    return rows

@torch.no_grad()
def score(model, tok, text, device, max_length):
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_length, add_special_tokens=True)
    ids = enc["input_ids"].to(device)
    mask = enc.get("attention_mask")
    mask = mask.to(device) if mask is not None else None
    if ids.shape[1] < 2:
        return {"avg_logprob": float("-inf"), "ppl": float("inf"), "n_tokens": 0}
    out = model(input_ids=ids, attention_mask=mask, use_cache=False)
    logits = out.logits[:, :-1, :].float()
    labels = ids[:, 1:]
    lp = F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    if mask is not None:
        m = mask[:, 1:].float()
        lp = lp * m
        n = int(m.sum().item())
    else:
        n = labels.numel()
    avg = float(lp.sum().cpu()) / max(n, 1)
    return {"avg_logprob": avg, "ppl": math.exp(-avg) if avg > -100 else float("inf"), "n_tokens": n}

def aggregate(rows):
    if not rows: return {}
    n = len(rows)
    return {
        "n": n,
        "pair_preference_acc": sum(r["correct_preferred"] for r in rows) / n,
        "mean_delta_avg_logprob": sum(r["delta_avg_logprob"] for r in rows) / n,
        "mean_ppl_error": sum(r["error_ppl"] for r in rows if math.isfinite(r["error_ppl"])) / max(sum(math.isfinite(r["error_ppl"]) for r in rows), 1),
        "mean_ppl_correct": sum(r["correct_ppl"] for r in rows if math.isfinite(r["correct_ppl"])) / max(sum(math.isfinite(r["correct_ppl"]) for r in rows), 1),
    }

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kwargs = {} if args.dtype == "fp32" else {"torch_dtype": dtype(args.dtype)}
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code, **kwargs).to(device).eval()

    pairs = load_pairs(Path(args.pairs_jsonl), args.max_examples)
    print("Loaded pairs:", len(pairs))
    results, by_dataset, by_error_type = [], defaultdict(list), defaultdict(list)
    for ex in tqdm(pairs, desc="external typo pair eval"):
        se = score(model, tok, ex["error_sentence"], device, args.max_length)
        sc = score(model, tok, ex["correct_sentence"], device, args.max_length)
        delta = sc["avg_logprob"] - se["avg_logprob"]
        row = {
            "id": ex.get("id"), "dataset": ex.get("dataset","unknown"),
            "error_type": str(ex.get("error_type","unknown")),
            "delta_avg_logprob": delta, "correct_preferred": int(delta > 0),
            "error_avg_logprob": se["avg_logprob"], "correct_avg_logprob": sc["avg_logprob"],
            "error_ppl": se["ppl"], "correct_ppl": sc["ppl"],
            "error_tokens": se["n_tokens"], "correct_tokens": sc["n_tokens"],
            "error_sentence": ex["error_sentence"], "correct_sentence": ex["correct_sentence"],
        }
        results.append(row); by_dataset[row["dataset"]].append(row); by_error_type[row["error_type"]].append(row)
    out = {
        "model_name_or_path": args.model_name_or_path,
        "pairs_jsonl": args.pairs_jsonl,
        "metrics": aggregate(results),
        "metrics_by_dataset": {k: aggregate(v) for k,v in sorted(by_dataset.items())},
        "metrics_by_error_type": {k: aggregate(v) for k,v in sorted(by_error_type.items())},
        "examples": results[:50],
    }
    op = Path(args.output_json); op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out["metrics"], ensure_ascii=False, indent=2))
    print("saved:", op)

if __name__ == "__main__":
    main()
