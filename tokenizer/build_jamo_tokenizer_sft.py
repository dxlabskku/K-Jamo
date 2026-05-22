# build_jamo_tokenizer_sft.py
"""
Build a Jamo-augmented tokenizer for SFT auxiliary labels.

Creates a tokenizer containing:
  1. Jamo label tokens generated from the base tokenizer vocab
  2. Additional Jamo label tokens generated from SFT answer/output tokens

Usage:
  python build_jamo_tokenizer_sft.py \
    --base_model EleutherAI/polyglot-ko-1.3b \
    --sft_jsonl your_path/instruction_train.balanced_250k.jsonl \
    --output_dir ./polyglot-ko-1.3b-jamo-tokenizer-sft

Then remove old jamo SFT cache:
  rm your_path/*sft.jamo*.pt

Then train with:
  --jamo_tokenizer_path ./polyglot-ko-1.3b-jamo-tokenizer-sft
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from tqdm import tqdm
from transformers import AutoTokenizer


CHO = [
    "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ",
    "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"
]

JUNG = [
    "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ",
    "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ"
]

JONG = [
    "_", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ",
    "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ",
    "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", type=str, default="EleutherAI/polyglot-ko-1.3b")
    p.add_argument(
        "--sft_jsonl",
        type=str,
        default="your-path/instruction_train.balanced_250k.jsonl",
    )
    p.add_argument("--output_dir", type=str, default="./polyglot-ko-1.3b-jamo-tokenizer-sft")
    p.add_argument("--max_jamo_len", type=int, default=16)
    p.add_argument("--max_examples", type=int, default=None)
    p.add_argument("--include_prompt", action="store_true")
    return p.parse_args()


def normalize_text(x: Any) -> str:
    if x is None:
        return ""
    x = str(x)
    x = x.replace("\x00", "").replace("\u200b", "").replace("\ufeff", "")
    return re.sub(r"\s+", " ", x).strip()


def is_hangul_syllable(ch: str) -> bool:
    return 0xAC00 <= ord(ch) <= 0xD7A3


def decompose_syllable(ch: str) -> Tuple[str, str, str]:
    code = ord(ch) - 0xAC00
    cho_idx = code // (21 * 28)
    jung_idx = (code % (21 * 28)) // 28
    jong_idx = code % 28
    return CHO[cho_idx], JUNG[jung_idx], JONG[jong_idx]


def text_to_jamo_patterns(text: str, max_jamo_len: int = 16) -> List[str]:
    chos: List[str] = []
    jungs: List[str] = []
    jongs: List[str] = []

    for ch in text:
        if is_hangul_syllable(ch):
            c, v, f = decompose_syllable(ch)
            chos.append(c)
            jungs.append(v)
            jongs.append(f)

    if not chos:
        return []

    if len(chos) > max_jamo_len:
        return []

    return [
        "<CHO:" + "".join(chos) + ">",
        "<JUNG:" + "".join(jungs) + ">",
        "<JONG:" + "".join(jongs) + ">",
    ]


def collect_from_base_vocab(tokenizer, max_jamo_len: int) -> Set[str]:
    tokens: Set[str] = set()
    base_vocab_size = tokenizer.vocab_size

    print("Scanning base vocab:", base_vocab_size)

    for token_id in tqdm(range(base_vocab_size), desc="Base vocab jamo"):
        decoded = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        for p in text_to_jamo_patterns(decoded, max_jamo_len=max_jamo_len):
            tokens.add(p)

    return tokens


def read_sft_texts(jsonl_path: Path, include_prompt: bool, max_examples: Optional[int]) -> Iterable[str]:
    with jsonl_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_examples is not None and idx >= max_examples:
                break

            line = line.strip()
            if not line:
                continue

            ex = json.loads(line)
            output = normalize_text(ex.get("output", ""))

            if include_prompt:
                instruction = normalize_text(ex.get("instruction", ""))
                input_text = normalize_text(ex.get("input", ""))
                text = "\n".join([instruction, input_text, output])
            else:
                text = output

            if text:
                yield text


def collect_from_sft_outputs(
    tokenizer,
    jsonl_path: Path,
    max_jamo_len: int,
    include_prompt: bool,
    max_examples: Optional[int],
) -> Set[str]:
    tokens: Set[str] = set()

    print("Scanning SFT JSONL:", jsonl_path)
    print("include_prompt:", include_prompt)

    for text in tqdm(
        read_sft_texts(jsonl_path, include_prompt=include_prompt, max_examples=max_examples),
        desc="SFT answer jamo",
    ):
        input_ids = tokenizer.encode(text, add_special_tokens=False)

        for tid in input_ids:
            decoded = tokenizer.decode([tid], clean_up_tokenization_spaces=False)
            for p in text_to_jamo_patterns(decoded, max_jamo_len=max_jamo_len):
                tokens.add(p)

    return tokens


def compute_coverage(
    tokenizer,
    jamo_tokens: Set[str],
    jsonl_path: Path,
    max_jamo_len: int,
    max_examples: Optional[int],
) -> Dict[str, float]:
    total_hangul_tokens = 0
    cho_hit = 0
    jung_hit = 0
    jong_hit = 0

    for text in tqdm(
        read_sft_texts(jsonl_path, include_prompt=False, max_examples=max_examples),
        desc="Coverage check",
    ):
        input_ids = tokenizer.encode(text, add_special_tokens=False)

        for tid in input_ids:
            decoded = tokenizer.decode([tid], clean_up_tokenization_spaces=False)
            patterns = text_to_jamo_patterns(decoded, max_jamo_len=max_jamo_len)

            if not patterns:
                continue

            total_hangul_tokens += 1
            cho, jung, jong = patterns

            cho_hit += int(cho in jamo_tokens)
            jung_hit += int(jung in jamo_tokens)
            jong_hit += int(jong in jamo_tokens)

    if total_hangul_tokens == 0:
        return {
            "total_hangul_tokens": 0,
            "cho_coverage": 0.0,
            "jung_coverage": 0.0,
            "jong_coverage": 0.0,
        }

    return {
        "total_hangul_tokens": total_hangul_tokens,
        "cho_coverage": cho_hit / total_hangul_tokens,
        "jung_coverage": jung_hit / total_hangul_tokens,
        "jong_coverage": jong_hit / total_hangul_tokens,
    }


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sft_jsonl = Path(args.sft_jsonl)
    if not sft_jsonl.exists():
        raise FileNotFoundError(f"SFT JSONL not found: {sft_jsonl}")

    print("Loading base tokenizer:", args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    print("BASE tokenizer.vocab_size:", tokenizer.vocab_size)
    print("BASE len(tokenizer):", len(tokenizer))

    base_tokens = collect_from_base_vocab(tokenizer, max_jamo_len=args.max_jamo_len)
    print("Base-vocab jamo tokens:", len(base_tokens))

    sft_tokens = collect_from_sft_outputs(
        tokenizer=tokenizer,
        jsonl_path=sft_jsonl,
        max_jamo_len=args.max_jamo_len,
        include_prompt=args.include_prompt,
        max_examples=args.max_examples,
    )
    print("SFT jamo tokens:", len(sft_tokens))

    all_jamo_tokens = sorted(base_tokens | sft_tokens)

    print("Total jamo tokens:", len(all_jamo_tokens))
    print("New from SFT only:", len(sft_tokens - base_tokens))
    print("Examples:", all_jamo_tokens[:30])

    with (output_dir / "jamo_tokens.json").open("w", encoding="utf-8") as f:
        json.dump(all_jamo_tokens, f, ensure_ascii=False, indent=2)

    with (output_dir / "jamo_tokens_base_only.json").open("w", encoding="utf-8") as f:
        json.dump(sorted(base_tokens), f, ensure_ascii=False, indent=2)

    with (output_dir / "jamo_tokens_sft_only.json").open("w", encoding="utf-8") as f:
        json.dump(sorted(sft_tokens - base_tokens), f, ensure_ascii=False, indent=2)

    added = tokenizer.add_special_tokens(
        {"additional_special_tokens": all_jamo_tokens}
    )

    print("Actually added:", added)
    print("NEW len(tokenizer):", len(tokenizer))

    tokenizer.save_pretrained(output_dir)

    coverage = compute_coverage(
        tokenizer=tokenizer,
        jamo_tokens=set(all_jamo_tokens),
        jsonl_path=sft_jsonl,
        max_jamo_len=args.max_jamo_len,
        max_examples=args.max_examples,
    )

    stats = {
        "base_model": args.base_model,
        "sft_jsonl": str(sft_jsonl),
        "output_dir": str(output_dir),
        "max_jamo_len": args.max_jamo_len,
        "max_examples": args.max_examples,
        "include_prompt": args.include_prompt,
        "base_vocab_jamo_tokens": len(base_tokens),
        "sft_jamo_tokens": len(sft_tokens),
        "sft_only_jamo_tokens": len(sft_tokens - base_tokens),
        "total_jamo_tokens": len(all_jamo_tokens),
        "actually_added": added,
        "new_tokenizer_len": len(tokenizer),
        "coverage": coverage,
    }

    with (output_dir / "build_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\n===== Build stats =====")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print("\nSaved to:", output_dir)


if __name__ == "__main__":
    main()
