#!/usr/bin/env python3
# make_jamo_substitution_instruction_data.py
"""
Create jamo-substitution augmented instruction JSONL.

Default:
  - Transform instruction/input only.
  - Keep output clean.
  - Preserve original JSON fields where possible.

Key options:
  --input_jamo_ratio 0.10
      Fraction of Hangul syllables to convert in instruction/input.

  --output_jamo_ratio 0.00
      Fraction of Hangul syllables to convert in output.
      This is ignored unless --include_output is passed.

  --include_output
      Enable output-side jamo substitution.

  --include_clean_copy
      Write the original clean example before the augmented example.
      This doubles data size and is useful for clean+aug mixed SFT.

  --skip_unchanged
      Skip augmented examples where no character changed.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Tuple


CHOSUNG = [
    "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ",
    "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
]

JUNGSUNG = [
    "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ",
    "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ",
]

JONGSUNG = [
    "", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ",
    "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ",
    "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--input_jsonl", type=str, required=True)
    p.add_argument("--output_jsonl", type=str, required=True)

    p.add_argument("--input_jamo_ratio", type=float, default=0.10)
    p.add_argument("--output_jamo_ratio", type=float, default=0.0)
    p.add_argument("--include_output", action="store_true")

    p.add_argument(
        "--fields",
        type=str,
        nargs="+",
        default=["instruction", "input"],
        choices=["instruction", "input"],
        help="Input-side fields to transform. Output is controlled separately by --include_output.",
    )

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_examples", type=int, default=None)

    p.add_argument(
        "--include_clean_copy",
        action="store_true",
        help="Write the original clean example before the augmented example.",
    )
    p.add_argument(
        "--skip_unchanged",
        action="store_true",
        help="Skip augmented examples where no transformed field changed.",
    )

    p.add_argument(
        "--mark_metadata",
        action="store_true",
        default=True,
        help="Add jamo_substitution metadata fields.",
    )
    p.add_argument(
        "--no_mark_metadata",
        action="store_false",
        dest="mark_metadata",
        help="Do not add jamo_substitution metadata fields.",
    )

    p.add_argument(
        "--write_clean_fields",
        action="store_true",
        help="Store original fields as clean_instruction/clean_input/clean_output in augmented examples.",
    )

    return p.parse_args()


def is_hangul_syllable(ch: str) -> bool:
    code = ord(ch)
    return 0xAC00 <= code <= 0xD7A3


def decompose_hangul_to_compat_jamo(ch: str) -> str:
    code = ord(ch)
    if not (0xAC00 <= code <= 0xD7A3):
        return ch

    s_index = code - 0xAC00
    cho = s_index // 588
    jung = (s_index % 588) // 28
    jong = s_index % 28

    return CHOSUNG[cho] + JUNGSUNG[jung] + JONGSUNG[jong]


def substitute_text_to_jamo(text: str, ratio: float, rng: random.Random) -> Tuple[str, int, int]:
    """
    Convert each Hangul syllable to compatibility jamo with probability ratio.

    Returns:
      new_text, num_hangul_syllables, num_changed
    """
    if not text:
        return text, 0, 0

    out = []
    n_hangul = 0
    n_changed = 0

    for ch in text:
        if is_hangul_syllable(ch):
            n_hangul += 1
            if ratio > 0 and rng.random() < ratio:
                out.append(decompose_hangul_to_compat_jamo(ch))
                n_changed += 1
            else:
                out.append(ch)
        else:
            out.append(ch)

    return "".join(out), n_hangul, n_changed


def get_text(ex: Dict[str, Any], key: str) -> str:
    val = ex.get(key, "")
    if val is None:
        return ""
    return str(val)


def transform_example(
    ex: Dict[str, Any],
    rng: random.Random,
    fields,
    input_ratio: float,
    include_output: bool,
    output_ratio: float,
    mark_metadata: bool,
    write_clean_fields: bool,
) -> Tuple[Dict[str, Any], Dict[str, int], bool]:
    new_ex = dict(ex)

    stats = Counter()
    changed_any = False

    if write_clean_fields:
        if "instruction" in ex:
            new_ex["clean_instruction"] = get_text(ex, "instruction")
        if "input" in ex:
            new_ex["clean_input"] = get_text(ex, "input")
        if "output" in ex:
            new_ex["clean_output"] = get_text(ex, "output")

    for field in fields:
        old = get_text(ex, field)
        new, n_hangul, n_changed = substitute_text_to_jamo(old, input_ratio, rng)
        new_ex[field] = new
        stats[f"{field}_hangul"] += n_hangul
        stats[f"{field}_changed"] += n_changed
        if new != old:
            changed_any = True

    if include_output:
        old = get_text(ex, "output")
        new, n_hangul, n_changed = substitute_text_to_jamo(old, output_ratio, rng)
        new_ex["output"] = new
        stats["output_hangul"] += n_hangul
        stats["output_changed"] += n_changed
        if new != old:
            changed_any = True

    if mark_metadata:
        new_ex["jamo_substitution"] = True
        new_ex["jamo_input_ratio"] = input_ratio
        new_ex["jamo_output_ratio"] = output_ratio if include_output else 0.0
        new_ex["jamo_include_output"] = bool(include_output)
        new_ex["jamo_changed"] = bool(changed_any)

    return new_ex, dict(stats), changed_any


def main() -> None:
    args = parse_args()

    if not (0.0 <= args.input_jamo_ratio <= 1.0):
        raise ValueError("--input_jamo_ratio must be in [0, 1]")
    if not (0.0 <= args.output_jamo_ratio <= 1.0):
        raise ValueError("--output_jamo_ratio must be in [0, 1]")

    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    counts = Counter()
    source_counts = Counter()
    stat_counts = Counter()

    with input_path.open("r", encoding="utf-8") as f, output_path.open("w", encoding="utf-8") as out:
        for line in f:
            if args.max_examples is not None and counts["read"] >= args.max_examples:
                break
            if not line.strip():
                continue

            ex = json.loads(line)
            counts["read"] += 1
            source_counts[ex.get("source", "unknown")] += 1

            if args.include_clean_copy:
                clean_ex = dict(ex)
                if args.mark_metadata:
                    clean_ex["jamo_substitution"] = False
                    clean_ex["jamo_input_ratio"] = 0.0
                    clean_ex["jamo_output_ratio"] = 0.0
                    clean_ex["jamo_include_output"] = False
                    clean_ex["jamo_changed"] = False
                out.write(json.dumps(clean_ex, ensure_ascii=False) + "\n")
                counts["written_clean"] += 1

            new_ex, stats, changed = transform_example(
                ex=ex,
                rng=rng,
                fields=args.fields,
                input_ratio=args.input_jamo_ratio,
                include_output=args.include_output,
                output_ratio=args.output_jamo_ratio,
                mark_metadata=args.mark_metadata,
                write_clean_fields=args.write_clean_fields,
            )

            stat_counts.update(stats)

            if args.skip_unchanged and not changed:
                counts["skipped_unchanged"] += 1
                continue

            out.write(json.dumps(new_ex, ensure_ascii=False) + "\n")
            counts["written_aug"] += 1
            if changed:
                counts["changed"] += 1
            else:
                counts["unchanged"] += 1

    summary = {
        "input_jsonl": str(input_path),
        "output_jsonl": str(output_path),
        "seed": args.seed,
        "max_examples": args.max_examples,
        "fields": args.fields,
        "input_jamo_ratio": args.input_jamo_ratio,
        "include_output": args.include_output,
        "output_jamo_ratio": args.output_jamo_ratio if args.include_output else 0.0,
        "include_clean_copy": args.include_clean_copy,
        "skip_unchanged": args.skip_unchanged,
        "counts": dict(counts),
        "source_counts": dict(source_counts),
        "substitution_stats": dict(stat_counts),
    }

    stats_path = Path(str(output_path) + ".stats.json")
    stats_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n===== Done =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("output:", output_path)
    print("stats:", stats_path)


if __name__ == "__main__":
    main()
