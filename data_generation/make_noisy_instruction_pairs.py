# make_noisy_instruction_pairs.py
"""
Create clean-noisy instruction pairs for Korean orthographic consistency training.

Input JSONL format is compatible with your SFT data:
{
  "instruction": "...",
  "input": "...",
  "output": "...",
  "source": "..."
}

Output JSONL format:
{
  "instruction": clean instruction,
  "input": clean input,
  "output": same output,
  "source": original source,
  "pair_id": "...",
  "noise_type": "spacing|vowel|consonant|jong|mixed",
  "clean_instruction": "...",
  "clean_input": "...",
  "noisy_instruction": "...",
  "noisy_input": "...",
  "text_changed": true
}

Notes:
  - Output/answer is NOT corrupted.
  - Only instruction/input are corrupted.
  - The default written "instruction" and "input" fields remain CLEAN so this file can
    still be inspected easily.
  - For pair training, use clean_instruction/noisy_instruction fields.
  - For quick noisy SFT/eval, you can set --write_noisy_as_main to replace
    instruction/input with noisy versions.

Example:
  python make_noisy_instruction_pairs.py \
    --input_jsonl path_to/instruction_train.balanced_250k.jsonl \
    --output_jsonl path_to/instruction_train.noisy_pairs.jsonl \
    --max_examples 100000 \
    --noise_types spacing vowel consonant jong mixed \
    --noise_prob 0.12 \
    --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm


CHO_LIST = [
    "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ",
    "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"
]

JUNG_LIST = [
    "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ",
    "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ"
]

JONG_LIST = [
    "", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ",
    "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ",
    "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"
]


VOWEL_CONFUSIONS = {
    "ㅐ": ["ㅔ"],
    "ㅔ": ["ㅐ"],
    "ㅒ": ["ㅖ"],
    "ㅖ": ["ㅒ"],
    "ㅗ": ["ㅜ"],
    "ㅜ": ["ㅗ"],
    "ㅛ": ["ㅠ"],
    "ㅠ": ["ㅛ"],
    "ㅓ": ["ㅗ"],
    "ㅏ": ["ㅓ"],
    "ㅚ": ["ㅙ", "ㅞ"],
    "ㅙ": ["ㅚ", "ㅞ"],
    "ㅞ": ["ㅙ", "ㅚ"],
    "ㅢ": ["ㅣ", "ㅡ"],
}

CONSONANT_CONFUSIONS = {
    "ㄱ": ["ㅋ", "ㄲ"],
    "ㅋ": ["ㄱ"],
    "ㄲ": ["ㄱ"],
    "ㄷ": ["ㅌ", "ㄸ"],
    "ㅌ": ["ㄷ"],
    "ㄸ": ["ㄷ"],
    "ㅂ": ["ㅍ", "ㅃ"],
    "ㅍ": ["ㅂ"],
    "ㅃ": ["ㅂ"],
    "ㅈ": ["ㅊ", "ㅉ"],
    "ㅊ": ["ㅈ"],
    "ㅉ": ["ㅈ"],
    "ㅅ": ["ㅆ"],
    "ㅆ": ["ㅅ"],
    "ㄴ": ["ㄹ"],
    "ㄹ": ["ㄴ"],
}


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--input_jsonl",
        type=str,
        default="your_path/instruction_train.balanced_250k.jsonl",
    )
    p.add_argument(
        "--output_jsonl",
        type=str,
        default="your_path/instruction_train.noisy_pairs.jsonl",
    )

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_examples", type=int, default=None)
    p.add_argument("--min_chars", type=int, default=4)
    p.add_argument("--max_chars", type=int, default=4096)

    p.add_argument(
        "--noise_types",
        nargs="+",
        default=["spacing", "vowel", "consonant", "jong", "mixed"],
        choices=["spacing", "vowel", "consonant", "jong", "mixed"],
    )
    p.add_argument(
        "--noise_prob",
        type=float,
        default=0.12,
        help="Per-character corruption probability for Hangul noise.",
    )
    p.add_argument(
        "--spacing_drop_prob",
        type=float,
        default=0.18,
        help="Probability of deleting an existing whitespace.",
    )
    p.add_argument(
        "--spacing_insert_prob",
        type=float,
        default=0.015,
        help="Probability of inserting a whitespace after a Hangul char.",
    )
    p.add_argument(
        "--noise_fields",
        nargs="+",
        default=["instruction", "input"],
        choices=["instruction", "input"],
    )
    p.add_argument(
        "--write_noisy_as_main",
        action="store_true",
        help="If set, write noisy instruction/input to the main instruction/input fields.",
    )
    p.add_argument(
        "--one_pair_per_noise_type",
        action="store_true",
        help="If set, write one pair per noise type for each example. Otherwise sample one noise type per example.",
    )
    p.add_argument(
        "--skip_unchanged",
        action="store_true",
        default=True,
        help="Skip examples where noise did not change text.",
    )
    p.add_argument(
        "--keep_unchanged",
        action="store_true",
        help="Override --skip_unchanged and keep unchanged pairs.",
    )

    return p.parse_args()


def normalize_text(x: Any) -> str:
    if x is None:
        return ""
    x = str(x)
    x = x.replace("\x00", "").replace("\u200b", "").replace("\ufeff", "")
    x = re.sub(r"[ \t]+", " ", x)
    x = re.sub(r"\n{3,}", "\n\n", x)
    return x.strip()


def is_hangul_syllable(ch: str) -> bool:
    return 0xAC00 <= ord(ch) <= 0xD7A3


def decompose(ch: str) -> Optional[Tuple[int, int, int]]:
    if not is_hangul_syllable(ch):
        return None
    code = ord(ch) - 0xAC00
    cho = code // (21 * 28)
    jung = (code % (21 * 28)) // 28
    jong = code % 28
    return cho, jung, jong


def compose(cho: int, jung: int, jong: int) -> str:
    return chr(0xAC00 + (cho * 21 + jung) * 28 + jong)


def corrupt_spacing(text: str, rng: random.Random, drop_prob: float, insert_prob: float) -> str:
    out: List[str] = []
    for i, ch in enumerate(text):
        if ch.isspace():
            if rng.random() < drop_prob:
                continue
            out.append(ch)
            continue

        out.append(ch)

        # Insert spacing after Korean syllables, not before punctuation/space.
        if is_hangul_syllable(ch) and rng.random() < insert_prob:
            nxt = text[i + 1] if i + 1 < len(text) else ""
            if nxt and (not nxt.isspace()) and nxt not in ".,!?;:)]}〉》』」'\"":
                out.append(" ")

    return "".join(out)


def corrupt_vowel(text: str, rng: random.Random, prob: float) -> str:
    out: List[str] = []

    for ch in text:
        dec = decompose(ch)
        if dec is None or rng.random() >= prob:
            out.append(ch)
            continue

        cho, jung, jong = dec
        cur = JUNG_LIST[jung]

        if cur in VOWEL_CONFUSIONS:
            new_v = rng.choice(VOWEL_CONFUSIONS[cur])
            new_jung = JUNG_LIST.index(new_v)
            out.append(compose(cho, new_jung, jong))
        else:
            out.append(ch)

    return "".join(out)


def corrupt_consonant(text: str, rng: random.Random, prob: float) -> str:
    out: List[str] = []

    for ch in text:
        dec = decompose(ch)
        if dec is None or rng.random() >= prob:
            out.append(ch)
            continue

        cho, jung, jong = dec
        cur = CHO_LIST[cho]

        if cur in CONSONANT_CONFUSIONS:
            new_c = rng.choice(CONSONANT_CONFUSIONS[cur])
            new_cho = CHO_LIST.index(new_c)
            out.append(compose(new_cho, jung, jong))
        else:
            out.append(ch)

    return "".join(out)


def corrupt_jong(text: str, rng: random.Random, prob: float) -> str:
    out: List[str] = []

    for ch in text:
        dec = decompose(ch)
        if dec is None or rng.random() >= prob:
            out.append(ch)
            continue

        cho, jung, jong = dec

        # If no final consonant, sometimes add a plausible final consonant.
        if jong == 0:
            new_jong_char = rng.choice(["ㄴ", "ㅇ", "ㄹ", "ㅁ", "ㄱ"])
            new_jong = JONG_LIST.index(new_jong_char)
        else:
            mode = rng.random()
            if mode < 0.45:
                # Drop final consonant.
                new_jong = 0
            elif mode < 0.75:
                # Confuse final consonant if possible.
                cur = JONG_LIST[jong]
                if cur in CONSONANT_CONFUSIONS:
                    cand = [x for x in CONSONANT_CONFUSIONS[cur] if x in JONG_LIST]
                    new_jong = JONG_LIST.index(rng.choice(cand)) if cand else 0
                else:
                    new_jong = rng.choice([0, jong])
            else:
                # Replace with common final consonant.
                new_jong = JONG_LIST.index(rng.choice(["ㄴ", "ㅇ", "ㄹ", "ㅁ", "ㄱ", "ㅂ"]))

        out.append(compose(cho, jung, new_jong))

    return "".join(out)


def corrupt_text(
    text: str,
    noise_type: str,
    rng: random.Random,
    noise_prob: float,
    spacing_drop_prob: float,
    spacing_insert_prob: float,
) -> str:
    if not text:
        return text

    if noise_type == "spacing":
        return corrupt_spacing(text, rng, spacing_drop_prob, spacing_insert_prob)
    if noise_type == "vowel":
        return corrupt_vowel(text, rng, noise_prob)
    if noise_type == "consonant":
        return corrupt_consonant(text, rng, noise_prob)
    if noise_type == "jong":
        return corrupt_jong(text, rng, noise_prob)
    if noise_type == "mixed":
        # Apply a moderate sequence of corruptions.
        x = corrupt_spacing(text, rng, spacing_drop_prob * 0.75, spacing_insert_prob * 0.75)
        x = corrupt_vowel(x, rng, noise_prob * 0.50)
        x = corrupt_consonant(x, rng, noise_prob * 0.40)
        x = corrupt_jong(x, rng, noise_prob * 0.50)
        return x

    raise ValueError(f"Unknown noise_type: {noise_type}")


def read_jsonl(path: Path, max_examples: Optional[int]) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_examples is not None and i >= max_examples:
                break
            line = line.strip()
            if not line:
                continue
            yield i, json.loads(line)


def make_pair(
    ex: Dict[str, Any],
    idx: int,
    noise_type: str,
    rng: random.Random,
    args,
) -> Optional[Dict[str, Any]]:
    clean_instruction = normalize_text(ex.get("instruction", ""))
    clean_input = normalize_text(ex.get("input", ""))
    output = normalize_text(ex.get("output", ""))

    if not output or not (clean_instruction or clean_input):
        return None

    if len(clean_instruction) + len(clean_input) < args.min_chars:
        return None
    if len(clean_instruction) + len(clean_input) > args.max_chars:
        return None

    noisy_instruction = clean_instruction
    noisy_input = clean_input

    if "instruction" in args.noise_fields:
        noisy_instruction = corrupt_text(
            clean_instruction,
            noise_type=noise_type,
            rng=rng,
            noise_prob=args.noise_prob,
            spacing_drop_prob=args.spacing_drop_prob,
            spacing_insert_prob=args.spacing_insert_prob,
        )

    if "input" in args.noise_fields:
        noisy_input = corrupt_text(
            clean_input,
            noise_type=noise_type,
            rng=rng,
            noise_prob=args.noise_prob,
            spacing_drop_prob=args.spacing_drop_prob,
            spacing_insert_prob=args.spacing_insert_prob,
        )

    changed = (clean_instruction != noisy_instruction) or (clean_input != noisy_input)

    if args.skip_unchanged and not args.keep_unchanged and not changed:
        return None

    row = dict(ex)

    row["pair_id"] = f"{idx}:{noise_type}"
    row["noise_type"] = noise_type
    row["text_changed"] = changed

    row["clean_instruction"] = clean_instruction
    row["clean_input"] = clean_input
    row["noisy_instruction"] = noisy_instruction
    row["noisy_input"] = noisy_input

    row["output"] = output
    row["source"] = ex.get("source", "unknown")

    if args.write_noisy_as_main:
        row["instruction"] = noisy_instruction
        row["input"] = noisy_input
    else:
        row["instruction"] = clean_instruction
        row["input"] = clean_input

    return row


def main():
    args = parse_args()

    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl)

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    rng = random.Random(args.seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    counts = {
        "read": 0,
        "written": 0,
        "skipped": 0,
        "changed": 0,
        "unchanged": 0,
    }
    by_noise = {k: 0 for k in args.noise_types}
    by_source: Dict[str, int] = {}

    with output_path.open("w", encoding="utf-8") as out:
        for idx, ex in tqdm(read_jsonl(input_path, args.max_examples), desc="Making noisy pairs"):
            counts["read"] += 1

            if args.one_pair_per_noise_type:
                noise_types = args.noise_types
            else:
                noise_types = [rng.choice(args.noise_types)]

            wrote_any = False
            for noise_type in noise_types:
                pair = make_pair(ex, idx, noise_type, rng, args)
                if pair is None:
                    continue

                out.write(json.dumps(pair, ensure_ascii=False) + "\n")
                counts["written"] += 1
                by_noise[noise_type] = by_noise.get(noise_type, 0) + 1
                by_source[pair.get("source", "unknown")] = by_source.get(pair.get("source", "unknown"), 0) + 1

                if pair["text_changed"]:
                    counts["changed"] += 1
                else:
                    counts["unchanged"] += 1

                wrote_any = True

            if not wrote_any:
                counts["skipped"] += 1

    stats = {
        "input_jsonl": str(input_path),
        "output_jsonl": str(output_path),
        "seed": args.seed,
        "max_examples": args.max_examples,
        "noise_types": args.noise_types,
        "noise_fields": args.noise_fields,
        "noise_prob": args.noise_prob,
        "spacing_drop_prob": args.spacing_drop_prob,
        "spacing_insert_prob": args.spacing_insert_prob,
        "one_pair_per_noise_type": args.one_pair_per_noise_type,
        "write_noisy_as_main": args.write_noisy_as_main,
        "counts": counts,
        "by_noise": by_noise,
        "by_source": by_source,
    }

    stats_path = output_path.with_suffix(output_path.suffix + ".stats.json")
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\n===== Done =====")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print("output:", output_path)
    print("stats:", stats_path)


if __name__ == "__main__":
    main()
