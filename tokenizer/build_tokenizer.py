# build_jamo_augmented_tokenizer.py

import json
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


MODEL_NAME = "EleutherAI/polyglot-ko-3.8b"
SAVE_DIR = "./polyglot-ko-3.8b-jamo-tokenizer"

CHO = [
    "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ",
    "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"
]

JUNG = [
    "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ",
    "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ"
]

JONG = [
    "", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ",
    "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ",
    "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"
]


def is_hangul_syllable(ch):
    return 0xAC00 <= ord(ch) <= 0xD7A3


def decompose_syllable(ch):
    code = ord(ch) - 0xAC00
    cho_idx = code // 588
    jung_idx = (code % 588) // 28
    jong_idx = code % 28
    return CHO[cho_idx], JUNG[jung_idx], JONG[jong_idx]


def text_to_jamo_patterns(text, max_len=8):
    chos, jungs, jongs = [], [], []

    for ch in text:
        if is_hangul_syllable(ch):
            c, v, f = decompose_syllable(ch)
            chos.append(c)
            jungs.append(v)
            jongs.append(f if f else "_")

    if not chos:
        return []

    if len(chos) > max_len:
        return []

    return [
        "<CHO:" + "".join(chos) + ">",
        "<JUNG:" + "".join(jungs) + ">",
        "<JONG:" + "".join(jongs) + ">",
    ]


def build_jamo_tokens(tokenizer, max_len=8):
    new_tokens = set()

    base_vocab_size = tokenizer.vocab_size

    for token_id in range(base_vocab_size):
        decoded = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)

        patterns = text_to_jamo_patterns(decoded, max_len=max_len)

        for p in patterns:
            new_tokens.add(p)

    return sorted(new_tokens)


def main():
    save_path = Path(SAVE_DIR)
    save_path.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("BASE vocab_size:", tokenizer.vocab_size)
    print("BASE len(tokenizer):", len(tokenizer))

    jamo_tokens = build_jamo_tokens(tokenizer, max_len=8)

    print("Generated jamo tokens:", len(jamo_tokens))
    print("Examples:", jamo_tokens[:30])

    added = tokenizer.add_special_tokens(
        {"additional_special_tokens": jamo_tokens}
    )

    print("Actually added:", added)
    print("NEW len(tokenizer):", len(tokenizer))

    tokenizer.save_pretrained(SAVE_DIR)

    with open(save_path / "jamo_tokens.json", "w", encoding="utf-8") as f:
        json.dump(jamo_tokens, f, ensure_ascii=False, indent=2)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    model.resize_token_embeddings(len(tokenizer))
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)

    print("Saved to:", SAVE_DIR)


if __name__ == "__main__":
    main()