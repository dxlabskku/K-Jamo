import argparse
import json
import re
import unicodedata
from pathlib import Path

# Compatibility jamo -> conjoining jamo maps for modern Hangul composition.
CHO_COMPAT = {
    "ㄱ":"ᄀ","ㄲ":"ᄁ","ㄴ":"ᄂ","ㄷ":"ᄃ","ㄸ":"ᄄ","ㄹ":"ᄅ","ㅁ":"ᄆ","ㅂ":"ᄇ","ㅃ":"ᄈ",
    "ㅅ":"ᄉ","ㅆ":"ᄊ","ㅇ":"ᄋ","ㅈ":"ᄌ","ㅉ":"ᄍ","ㅊ":"ᄎ","ㅋ":"ᄏ","ㅌ":"ᄐ","ㅍ":"ᄑ","ㅎ":"ᄒ",
}
JUNG_COMPAT = {
    "ㅏ":"ᅡ","ㅐ":"ᅢ","ㅑ":"ᅣ","ㅒ":"ᅤ","ㅓ":"ᅥ","ㅔ":"ᅦ","ㅕ":"ᅧ","ㅖ":"ᅨ","ㅗ":"ᅩ",
    "ㅘ":"ᅪ","ㅙ":"ᅫ","ㅚ":"ᅬ","ㅛ":"ᅭ","ㅜ":"ᅮ","ㅝ":"ᅯ","ㅞ":"ᅰ","ㅟ":"ᅱ","ㅠ":"ᅲ",
    "ㅡ":"ᅳ","ㅢ":"ᅴ","ㅣ":"ᅵ",
}
JONG_COMPAT = {
    "ㄱ":"ᆨ","ㄲ":"ᆩ","ㄳ":"ᆪ","ㄴ":"ᆫ","ㄵ":"ᆬ","ㄶ":"ᆭ","ㄷ":"ᆮ","ㄹ":"ᆯ","ㄺ":"ᆰ",
    "ㄻ":"ᆱ","ㄼ":"ᆲ","ㄽ":"ᆳ","ㄾ":"ᆴ","ㄿ":"ᆵ","ㅀ":"ᆶ","ㅁ":"ᆷ","ㅂ":"ᆸ","ㅄ":"ᆹ",
    "ㅅ":"ᆺ","ㅆ":"ᆻ","ㅇ":"ᆼ","ㅈ":"ᆽ","ㅊ":"ᆾ","ㅋ":"ᆿ","ㅌ":"ᇀ","ㅍ":"ᇁ","ㅎ":"ᇂ",
}

VOWELS = set(JUNG_COMPAT)
CONSONANTS = set(CHO_COMPAT)

def compat_to_conjoining_for_composition(s: str) -> str:
    # Greedy local conversion:
    # consonant + vowel => choseong + jungseong
    # consonant + vowel + consonant where next is not vowel => jongseong
    out = []
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch in CONSONANTS and i + 1 < n and s[i + 1] in VOWELS:
            out.append(CHO_COMPAT[ch])
            out.append(JUNG_COMPAT[s[i + 1]])
            i += 2
            if i < n and s[i] in CONSONANTS and not (i + 1 < n and s[i + 1] in VOWELS):
                out.append(JONG_COMPAT.get(s[i], s[i]))
                i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)

def normalize_ko_text(s: str) -> str:
    if not isinstance(s, str):
        return s
    s = unicodedata.normalize("NFKC", s)
    s = compat_to_conjoining_for_composition(s)
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

SKIP_KEYS = {
    "id", "idx", "qid", "source", "task", "dataset",
    "label", "answer", "gold", "target", "correct", "split",
    "noise", "noise_type"
}

def normalize_obj(x):
    if isinstance(x, dict):
        y = {}
        for k, v in x.items():
            if k in SKIP_KEYS:
                y[k] = v
            else:
                y[k] = normalize_obj(v)
        return y
    if isinstance(x, list):
        return [normalize_obj(v) for v in x]
    if isinstance(x, str):
        return normalize_ko_text(x)
    return x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="data/noisy_korean_benchmarks")
    ap.add_argument("--out_dir", default="data/noisy_korean_benchmarks_normalized")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for src in sorted(in_dir.glob("*.jsonl")):
        dst = out_dir / src.name
        n = 0
        changed = 0
        with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
            for line in fin:
                if not line.strip():
                    continue
                obj = json.loads(line)
                norm = normalize_obj(obj)
                if norm != obj:
                    changed += 1
                fout.write(json.dumps(norm, ensure_ascii=False) + "\n")
                n += 1
        print(f"[SAVED] {dst} n={n} changed={changed}")

if __name__ == "__main__":
    main()
