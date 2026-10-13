import json
from pathlib import Path

DATA_DIR = Path("/home/irteam/data-vol1/minsu/kolm/rebuttal_analysis/stronger_bench/data/jsonl")
OUT_DIR = Path("/home/irteam/data-vol1/minsu/kolm/rebuttal_analysis/stronger_bench/data/jsonl_norm")
OUT_DIR.mkdir(parents=True, exist_ok=True)

L_COMPAT = {
    "ㄱ":0, "ㄲ":1, "ㄴ":2, "ㄷ":3, "ㄸ":4, "ㄹ":5, "ㅁ":6, "ㅂ":7, "ㅃ":8,
    "ㅅ":9, "ㅆ":10, "ㅇ":11, "ㅈ":12, "ㅉ":13, "ㅊ":14, "ㅋ":15, "ㅌ":16, "ㅍ":17, "ㅎ":18,
}
V_COMPAT = {
    "ㅏ":0, "ㅐ":1, "ㅑ":2, "ㅒ":3, "ㅓ":4, "ㅔ":5, "ㅕ":6, "ㅖ":7, "ㅗ":8,
    "ㅘ":9, "ㅙ":10, "ㅚ":11, "ㅛ":12, "ㅜ":13, "ㅝ":14, "ㅞ":15, "ㅟ":16, "ㅠ":17,
    "ㅡ":18, "ㅢ":19, "ㅣ":20,
}
T_COMPAT = {
    "":0, "ㄱ":1, "ㄲ":2, "ㄳ":3, "ㄴ":4, "ㄵ":5, "ㄶ":6, "ㄷ":7, "ㄹ":8, "ㄺ":9,
    "ㄻ":10, "ㄼ":11, "ㄽ":12, "ㄾ":13, "ㄿ":14, "ㅀ":15, "ㅁ":16, "ㅂ":17, "ㅄ":18,
    "ㅅ":19, "ㅆ":20, "ㅇ":21, "ㅈ":22, "ㅊ":23, "ㅋ":24, "ㅌ":25, "ㅍ":26, "ㅎ":27,
}

def compose_syllable(l, v, t=""):
    return chr(0xAC00 + ((L_COMPAT[l] * 21 + V_COMPAT[v]) * 28 + T_COMPAT.get(t, 0)))

def normalize_jamo_sequences(s):
    out = []
    i = 0
    n = len(s)

    while i < n:
        ch = s[i]

        # Compose compatibility jamo sequence: L + V + optional T
        if ch in L_COMPAT and i + 1 < n and s[i + 1] in V_COMPAT:
            l = ch
            v = s[i + 1]
            t = ""
            step = 2

            if i + 2 < n and s[i + 2] in T_COMPAT and s[i + 2] not in V_COMPAT:
                # Use as final consonant only when the following char is not a vowel.
                if not (i + 3 < n and s[i + 3] in V_COMPAT):
                    t = s[i + 2]
                    step = 3

            out.append(compose_syllable(l, v, t))
            i += step
            continue

        out.append(ch)
        i += 1

    return "".join(out)

def normalize_text(x):
    if not isinstance(x, str):
        return x
    return normalize_jamo_sequences(x)

def normalize_record(r):
    r = dict(r)
    for k in ["paragraph", "context", "passage", "prompt", "question", "instruction", "input"]:
        if k in r:
            r[k] = normalize_text(r[k])
    if "choices" in r and isinstance(r["choices"], list):
        r["choices"] = [normalize_text(c) for c in r["choices"]]
    r["normalized"] = True
    return r

for src in sorted(DATA_DIR.glob("*.jsonl")):
    rows = [json.loads(x) for x in src.read_text(encoding="utf-8").splitlines() if x.strip()]
    out = OUT_DIR / src.name.replace(".jsonl", ".norm.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(normalize_record(r), ensure_ascii=False) + "\n")
    print("WROTE", out, "n=", len(rows))

print("DONE", OUT_DIR)
