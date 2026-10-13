import json
import random
import sys
from pathlib import Path

ROOT = Path("/home/irteam/data-vol1/minsu/kolm")
sys.path.insert(0, str(ROOT))

from make_noisy_korean_benchmarks import corrupt
from make_normalized_noisy_benchmarks import normalize_obj

IN_DIR = ROOT / "rebuttal_analysis/stronger_bench/data/jsonl"
OUT_DIR = ROOT / "rebuttal_analysis/stronger_bench/data/jsonl_official"
OUT_NORM_DIR = ROOT / "rebuttal_analysis/stronger_bench/data/jsonl_official_norm"

OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_NORM_DIR.mkdir(parents=True, exist_ok=True)

BENCHES = ["click", "kopiqa"]
NOISES = ["clean", "spacing", "jamo", "pronunciation", "mixed"]
TEXT_KEYS = ["question", "paragraph", "context", "passage", "prompt", "instruction", "input"]
SEED = 42
P = 0.12

def corrupt_record(r, noise, rng):
    r = dict(r)
    r["noise_type"] = noise
    for k in TEXT_KEYS:
        if k in r and isinstance(r[k], str):
            r[k] = corrupt(r[k], noise, rng, P)
    # choices are intentionally not corrupted, matching current stronger_bench setup.
    return r

def read_jsonl(p):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]

def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

for bench in BENCHES:
    clean_rows = read_jsonl(IN_DIR / f"{bench}.clean.jsonl")

    for noise in NOISES:
        rng = random.Random(SEED)

        if noise == "clean":
            rows = [dict(r, noise_type="clean") for r in clean_rows]
        else:
            rows = [corrupt_record(r, noise, rng) for r in clean_rows]

        out = OUT_DIR / f"{bench}.{noise}.jsonl"
        write_jsonl(out, rows)

        norm_rows = [normalize_obj(r) for r in rows]
        out_norm = OUT_NORM_DIR / f"{bench}.{noise}.norm.jsonl"
        write_jsonl(out_norm, norm_rows)

        print("WROTE", out, "n=", len(rows))
        print("WROTE", out_norm, "n=", len(norm_rows))

print("DONE")
