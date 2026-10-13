import json
import math
import csv
import sys
from pathlib import Path

ROOT = Path("/home/irteam/data-vol1/minsu/kolm")
RUN = Path((ROOT / "rebuttal_analysis/stronger_bench/results/latest_run.txt").read_text().strip())
OFF = RUN / "official_noise_eval"
NORM = RUN / "official_norm_eval"
OUT_DIR = RUN / "aggregate_official"
OUT_DIR.mkdir(exist_ok=True)

BENCHES = ["click", "kopiqa"]
NOISES = ["jamo", "mixed", "pronunciation"]
SETTINGS = ["clean", "noisy", "jamo"]

def exact_mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    prob = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * prob)

def bh_fdr(pvals):
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvals[i])
    q = [1.0] * m
    prev = 1.0
    for rev_rank, i in enumerate(reversed(order), start=1):
        rank = m - rev_rank + 1
        val = pvals[i] * m / rank
        prev = min(prev, val)
        q[i] = min(prev, 1.0)
    return q

def load(path):
    obj = json.loads(path.read_text())
    return [bool(x["correct"]) for x in obj["examples"]], obj["metrics"]["accuracy"], obj["metrics"]["n"]

missing = []
for bench in BENCHES:
    for noise in NOISES:
        for setting in SETTINGS:
            p = NORM / f"{bench}.{noise}.{setting}.norm.json"
            if not p.exists():
                missing.append(str(p))

print(f"[STATUS] official norm json count = {18 - len(missing)}/18")
if missing:
    print("[MISSING]")
    for x in missing:
        print(x)
    sys.exit(2)

rows = []
paired_rows = []

for bench in BENCHES:
    for noise in NOISES:
        row = {"benchmark": bench, "noise": noise}
        for setting in SETTINGS:
            base_p = OFF / f"{bench}.{noise}.{setting}.json"
            norm_p = NORM / f"{bench}.{noise}.{setting}.norm.json"

            base_corr, base_acc, n = load(base_p)
            norm_corr, norm_acc, n2 = load(norm_p)
            assert n == n2 and len(base_corr) == len(norm_corr)

            row[setting] = base_acc
            row[f"{setting}_norm"] = norm_acc
            row[f"{setting}_norm_delta"] = norm_acc - base_acc

            norm_only = sum(norm_corr[i] and not base_corr[i] for i in range(n))
            base_only = sum((not norm_corr[i]) and base_corr[i] for i in range(n))
            p = exact_mcnemar_p(norm_only, base_only)

            paired_rows.append({
                "benchmark": bench,
                "noise": noise,
                "setting": setting,
                "n": n,
                "base_acc": base_acc,
                "norm_acc": norm_acc,
                "delta": norm_acc - base_acc,
                "norm_correct_base_wrong": norm_only,
                "norm_wrong_base_correct": base_only,
                "p_mcnemar": p,
            })

        rows.append(row)

qvals = bh_fdr([r["p_mcnemar"] for r in paired_rows])
for r, q in zip(paired_rows, qvals):
    r["q_bh_all"] = q
    r["significant_0.05"] = q < 0.05

csv_path = OUT_DIR / "official_norm_summary.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    fieldnames = ["benchmark", "noise"]
    for s in SETTINGS:
        fieldnames += [s, f"{s}_norm", f"{s}_norm_delta"]
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
        w.writerow(r)

md_path = OUT_DIR / "official_norm_summary.md"
with open(md_path, "w", encoding="utf-8") as f:
    f.write("| Benchmark | Noise | Clean | Clean+Norm | Δ | Noisy | Noisy+Norm | Δ | +Jamo | +Jamo+Norm | Δ |\n")
    f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in rows:
        f.write(
            f"| {r['benchmark']} | {r['noise']} | "
            f"{r['clean']:.4f} | {r['clean_norm']:.4f} | {r['clean_norm_delta']:.4f} | "
            f"{r['noisy']:.4f} | {r['noisy_norm']:.4f} | {r['noisy_norm_delta']:.4f} | "
            f"{r['jamo']:.4f} | {r['jamo_norm']:.4f} | {r['jamo_norm_delta']:.4f} |\n"
        )

csv_paired = OUT_DIR / "official_norm_paired_mcnemar.csv"
with open(csv_paired, "w", newline="", encoding="utf-8") as f:
    fieldnames = [
        "benchmark", "noise", "setting", "n",
        "base_acc", "norm_acc", "delta",
        "norm_correct_base_wrong", "norm_wrong_base_correct",
        "p_mcnemar", "q_bh_all", "significant_0.05",
    ]
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in paired_rows:
        w.writerow(r)

md_paired = OUT_DIR / "official_norm_paired_mcnemar.md"
with open(md_paired, "w", encoding="utf-8") as f:
    f.write("| Benchmark | Noise | Setting | n | Base | Norm | Δ | b/c | p | q | sig |\n")
    f.write("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|\n")
    for r in paired_rows:
        f.write(
            f"| {r['benchmark']} | {r['noise']} | {r['setting']} | {r['n']} | "
            f"{r['base_acc']:.4f} | {r['norm_acc']:.4f} | {r['delta']:.4f} | "
            f"{r['norm_correct_base_wrong']}/{r['norm_wrong_base_correct']} | "
            f"{r['p_mcnemar']:.3g} | {r['q_bh_all']:.3g} | {r['significant_0.05']} |\n"
        )

print("[WROTE]", csv_path)
print("[WROTE]", md_path)
print("[WROTE]", csv_paired)
print("[WROTE]", md_paired)
print()
print("===== NORM SUMMARY =====")
print(md_path.read_text())
print()
print("===== NORM PAIRED =====")
print(md_paired.read_text())
