#!/usr/bin/env python3
# make_noisy_korean_benchmarks.py
from __future__ import annotations
import argparse, csv, json, random, re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

COMPAT_CHO = list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
COMPAT_JUNG_FULL = ["ㅏ","ㅐ","ㅑ","ㅒ","ㅓ","ㅔ","ㅕ","ㅖ","ㅗ","ㅘ","ㅙ","ㅚ","ㅛ","ㅜ","ㅝ","ㅞ","ㅟ","ㅠ","ㅡ","ㅢ","ㅣ"]
COMPAT_JONG_FULL = ["", "ㄱ","ㄲ","ㄳ","ㄴ","ㄵ","ㄶ","ㄷ","ㄹ","ㄺ","ㄻ","ㄼ","ㄽ","ㄾ","ㄿ","ㅀ","ㅁ","ㅂ","ㅄ","ㅅ","ㅆ","ㅇ","ㅈ","ㅊ","ㅋ","ㅌ","ㅍ","ㅎ"]
PRON_SUBS = [("의","에"),("예","에"),("왜","외"),("않","안"),("돼","되"),("되","돼"),("어","으"),("여","이")]

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=["kmmlu","kobest"])
    p.add_argument("--output_dir", required=True)
    p.add_argument("--noise_types", nargs="+", default=["clean","spacing","jamo","pronunciation","mixed"])
    p.add_argument("--noise_prob", type=float, default=0.12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_examples_per_task", type=int, default=2000)
    p.add_argument("--kmmlu_local")
    p.add_argument("--kobest_local")
    p.add_argument("--kmmlu_hf_name", default="haerae-hub/KMMLU")
    p.add_argument("--kobest_hf_name", default="skt/kobest_v1")
    p.add_argument("--split", default="test")
    p.add_argument("--corrupt_choices", action="store_true")
    return p.parse_args()

def is_hangul(ch): return 0xAC00 <= ord(ch) <= 0xD7A3

def decomp(ch):
    code=ord(ch)-0xAC00
    if code<0 or code>11171: return ch
    cho=code//588; jung=(code%588)//28; jong=code%28
    return COMPAT_CHO[cho]+COMPAT_JUNG_FULL[jung]+(COMPAT_JONG_FULL[jong] if jong else "")

def jamo(s,rng,p):
    return "".join(decomp(ch) if is_hangul(ch) and rng.random()<p else ch for ch in s)

def spacing(s,rng,p):
    out=[]
    for i,ch in enumerate(s):
        if ch.isspace():
            if rng.random()<p*0.7: continue
            out.append(ch); continue
        out.append(ch)
        if is_hangul(ch) and i+1<len(s) and is_hangul(s[i+1]) and rng.random()<p*0.25:
            out.append(" ")
    return "".join(out)

def pron(s,rng,p):
    for a,b in PRON_SUBS:
        if a in s and rng.random()<p: s=s.replace(a,b,1)
    out=[]
    for ch in s:
        if is_hangul(ch) and rng.random()<p*0.35:
            code=ord(ch)-0xAC00; cho=code//588; jung=(code%588)//28; jong=code%28
            if jong: out.append(chr(0xAC00+cho*588+jung*28+0))
            else: out.append(ch)
        else: out.append(ch)
    return "".join(out)

def corrupt(s,nt,rng,p):
    if nt=="clean": return s
    if nt=="jamo": return jamo(s,rng,p)
    if nt=="spacing": return spacing(s,rng,p)
    if nt=="pronunciation": return pron(s,rng,p)
    if nt=="mixed": return jamo(pron(spacing(s,rng,p),rng,p),rng,p*0.7)
    raise ValueError(nt)

def read_local(path):
    p=Path(path)
    if p.suffix.lower()==".jsonl":
        return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    rows=[]
    for enc in ["utf-8-sig","utf-8","cp949"]:
        try:
            with p.open("r",encoding=enc,newline="") as f:
                sample=f.read(4096); f.seek(0)
                try: dialect=csv.Sniffer().sniff(sample)
                except Exception: dialect=csv.excel
                rows=list(csv.DictReader(f,dialect=dialect))
            return rows
        except Exception: pass
    return rows

def norm_ans(ans, choices):
    if ans is None: return None
    if isinstance(ans,int):
        if 0<=ans<len(choices): return ans
        if 1<=ans<=len(choices): return ans-1
    s=str(ans).strip()
    if s.isdigit():
        v=int(s)
        if 0<=v<len(choices): return v
        if 1<=v<=len(choices): return v-1
    mp={"A":0,"B":1,"C":2,"D":3,"E":4,"a":0,"b":1,"c":2,"d":3,"e":4}
    if s in mp and mp[s]<len(choices): return mp[s]
    for i,c in enumerate(choices):
        if s==str(c).strip(): return i
    return None


def extract(ex,task,subset,i,source):

    # -------------------------
    # Explicit KoBEST schemas
    # -------------------------
    if task == "kobest":
        sub = str(subset).lower()

        if sub == "boolq":
            q = (
                f"지문: {ex.get('paragraph','')}"
                f"질문: {ex.get('question','')}"
            )
            choices = ["아니오", "예"]
            ans = norm_ans(ex.get("label"), choices)

            if ans is None:
                return None

            return {
                "id": ex.get("id", f"{subset}-{i}"),
                "task": task,
                "subset": subset,
                "question": q.strip(),
                "choices": choices,
                "answer": ans,
                "source": source,
            }

        if sub == "copa":
            q = (
                f"전제: {ex.get('premise','')}"
                f"질문 유형: {ex.get('question','')}"
            )

            choices = [
                str(ex.get("alternative_1", "")),
                str(ex.get("alternative_2", "")),
            ]

            ans = norm_ans(ex.get("label"), choices)

            if ans is None:
                return None

            return {
                "id": ex.get("id", f"{subset}-{i}"),
                "task": task,
                "subset": subset,
                "question": q.strip(),
                "choices": choices,
                "answer": ans,
                "source": source,
            }

        if sub == "hellaswag":
            q = (
                f"문맥: {ex.get('context','')}"
                f"다음에 이어질 가장 자연스러운 문장은?"
            )

            choices = [
                str(ex.get(f"ending_{j}", ""))
                for j in range(1, 5)
            ]

            ans = norm_ans(ex.get("label"), choices)

            if ans is None:
                return None

            return {
                "id": ex.get("id", f"{subset}-{i}"),
                "task": task,
                "subset": subset,
                "question": q.strip(),
                "choices": choices,
                "answer": ans,
                "source": source,
            }

        if sub == "sentineg":
            q = (
                f"문장: {ex.get('sentence','')}"
                f"이 문장의 감성은?"
            )

            choices = ["부정", "긍정"]

            ans = norm_ans(ex.get("label"), choices)

            if ans is None:
                return None

            return {
                "id": ex.get("id", f"{subset}-{i}"),
                "task": task,
                "subset": subset,
                "question": q.strip(),
                "choices": choices,
                "answer": ans,
                "source": source,
            }

        if sub == "wic":
            q = (
                f"단어: {ex.get('word','')}"
                f"문맥 1: {ex.get('context_1','')}"
                f"문맥 2: {ex.get('context_2','')}"
                f"두 문맥에서 단어 의미가 같은가?"
            )

            choices = ["다르다", "같다"]

            ans = norm_ans(ex.get("label"), choices)

            if ans is None:
                return None

            return {
                "id": ex.get("id", f"{subset}-{i}"),
                "task": task,
                "subset": subset,
                "question": q.strip(),
                "choices": choices,
                "answer": ans,
                "source": source,
            }

    # -------------------------
    # Generic MCQ parser
    # -------------------------
    q = None

    for k in [
        "question",
        "query",
        "input",
        "premise",
        "sentence",
        "context",
        "paragraph",
        "passage",
    ]:
        if k in ex and ex[k] is not None and str(ex[k]).strip():
            q = str(ex[k])
            break

    if not q:
        return None

    choices = None

    for k in ["choices", "options", "candidates", "answer_choices"]:
        if k in ex:
            v = ex[k]

            if isinstance(v, list):
                choices = [str(x) for x in v]

            elif isinstance(v, str):
                try:
                    vv = json.loads(v)

                    if isinstance(vv, list):
                        choices = [str(x) for x in vv]

                except Exception:
                    pass

    if choices is None:
        vals = []

        for k in [
            "A","B","C","D","E",
            "a","b","c","d","e",
            "option1","option2","option3","option4","option5"
        ]:
            if k in ex and ex[k] is not None and str(ex[k]).strip():
                vals.append(str(ex[k]))

        if len(vals) >= 2:
            choices = vals

    if not choices or len(choices) < 2:
        return None

    ans = None

    for k in [
        "answer",
        "label",
        "gold",
        "target",
        "correct",
        "correct_answer",
    ]:
        if k in ex:
            ans = norm_ans(ex[k], choices)

            if ans is not None:
                break

    if ans is None:
        return None

    return {
        "id": ex.get("id", f"{subset}-{i}"),
        "task": task,
        "subset": subset,
        "question": q.strip(),
        "choices": [c.strip() for c in choices],
        "answer": ans,
        "source": source,
    }

def load_hf(name,task,split):
    try:
        from datasets import get_dataset_config_names, load_dataset
    except Exception as e:
        print("[WARN] datasets unavailable",e); return []
    try: configs=get_dataset_config_names(name)
    except Exception as e:
        print("[WARN] no configs", name, e); configs=[None]
    rows=[]
    for cfg in configs:
        ds=None; used=None
        for sp in [split,"validation","test","dev","train"]:
            try:
                ds=load_dataset(name,cfg,split=sp,) if cfg else load_dataset(name,split=sp,)
                used=sp; break
            except Exception: pass
        if ds is None: continue
        subset=str(cfg or name.split("/")[-1])
        print("[INFO] loaded", name, subset, used, len(ds))
        for i,ex in enumerate(ds):
            item=extract(dict(ex),task,subset,i,f"{name}:{subset}:{used}")
            if item: rows.append(item)
    return rows

def save(rows,out_dir,task,noise_types,seed,prob,corrupt_choices,maxn):
    if maxn: rows=rows[:maxn]
    out_dir.mkdir(parents=True,exist_ok=True)
    for nt in noise_types:
        rng=random.Random(seed + abs(hash((task,nt)))%1000000)
        out=out_dir/f"{task}.{nt}.jsonl"
        with out.open("w",encoding="utf-8") as f:
            for ex in rows:
                y=dict(ex); y["noise_type"]=nt
                y["question"]=corrupt(ex["question"],nt,rng,prob)
                if corrupt_choices: y["choices"]=[corrupt(c,nt,rng,prob) for c in ex["choices"]]
                f.write(json.dumps(y,ensure_ascii=False)+"\n")
        print("[SAVED]", out, "n=", len(rows))

def main():
    a=parse_args(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    for task in a.tasks:
        if task=="kmmlu":
            raw=read_local(a.kmmlu_local) if a.kmmlu_local else load_hf(a.kmmlu_hf_name,task,a.split)
        else:
            raw=read_local(a.kobest_local) if a.kobest_local else load_hf(a.kobest_hf_name,task,a.split)
        rows=[]
        if raw and isinstance(raw[0],dict) and "question" in raw[0] and "choices" in raw[0] and "answer" in raw[0] and "task" in raw[0]:
            rows=raw
        else:
            for i,ex in enumerate(raw):
                item=extract(ex,task,Path(a.kmmlu_local or a.kobest_local or task).stem,i,"local_or_hf")
                if item: rows.append(item)
        seen=set(); ded=[]
        for r in rows:
            key=(r["question"],tuple(r["choices"]),r["answer"])
            if key not in seen: seen.add(key); ded.append(r)
        print("[INFO]",task,"usable",len(ded))
        if ded: save(ded,out,task,a.noise_types,a.seed,a.noise_prob,a.corrupt_choices,a.max_examples_per_task)
    stats={p.name:sum(1 for _ in p.open(encoding="utf-8")) for p in out.glob("*.jsonl")}
    (out/"stats.json").write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding="utf-8")
    print("[STATS]",json.dumps(stats,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
