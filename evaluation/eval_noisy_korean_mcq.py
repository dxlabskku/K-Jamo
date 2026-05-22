#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import torch, torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--model_name_or_path",required=True)
    p.add_argument("--input_jsonl",required=True)
    p.add_argument("--output_json",required=True)
    p.add_argument("--dtype",choices=["bf16","fp16","fp32"],default="bf16")
    p.add_argument("--max_length",type=int,default=1024)
    p.add_argument("--max_examples",type=int,default=None)
    p.add_argument("--trust_remote_code",action="store_true")
    return p.parse_args()
def dt(x): return torch.bfloat16 if x=="bf16" else torch.float16 if x=="fp16" else torch.float32
def load(path,maxn=None):
    rows=[]
    for line in Path(path).open(encoding="utf-8"):
        if line.strip(): rows.append(json.loads(line))
        if maxn and len(rows)>=maxn: break
    return rows
def prefix(ex):
    lines=[f"질문: {ex['question'].strip()}","선택지:"]
    for i,c in enumerate(ex["choices"]): lines.append(f"{chr(65+i)}. {c}")
    lines.append("정답:")
    return "\n".join(lines)
@torch.no_grad()
def score(model,tok,pref,cont,device,maxlen):
    full=pref+" "+cont
    ef=tok(full,return_tensors="pt",truncation=True,max_length=maxlen,add_special_tokens=True)
    ep=tok(pref,return_tensors="pt",truncation=True,max_length=maxlen,add_special_tokens=True)
    ids=ef["input_ids"].to(device); mask=ef.get("attention_mask")
    mask=mask.to(device) if mask is not None else None
    plen=ep["input_ids"].shape[1]
    if ids.shape[1]<=plen: return -1e9
    out=model(input_ids=ids,attention_mask=mask,use_cache=False)
    logits=out.logits[:,:-1,:].float(); labels=ids[:,1:]
    start=max(plen-1,0)
    logits=logits[:,start:,:]; labels=labels[:,start:]
    lp=F.log_softmax(logits,dim=-1).gather(-1,labels.unsqueeze(-1)).squeeze(-1)
    return float(lp.sum().detach().cpu())
def main():
    a=parse_args(); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok=AutoTokenizer.from_pretrained(a.model_name_or_path,trust_remote_code=a.trust_remote_code, use_fast=False)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    kwargs={} if a.dtype=="fp32" else {"torch_dtype":dt(a.dtype)}
    model=AutoModelForCausalLM.from_pretrained(a.model_name_or_path,trust_remote_code=a.trust_remote_code,**kwargs).to(device).eval()
    rows=load(a.input_jsonl,a.max_examples); ok=0; by={}; examples=[]
    for ex in tqdm(rows,desc="MCQ eval"):
        pref=prefix(ex); scores=[score(model,tok,pref,c,device,a.max_length) for c in ex["choices"]]
        pred=max(range(len(scores)),key=lambda i:scores[i]); gold=int(ex["answer"]); good=int(pred==gold); ok+=good
        sub=ex.get("subset","unknown"); by.setdefault(sub,[0,0]); by[sub][0]+=good; by[sub][1]+=1
        if len(examples)<50: examples.append({"id":ex.get("id"),"subset":sub,"noise_type":ex.get("noise_type"),"gold":gold,"pred":pred,"scores":scores})
    n=len(rows); metrics={"n":n,"accuracy":ok/n if n else None,"by_subset":{k:{"accuracy":v[0]/v[1],"n":v[1]} for k,v in by.items()}}
    out={"model_name_or_path":a.model_name_or_path,"input_jsonl":a.input_jsonl,"metrics":metrics,"examples":examples}
    op=Path(a.output_json); op.parent.mkdir(parents=True,exist_ok=True); op.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(metrics,ensure_ascii=False,indent=2)); print("saved:",op)
if __name__=="__main__": main()
