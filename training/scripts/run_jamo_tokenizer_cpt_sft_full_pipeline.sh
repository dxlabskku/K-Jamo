#!/usr/bin/env bash
# run_jamo_tokenizer_cpt_sft_full_pipeline.sh
# Full pipeline: extended-tokenizer CPT -> SFT -> KMMLU/KoBEST + clean/noisy PPL.
# Uses torchrun --master_port 29501 by default.

set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export PATH="${CONDA_PREFIX:-}/bin:${PATH}"
hash -r

PROJECT_DIR="${PROJECT_DIR:-your_path}"
cd "${PROJECT_DIR}"

PYTHON="${PYTHON:-}"

MODEL_FAMILY="${MODEL_FAMILY:-polyglot}"   # qwen | polyglot
BASE_MODEL="${BASE_MODEL:-EleutherAI/polyglot-ko-3.8b}"
MODEL_TAG="${MODEL_TAG:-polyglot_3.8b}"
TEMPLATE_NAME="${TEMPLATE_NAME:-simple}"
JAMO_TOKENIZER_PATH="${JAMO_TOKENIZER_PATH:-./polyglot-ko-3.8b-jamo-tokenizer-sft}"

DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/data/processed/instruction}"
BASE_TRAIN_JSONL="${BASE_TRAIN_JSONL:-${DATA_DIR}/instruction_train.balanced_250k.jsonl}"
EVAL_JSONL="${EVAL_JSONL:-${DATA_DIR}/instruction_valid.jsonl}"

JAMO_RATIO="${JAMO_RATIO:-0.05}"
JAMO_RATIO_TAG="${JAMO_RATIO_TAG:-in05}"
JAMO_SUBST_JSONL="${JAMO_SUBST_JSONL:-${DATA_DIR}/instruction_train.jamo_subst.${JAMO_RATIO_TAG}.out0.jsonl}"
JAMO_SUBST_TXT="${JAMO_SUBST_TXT:-${DATA_DIR}/instruction_train.jamo_subst.${JAMO_RATIO_TAG}.out0.txt}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs_jamo_tokenizer_cpt_sft/${MODEL_TAG}}"
EVAL_ROOT="${EVAL_ROOT:-${PROJECT_DIR}/eval_results_jamo_tokenizer_cpt_sft/${MODEL_TAG}}"

TRAIN_GPUS="${TRAIN_GPUS:-1,2,3,4}"
EVAL_GPU="${EVAL_GPU:-1}"
NPROC="${NPROC:-4}"
MASTER_PORT="${MASTER_PORT:-29501}"

SEED="${SEED:-42}"
MAX_LENGTH="${MAX_LENGTH:-1024}"

CPT_MAX_STEPS="${CPT_MAX_STEPS:-5000}"
CPT_BLOCK_SIZE="${CPT_BLOCK_SIZE:-1024}"
CPT_PER_DEVICE_BATCH="${CPT_PER_DEVICE_BATCH:-2}"
CPT_GRAD_ACCUM="${CPT_GRAD_ACCUM:-8}"
CPT_WEIGHT_DECAY="${CPT_WEIGHT_DECAY:-0.01}"
CPT_WARMUP_RATIO="${CPT_WARMUP_RATIO:-0.03}"
CPT_SAVE_STEPS="${CPT_SAVE_STEPS:-1000}"
CPT_LOGGING_STEPS="${CPT_LOGGING_STEPS:-25}"
CPT_EVAL_STEPS="${CPT_EVAL_STEPS:-1000}"
CPT_SAVE_TOTAL_LIMIT="${CPT_SAVE_TOTAL_LIMIT:-2}"

SFT_MAX_STEPS="${SFT_MAX_STEPS:-1000}"
SFT_PER_DEVICE_BATCH="${SFT_PER_DEVICE_BATCH:-1}"
SFT_GRAD_ACCUM="${SFT_GRAD_ACCUM:-16}"
SFT_LR="${SFT_LR:-5e-6}"
SFT_SAVE_STEPS="${SFT_SAVE_STEPS:-500}"
SFT_LOGGING_STEPS="${SFT_LOGGING_STEPS:-25}"
SFT_SAVE_TOTAL_LIMIT="${SFT_SAVE_TOTAL_LIMIT:-2}"

RUN_CPT="${RUN_CPT:-1}"
RUN_SFT="${RUN_SFT:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_LMEVAL="${RUN_LMEVAL:-1}"
RUN_PPL="${RUN_PPL:-1}"

LM_EVAL_BATCH_SIZE="${LM_EVAL_BATCH_SIZE:-4}"
PPL_BATCH_SIZE="${PPL_BATCH_SIZE:-1}"

if [[ "${MODEL_FAMILY}" == "qwen" ]]; then
  CPT_LRS="${CPT_LRS:-5e-6 1e-5 2e-5}"
elif [[ "${MODEL_FAMILY}" == "polyglot" ]]; then
  CPT_LRS="${CPT_LRS:-1e-5 2e-5 5e-5}"
else
  CPT_LRS="${CPT_LRS:-1e-5}"
fi

timestamp() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo; echo "[$(timestamp)] $*"; echo; }

safe_lr_tag() {
  echo "$1" | sed 's/-/m/g' | sed 's/\.//g'
}

require_file() {
  local p="$1"
  if [[ ! -e "${p}" ]]; then
    echo "[ERROR] Missing: ${p}" >&2
    exit 1
  fi
}

require_cmd() {
  local c="$1"
  if ! command -v "${c}" >/dev/null 2>&1; then
    echo "[ERROR] Missing command: ${c}" >&2
    exit 1
  fi
}

make_jamo_subst_jsonl_if_needed() {
  if [[ -f "${JAMO_SUBST_JSONL}" ]]; then
    log "Jamo-subst jsonl exists: ${JAMO_SUBST_JSONL}"
    return 0
  fi
  log "Making jamo-subst jsonl ratio=${JAMO_RATIO}"
  "${PYTHON}" make_jamo_substitution_instruction_data.py \
    --input_jsonl "${BASE_TRAIN_JSONL}" \
    --output_jsonl "${JAMO_SUBST_JSONL}" \
    --input_jamo_ratio "${JAMO_RATIO}" \
    --seed "${SEED}"
}

make_cpt_txt_if_needed() {
  if [[ -f "${JAMO_SUBST_TXT}" ]]; then
    log "CPT txt exists: ${JAMO_SUBST_TXT}"
    return 0
  fi
  log "Converting jsonl to CPT txt: ${JAMO_SUBST_TXT}"
  "${PYTHON}" - << PY
import json
from pathlib import Path
inp = Path("${JAMO_SUBST_JSONL}")
out = Path("${JAMO_SUBST_TXT}")
out.parent.mkdir(parents=True, exist_ok=True)
n = 0
with inp.open("r", encoding="utf-8") as f, out.open("w", encoding="utf-8") as w:
    for line in f:
        if not line.strip():
            continue
        ex = json.loads(line)
        parts = [ex.get("instruction", ""), ex.get("input", ""), ex.get("output", "")]
        text = "\\n".join([str(p) for p in parts if p])
        if text.strip():
            w.write(text.strip().replace("\\r\\n", "\\n") + "\\n")
            n += 1
print("saved", out, "lines", n)
PY
}

verify_extended_checkpoint() {
  local ckpt="$1"
  log "Verify extended tokenizer/model vocab: ${ckpt}"
  "${PYTHON}" - << PY
from transformers import AutoTokenizer, AutoModelForCausalLM
p = "${ckpt}"
tok = AutoTokenizer.from_pretrained(p)
m = AutoModelForCausalLM.from_pretrained(p)
print("tokenizer len:", len(tok))
print("model vocab:", m.get_input_embeddings().weight.shape[0])
print("contains ㄱ:", "ㄱ" in tok.get_vocab())
assert len(tok) == m.get_input_embeddings().weight.shape[0], "tokenizer/model vocab mismatch"
assert "ㄱ" in tok.get_vocab(), "jamo token ㄱ missing"
PY
}

train_cpt_one() {
  local lr="$1"
  local lr_tag
  lr_tag="$(safe_lr_tag "${lr}")"
  local out_dir="${OUTPUT_ROOT}/cpt_${MODEL_TAG}_${JAMO_RATIO_TAG}_lr${lr_tag}_steps${CPT_MAX_STEPS}_seed${SEED}"

  if [[ "${RUN_CPT}" == "1" && ! -f "${out_dir}/config.json" ]]; then
    log "CPT train lr=${lr} out=${out_dir}"
    CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" torchrun --master_port "${MASTER_PORT}" --nproc_per_node="${NPROC}" train.py \
      --base_model "${BASE_MODEL}" \
      --jamo_tokenizer_path "${JAMO_TOKENIZER_PATH}" \
      --train_text "${JAMO_SUBST_TXT}" \
      --eval_text "${DATA_DIR}/instruction_valid.txt" \
      --block_size "${CPT_BLOCK_SIZE}" \
      --overwrite_cache \
      --per_device_train_batch_size "${CPT_PER_DEVICE_BATCH}" \
      --per_device_eval_batch_size 1 \
      --gradient_accumulation_steps "${CPT_GRAD_ACCUM}" \
      --max_steps "${CPT_MAX_STEPS}" \
      --bf16 \
      --learning_rate "${lr}" \
      --weight_decay "${CPT_WEIGHT_DECAY}" \
      --warmup_ratio "${CPT_WARMUP_RATIO}" \
      --cho_loss_weight 0.0 \
      --jung_loss_weight 0.0 \
      --jong_loss_weight 0.0 \
      --add_jamo_tokens \
      --logging_steps "${CPT_LOGGING_STEPS}" \
      --eval_steps "${CPT_EVAL_STEPS}" \
      --save_steps "${CPT_SAVE_STEPS}" \
      --save_total_limit "${CPT_SAVE_TOTAL_LIMIT}" \
      --output_dir "${out_dir}" \
      --run_name "cpt_${MODEL_TAG}_${JAMO_RATIO_TAG}_lr${lr_tag}"
  else
    log "CPT exists or RUN_CPT=0, skip: ${out_dir}"
  fi

  verify_extended_checkpoint "${out_dir}"
  echo "${out_dir}"
}

train_sft_one() {
  local cpt_dir="$1"
  local lr="$2"
  local lr_tag
  lr_tag="$(safe_lr_tag "${lr}")"
  local out_dir="${OUTPUT_ROOT}/sft_${MODEL_TAG}_${JAMO_RATIO_TAG}_cptlr${lr_tag}_sftlr$(safe_lr_tag "${SFT_LR}")_steps${SFT_MAX_STEPS}_seed${SEED}"

  if [[ "${RUN_SFT}" == "1" && ! -f "${out_dir}/config.json" ]]; then
    log "SFT train from ${cpt_dir}"
    CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" torchrun --master_port "${MASTER_PORT}" --nproc_per_node="${NPROC}" train_sft_jamo_residual_ablation.py \
      --base_model "${cpt_dir}" \
      --jamo_tokenizer_path "${JAMO_TOKENIZER_PATH}" \
      --train_jsonl "${BASE_TRAIN_JSONL}" \
      --eval_jsonl "${EVAL_JSONL}" \
      --template_name "${TEMPLATE_NAME}" \
      --max_length "${MAX_LENGTH}" \
      --overwrite_cache \
      --per_device_train_batch_size "${SFT_PER_DEVICE_BATCH}" \
      --per_device_eval_batch_size 1 \
      --gradient_accumulation_steps "${SFT_GRAD_ACCUM}" \
      --max_steps "${SFT_MAX_STEPS}" \
      --bf16 \
      --learning_rate "${SFT_LR}" \
      --seed "${SEED}" \
      --cho_loss_weight 0.0 \
      --jung_loss_weight 0.0 \
      --jong_loss_weight 0.0 \
      --residual_alpha_init 0.0 \
      --logging_steps "${SFT_LOGGING_STEPS}" \
      --save_steps "${SFT_SAVE_STEPS}" \
      --save_total_limit "${SFT_SAVE_TOTAL_LIMIT}" \
      --output_dir "${out_dir}" \
      --run_name "sft_${MODEL_TAG}_${JAMO_RATIO_TAG}_cptlr${lr_tag}"
  else
    log "SFT exists or RUN_SFT=0, skip: ${out_dir}"
  fi
  echo "${out_dir}"
}

task_exists() {
  local task="$1"
  lm-eval ls tasks 2>/dev/null | grep -E "^\|${task}[[:space:]\|]" >/dev/null 2>&1
}

run_lmeval_one() {
  local model_dir="$1"
  local eval_dir="$2"
  local task="$3"
  [[ "${RUN_LMEVAL}" == "1" ]] || return 0
  mkdir -p "${eval_dir}/benchmarks"
  [[ ! -f "${eval_dir}/benchmarks/${task}.json" ]] || { echo "[SKIP] ${task} exists"; return 0; }
  if task_exists "${task}"; then
    log "lm-eval task=${task} model=${model_dir}"
    CUDA_VISIBLE_DEVICES="${EVAL_GPU}" lm-eval run \
      --model hf \
      --model_args "pretrained=${model_dir},trust_remote_code=True" \
      --tasks "${task}" \
      --device cuda:0 \
      --batch_size "${LM_EVAL_BATCH_SIZE}" \
      --output_path "${eval_dir}/benchmarks/${task}.json" \
      || echo "[WARN] lm-eval failed task=${task} model=${model_dir}" >&2
  else
    echo "[WARN] lm-eval task not found: ${task}" >&2
  fi
}

run_ppl_one() {
  local model_dir="$1"
  local eval_text="$2"
  local out_json="$3"
  local tag="$4"
  [[ "${RUN_PPL}" == "1" ]] || return 0
  mkdir -p "$(dirname "${out_json}")"
  [[ -f "${eval_text}" ]] || { echo "[WARN] missing eval text ${tag}: ${eval_text}" >&2; return 0; }
  [[ ! -f "${out_json}" ]] || { echo "[SKIP] exists: ${out_json}"; return 0; }

  log "PPL ${tag} model=${model_dir}"
  CUDA_VISIBLE_DEVICES="${EVAL_GPU}" "${PYTHON}" eval_ppl.py \
    --model_name_or_path "${model_dir}" \
    --eval_text "${eval_text}" \
    --block_size "${MAX_LENGTH}" \
    --batch_size "${PPL_BATCH_SIZE}" \
    --dtype bf16 \
    --output_json "${out_json}"
}

summarize_one() {
  local eval_dir="$1"
  "${PYTHON}" - << PY
import json
from pathlib import Path
eval_dir = Path("${eval_dir}")
ppl_dir = eval_dir / "ppl"
files = {
    "valid": "instruction_valid_ppl.json",
    "clean": "instruction_test_ppl.json",
    "noisy_jong": "test_noisy_jong.json",
    "noisy_pron": "test_noisy_pronunciation.json",
    "noisy_spacing": "test_noisy_spacing.json",
    "noisy_mixed": "test_noisy_mixed.json",
}
def load_ppl(fn):
    p = ppl_dir / fn
    if not p.exists():
        return None
    return json.load(open(p)).get("ppl")
row = {k: load_ppl(v) for k, v in files.items()}
clean = row.get("clean")
lines = ["| metric | ppl | degradation_vs_clean |", "|:--|--:|--:|"]
for k in ["valid", "clean", "noisy_jong", "noisy_pron", "noisy_spacing", "noisy_mixed"]:
    v = row.get(k)
    if v is None:
        lines.append(f"| {k} |  |  |")
    elif clean and k.startswith("noisy"):
        deg = (v / clean - 1.0) * 100.0
        lines.append(f"| {k} | {v:.4f} | {deg:.2f}% |")
    else:
        lines.append(f"| {k} | {v:.4f} |  |")
out = eval_dir / "ppl_summary.md"
out.write_text("\\n".join(lines) + "\\n", encoding="utf-8")
print("\\n".join(lines))
print("saved:", out)
PY
}

eval_model() {
  local model_dir="$1"
  local lr="$2"
  [[ "${RUN_EVAL}" == "1" ]] || return 0
  local lr_tag
  lr_tag="$(safe_lr_tag "${lr}")"
  local eval_dir="${EVAL_ROOT}/sft_${MODEL_TAG}_${JAMO_RATIO_TAG}_cptlr${lr_tag}_sftlr$(safe_lr_tag "${SFT_LR}")_steps${SFT_MAX_STEPS}_seed${SEED}"
  mkdir -p "${eval_dir}"

  run_lmeval_one "${model_dir}" "${eval_dir}" "kmmlu"
  run_lmeval_one "${model_dir}" "${eval_dir}" "kobest"
  run_ppl_one "${model_dir}" "${DATA_DIR}/instruction_valid.txt" "${eval_dir}/ppl/instruction_valid_ppl.json" "valid"
  run_ppl_one "${model_dir}" "${DATA_DIR}/instruction_test.txt" "${eval_dir}/ppl/instruction_test_ppl.json" "test_clean"
  run_ppl_one "${model_dir}" "${DATA_DIR}/instruction_test.noisy_jong.txt" "${eval_dir}/ppl/test_noisy_jong.json" "noisy_jong"
  run_ppl_one "${model_dir}" "${DATA_DIR}/instruction_test.noisy_pronunciation.txt" "${eval_dir}/ppl/test_noisy_pronunciation.json" "noisy_pronunciation"
  run_ppl_one "${model_dir}" "${DATA_DIR}/instruction_test.noisy_spacing.txt" "${eval_dir}/ppl/test_noisy_spacing.json" "noisy_spacing"
  run_ppl_one "${model_dir}" "${DATA_DIR}/instruction_test.noisy_mixed.txt" "${eval_dir}/ppl/test_noisy_mixed.json" "noisy_mixed"
  summarize_one "${eval_dir}"
}

check_inputs() {
  log "Checking inputs"
  require_file "train.py"
  require_file "train_sft_jamo_residual_ablation.py"
  require_file "eval_ppl.py"
  require_file "make_jamo_substitution_instruction_data.py"
  require_file "${BASE_TRAIN_JSONL}"
  require_file "${EVAL_JSONL}"
  require_file "${DATA_DIR}/instruction_valid.txt"
  require_file "${DATA_DIR}/instruction_test.txt"
  require_file "${JAMO_TOKENIZER_PATH}"
  require_cmd "torchrun"
  require_cmd "${PYTHON}"
  mkdir -p "${OUTPUT_ROOT}" "${EVAL_ROOT}"

  echo "MODEL_FAMILY=${MODEL_FAMILY}"
  echo "BASE_MODEL=${BASE_MODEL}"
  echo "MODEL_TAG=${MODEL_TAG}"
  echo "TEMPLATE_NAME=${TEMPLATE_NAME}"
  echo "JAMO_TOKENIZER_PATH=${JAMO_TOKENIZER_PATH}"
  echo "JAMO_RATIO=${JAMO_RATIO}"
  echo "CPT_LRS=${CPT_LRS}"
  echo "CPT_MAX_STEPS=${CPT_MAX_STEPS}"
  echo "SFT_LR=${SFT_LR}"
  echo "SFT_MAX_STEPS=${SFT_MAX_STEPS}"
  echo "TRAIN_GPUS=${TRAIN_GPUS}"
  echo "MASTER_PORT=${MASTER_PORT}"
  echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
  echo "EVAL_ROOT=${EVAL_ROOT}"
}

check_inputs
make_jamo_subst_jsonl_if_needed
make_cpt_txt_if_needed

for cpt_lr in ${CPT_LRS}; do
  cpt_dir="$(train_cpt_one "${cpt_lr}" | tail -n 1)"
  sft_dir="$(train_sft_one "${cpt_dir}" "${cpt_lr}" | tail -n 1)"
  eval_model "${sft_dir}" "${cpt_lr}"
done

log "Done"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "EVAL_ROOT=${EVAL_ROOT}"
