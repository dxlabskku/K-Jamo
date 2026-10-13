#!/usr/bin/env bash
set -euo pipefail

cd /home/irteam/data-vol1/minsu/kolm
source rebuttal_analysis/stronger_bench/checkpoint_resolve/qwen3b_model_paths.sh

TOK_SRC="$MODEL_NOISY"
OUT_ROOT="/home/irteam/data-vol1/minsu/kolm/rebuttal_analysis/stronger_bench/checkpoint_resolve/bpe_eval_ready"
mkdir -p "$OUT_ROOT"

make_one () {
  local NAME="$1"
  local CKPT="$2"
  local OUT="$OUT_ROOT/$NAME"

  rm -rf "$OUT"
  mkdir -p "$OUT"

  echo "CREATE $OUT"
  echo "  CKPT=$CKPT"
  echo "  TOK_SRC=$TOK_SRC"

  # Model/config files from BPE checkpoint.
  for f in \
    config.json \
    generation_config.json \
    pytorch_model.bin.index.json \
    pytorch_model-00001-of-00002.bin \
    pytorch_model-00002-of-00002.bin
  do
    if [ -f "$CKPT/$f" ]; then
      ln -s "$CKPT/$f" "$OUT/$f"
    else
      echo "MISSING model file: $CKPT/$f" >&2
      exit 1
    fi
  done

  # Tokenizer files from Noisy model.
  for f in \
    added_tokens.json \
    chat_template.jinja \
    merges.txt \
    special_tokens_map.json \
    tokenizer.json \
    tokenizer_config.json \
    vocab.json
  do
    if [ -f "$TOK_SRC/$f" ]; then
      ln -s "$TOK_SRC/$f" "$OUT/$f"
    else
      echo "WARN missing tokenizer file: $TOK_SRC/$f"
    fi
  done

  # Optional tokenizer subdir, if any code expects it.
  if [ -d "$TOK_SRC/jamo_tokenizer" ]; then
    ln -s "$TOK_SRC/jamo_tokenizer" "$OUT/jamo_tokenizer"
  fi

  ls -lh "$OUT"
}

make_one "bpe005" "$MODEL_BPE005"
make_one "bpe010" "$MODEL_BPE010"

echo "DONE $OUT_ROOT"
