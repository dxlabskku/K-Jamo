# sft_dataset.py
"""
SFT dataset for Korean instruction tuning with optional Jamo auxiliary labels.

Input JSONL format:
{
  "instruction": "...",
  "input": "...",
  "output": "...",
  "source": "...",
  ...
}

This dataset creates:
  input_ids
  attention_mask
  labels              # -100 for prompt tokens, target token ids for answer tokens
  cho_labels          # -100 for prompt tokens, jamo token id for answer tokens
  jung_labels
  jong_labels

Important:
  - LM loss is computed ONLY on the answer part.
  - Prompt tokens are masked with -100.
  - Jamo labels are also masked on prompt tokens.
  - Uses the existing base tokenizer for input_ids.
  - Uses jamo tokenizer only to map decoded base tokens -> CHO/JUNG/JONG label ids.

Recommended prompt template:

<|system|>
당신은 한국어로 정확하고 도움이 되는 답변을 제공하는 AI 어시스턴트입니다.
<|user|>
{instruction}

{input}
<|assistant|>
{output}

Why this format:
  - explicit role boundaries
  - Korean system instruction
  - stable answer boundary for label masking
  - compatible with causal LM SFT

Example usage:

from transformers import AutoTokenizer
from sft_dataset import SFTJsonlDataset, SFTDataCollator

base_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/polyglot-ko-1.3b")
jamo_tokenizer = AutoTokenizer.from_pretrained("./polyglot-ko-1.3b-jamo-tokenizer")

dataset = SFTJsonlDataset(
    jsonl_path="/path/instruction_train.balanced_250k.jsonl",
    base_tokenizer=base_tokenizer,
    jamo_tokenizer=jamo_tokenizer,
    max_length=1024,
)

collator = SFTDataCollator(
    pad_token_id=base_tokenizer.pad_token_id or base_tokenizer.eos_token_id
)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


IGNORE_INDEX = -100


SYSTEM_PROMPT = "당신은 한국어로 정확하고 도움이 되는 답변을 제공하는 AI 어시스턴트입니다."


def normalize_text(x: Any) -> str:
    if x is None:
        return ""
    x = str(x)
    x = x.replace("\x00", "")
    x = x.replace("\u200b", "")
    x = x.replace("\ufeff", "")
    x = re.sub(r"[ \t]+", " ", x)
    x = re.sub(r"\n{3,}", "\n\n", x)
    return x.strip()


def build_prompt(
    instruction: str,
    input_text: str = "",
    system_prompt: str = SYSTEM_PROMPT,
    template_name: str = "simple",
) -> str:
    """
    Build prompt without answer.

    Keep this EXACTLY aligned with build_full_text().
    The answer starts immediately after this prompt.
    """
    instruction = normalize_text(instruction)
    input_text = normalize_text(input_text)
    system_prompt = normalize_text(system_prompt)

    if input_text:
        user_content = f"{instruction}\n\n{input_text}"
    else:
        user_content = instruction

    if template_name == "qwen":
        prompt = (
            f"<|im_start|>system\n"
            f"{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n"
            f"{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    else:
        prompt = (
            f"<|system|>\n"
            f"{system_prompt}\n"
            f"<|user|>\n"
            f"{user_content}\n"
            f"<|assistant|>\n"
        )
    return prompt


def build_full_text(
    instruction: str,
    input_text: str,
    output: str,
    system_prompt: str = SYSTEM_PROMPT,
    add_eos_text: str = "",
    template_name: str = "simple",
) -> Tuple[str, str]:
    """
    Returns:
      prompt_text, full_text

    full_text = prompt_text + output + optional eos text.
    """
    prompt_text = build_prompt(
        instruction,
        input_text,
        system_prompt,
        template_name=template_name,
    )
    output = normalize_text(output)

    full_text = prompt_text + output
    if template_name == "qwen":
        full_text += "<|im_end|>"
    if add_eos_text:
        full_text += add_eos_text

    return prompt_text, full_text


# Hangul decomposition tables.
CHO_LIST = [
    "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ",
    "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"
]

JUNG_LIST = [
    "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ",
    "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ"
]

JONG_LIST = [
    "_", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ",
    "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ",
    "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"
]


def decompose_hangul_char(ch: str) -> Optional[Tuple[str, str, str]]:
    code = ord(ch)
    base = 0xAC00
    last = 0xD7A3

    if code < base or code > last:
        return None

    sidx = code - base
    cho = sidx // (21 * 28)
    jung = (sidx % (21 * 28)) // 28
    jong = sidx % 28

    return CHO_LIST[cho], JUNG_LIST[jung], JONG_LIST[jong]


def token_to_jamo_strings(token_text: str) -> Tuple[str, str, str]:
    """
    Convert decoded token text into CHO/JUNG/JONG strings.

    Non-Hangul chars are ignored.
    Spaces/punctuation do not contribute.
    If no Hangul exists, returns underscores.
    """
    chos: List[str] = []
    Jungs: List[str] = []
    jongs: List[str] = []

    for ch in token_text:
        dec = decompose_hangul_char(ch)
        if dec is None:
            continue
        c, j, t = dec
        chos.append(c)
        Jungs.append(j)
        jongs.append(t)

    if not chos:
        return "_", "_", "_"

    return "".join(chos), "".join(Jungs), "".join(jongs)


def make_jamo_label_token(kind: str, value: str) -> str:
    assert kind in ["CHO", "JUNG", "JONG"]
    return f"<{kind}:{value}>"


def jamo_label_id(
    jamo_tokenizer,
    kind: str,
    value: str,
    unk_fallback: int = IGNORE_INDEX,
) -> int:
    tok = make_jamo_label_token(kind, value)
    tok_id = jamo_tokenizer.convert_tokens_to_ids(tok)

    # HF tokenizers often return unk_token_id or None when not found.
    if tok_id is None:
        return unk_fallback

    unk_id = getattr(jamo_tokenizer, "unk_token_id", None)
    if unk_id is not None and tok_id == unk_id and tok != getattr(jamo_tokenizer, "unk_token", None):
        return unk_fallback

    return int(tok_id)


def build_jamo_labels_for_ids(
    input_ids: List[int],
    base_tokenizer,
    jamo_tokenizer,
) -> Tuple[List[int], List[int], List[int]]:
    """
    For each base token id, decode token text, decompose Hangul, and map to
    jamo tokenizer label ids.
    """
    cho_labels: List[int] = []
    jung_labels: List[int] = []
    jong_labels: List[int] = []

    for tid in input_ids:
        token_text = base_tokenizer.decode([tid], clean_up_tokenization_spaces=False)
        cho, jung, jong = token_to_jamo_strings(token_text)

        cho_labels.append(jamo_label_id(jamo_tokenizer, "CHO", cho))
        jung_labels.append(jamo_label_id(jamo_tokenizer, "JUNG", jung))
        jong_labels.append(jamo_label_id(jamo_tokenizer, "JONG", jong))

    return cho_labels, jung_labels, jong_labels


class SFTJsonlDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str | Path,
        base_tokenizer,
        jamo_tokenizer=None,
        max_length: int = 1024,
        system_prompt: str = SYSTEM_PROMPT,
        add_eos: bool = True,
        max_examples: Optional[int] = None,
        skip_long_prompts: bool = True,
        cache_path: Optional[str | Path] = None,
        overwrite_cache: bool = False,
        template_name: str = "simple",
    ):
        self.jsonl_path = Path(jsonl_path)
        self.base_tokenizer = base_tokenizer
        self.jamo_tokenizer = jamo_tokenizer
        self.max_length = max_length
        self.system_prompt = system_prompt
        self.add_eos = add_eos
        self.max_examples = max_examples
        self.skip_long_prompts = skip_long_prompts
        self.template_name = template_name

        if cache_path is not None:
            self.cache_path = Path(cache_path)
        else:
            safe_name = self.jsonl_path.stem.replace(".", "_")
            self.cache_path = self.jsonl_path.parent / f"{safe_name}.sft.max{max_length}.pt"

        if self.cache_path.exists() and not overwrite_cache:
            print(f"Loading SFT dataset cache: {self.cache_path}")
            self.examples = torch.load(self.cache_path, map_location="cpu")
        else:
            self.examples = self._build_examples()
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.examples, self.cache_path)
            print(f"Saved SFT dataset cache: {self.cache_path}")

        print(f"SFT examples: {len(self.examples)}")

    def _read_jsonl(self):
        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if self.max_examples is not None and i >= self.max_examples:
                    break
                line = line.strip()
                if not line:
                    continue
                yield i, json.loads(line)

    def _build_examples(self) -> List[Dict[str, torch.Tensor]]:
        examples: List[Dict[str, torch.Tensor]] = []

        eos_text = ""
        if self.add_eos and self.base_tokenizer.eos_token is not None:
            eos_text = self.base_tokenizer.eos_token

        skipped_empty = 0
        skipped_long_prompt = 0
        skipped_no_answer_tokens = 0

        for idx, ex in tqdm(self._read_jsonl(), desc=f"Building SFT dataset {self.jsonl_path.name}"):
            instruction = normalize_text(ex.get("instruction", ""))
            input_text = normalize_text(ex.get("input", ""))
            output = normalize_text(ex.get("output", ""))

            if not output or not (instruction or input_text):
                skipped_empty += 1
                continue

            prompt_text, full_text = build_full_text(
                instruction=instruction,
                input_text=input_text,
                output=output,
                system_prompt=self.system_prompt,
                add_eos_text=eos_text,
                template_name=self.template_name,
            )

            prompt_ids = self.base_tokenizer.encode(
                prompt_text,
                add_special_tokens=False,
            )

            full_ids = self.base_tokenizer.encode(
                full_text,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_length,
            )

            prompt_len = len(prompt_ids)

            # If prompt itself is too long, there is no answer supervision left.
            if prompt_len >= self.max_length:
                if self.skip_long_prompts:
                    skipped_long_prompt += 1
                    continue
                prompt_len = self.max_length - 1

            labels = list(full_ids)
            labels[:prompt_len] = [IGNORE_INDEX] * min(prompt_len, len(labels))

            # Must have at least one supervised answer token.
            if all(x == IGNORE_INDEX for x in labels):
                skipped_no_answer_tokens += 1
                continue

            attention_mask = [1] * len(full_ids)

            item: Dict[str, Any] = {
                "input_ids": torch.tensor(full_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }

            if self.jamo_tokenizer is not None:
                cho, jung, jong = build_jamo_labels_for_ids(
                    input_ids=full_ids,
                    base_tokenizer=self.base_tokenizer,
                    jamo_tokenizer=self.jamo_tokenizer,
                )

                # Mask prompt region for auxiliary losses too.
                mask_len = min(prompt_len, len(full_ids))
                cho[:mask_len] = [IGNORE_INDEX] * mask_len
                jung[:mask_len] = [IGNORE_INDEX] * mask_len
                jong[:mask_len] = [IGNORE_INDEX] * mask_len

                item["cho_labels"] = torch.tensor(cho, dtype=torch.long)
                item["jung_labels"] = torch.tensor(jung, dtype=torch.long)
                item["jong_labels"] = torch.tensor(jong, dtype=torch.long)

            examples.append(item)

        print("Skipped empty:", skipped_empty)
        print("Skipped long prompt:", skipped_long_prompt)
        print("Skipped no answer tokens:", skipped_no_answer_tokens)

        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int):
        return self.examples[idx]

class PairedSFTJsonlDataset(Dataset):
    """
    Clean-noisy paired SFT dataset for consistency training.

    Expected JSONL fields:
      clean_instruction
      clean_input
      noisy_instruction
      noisy_input
      output
      noise_type

    Returns:
      clean_input_ids, clean_attention_mask, clean_labels
      noisy_input_ids, noisy_attention_mask, noisy_labels
      clean_cho_labels, clean_jung_labels, clean_jong_labels
      noisy_cho_labels, noisy_jung_labels, noisy_jong_labels
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        base_tokenizer,
        jamo_tokenizer=None,
        max_length: int = 1024,
        system_prompt: str = SYSTEM_PROMPT,
        add_eos: bool = True,
        max_examples: Optional[int] = None,
        skip_long_prompts: bool = True,
        cache_path: Optional[str | Path] = None,
        overwrite_cache: bool = False,
        template_name: str = "simple",
    ):
        self.jsonl_path = Path(jsonl_path)
        self.base_tokenizer = base_tokenizer
        self.jamo_tokenizer = jamo_tokenizer
        self.max_length = max_length
        self.system_prompt = system_prompt
        self.add_eos = add_eos
        self.max_examples = max_examples
        self.skip_long_prompts = skip_long_prompts
        self.template_name = template_name

        if cache_path is not None:
            self.cache_path = Path(cache_path)
        else:
            safe_name = self.jsonl_path.stem.replace(".", "_")
            self.cache_path = self.jsonl_path.parent / f"{safe_name}.paired_sft.{template_name}.max{max_length}.pt"

        if self.cache_path.exists() and not overwrite_cache:
            print(f"Loading Paired SFT dataset cache: {self.cache_path}")
            self.examples = torch.load(self.cache_path, map_location="cpu")
        else:
            self.examples = self._build_examples()
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.examples, self.cache_path)
            print(f"Saved Paired SFT dataset cache: {self.cache_path}")

        print(f"Paired SFT examples: {len(self.examples)}")

    def _read_jsonl(self):
        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if self.max_examples is not None and i >= self.max_examples:
                    break
                line = line.strip()
                if not line:
                    continue
                yield i, json.loads(line)

    def _encode_one(
        self,
        instruction: str,
        input_text: str,
        output: str,
        eos_text: str,
    ) -> Optional[Dict[str, torch.Tensor]]:
        instruction = normalize_text(instruction)
        input_text = normalize_text(input_text)
        output = normalize_text(output)

        if not output or not (instruction or input_text):
            return None

        prompt_text, full_text = build_full_text(
            instruction=instruction,
            input_text=input_text,
            output=output,
            system_prompt=self.system_prompt,
            add_eos_text=eos_text,
            template_name=self.template_name,
        )

        prompt_ids = self.base_tokenizer.encode(
            prompt_text,
            add_special_tokens=False,
        )

        full_ids = self.base_tokenizer.encode(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )

        prompt_len = len(prompt_ids)

        if prompt_len >= self.max_length:
            if self.skip_long_prompts:
                return None
            prompt_len = self.max_length - 1

        labels = list(full_ids)
        labels[:prompt_len] = [IGNORE_INDEX] * min(prompt_len, len(labels))

        if all(x == IGNORE_INDEX for x in labels):
            return None

        item: Dict[str, torch.Tensor] = {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "attention_mask": torch.tensor([1] * len(full_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

        if self.jamo_tokenizer is not None:
            cho, jung, jong = build_jamo_labels_for_ids(
                input_ids=full_ids,
                base_tokenizer=self.base_tokenizer,
                jamo_tokenizer=self.jamo_tokenizer,
            )

            mask_len = min(prompt_len, len(full_ids))
            cho[:mask_len] = [IGNORE_INDEX] * mask_len
            jung[:mask_len] = [IGNORE_INDEX] * mask_len
            jong[:mask_len] = [IGNORE_INDEX] * mask_len

            item["cho_labels"] = torch.tensor(cho, dtype=torch.long)
            item["jung_labels"] = torch.tensor(jung, dtype=torch.long)
            item["jong_labels"] = torch.tensor(jong, dtype=torch.long)

        return item

    def _build_examples(self) -> List[Dict[str, torch.Tensor]]:
        examples: List[Dict[str, torch.Tensor]] = []

        eos_text = ""
        if self.add_eos and self.base_tokenizer.eos_token is not None:
            eos_text = self.base_tokenizer.eos_token

        skipped_empty = 0
        skipped_missing_pair = 0
        skipped_encoding = 0

        for idx, ex in tqdm(self._read_jsonl(), desc=f"Building Paired SFT dataset {self.jsonl_path.name}"):
            output = normalize_text(ex.get("output", ""))

            clean_instruction = normalize_text(ex.get("clean_instruction", ex.get("instruction", "")))
            clean_input = normalize_text(ex.get("clean_input", ex.get("input", "")))
            noisy_instruction = normalize_text(ex.get("noisy_instruction", ""))
            noisy_input = normalize_text(ex.get("noisy_input", ""))

            if not output or not (clean_instruction or clean_input):
                skipped_empty += 1
                continue

            if not (noisy_instruction or noisy_input):
                skipped_missing_pair += 1
                continue

            clean_item = self._encode_one(
                instruction=clean_instruction,
                input_text=clean_input,
                output=output,
                eos_text=eos_text,
            )
            noisy_item = self._encode_one(
                instruction=noisy_instruction,
                input_text=noisy_input,
                output=output,
                eos_text=eos_text,
            )

            if clean_item is None or noisy_item is None:
                skipped_encoding += 1
                continue

            pair: Dict[str, Any] = {}

            for k, v in clean_item.items():
                pair[f"clean_{k}"] = v

            for k, v in noisy_item.items():
                pair[f"noisy_{k}"] = v

            pair["noise_type"] = ex.get("noise_type", "unknown")
            examples.append(pair)

        print("Skipped empty:", skipped_empty)
        print("Skipped missing pair:", skipped_missing_pair)
        print("Skipped encoding:", skipped_encoding)

        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int):
        return self.examples[idx]

class SFTDataCollator:
    def __init__(
        self,
        pad_token_id: int,
        label_pad_token_id: int = IGNORE_INDEX,
        pad_to_multiple_of: Optional[int] = 8,
    ):
        self.pad_token_id = int(pad_token_id)
        self.label_pad_token_id = int(label_pad_token_id)
        self.pad_to_multiple_of = pad_to_multiple_of

    def _pad_1d(self, tensors: List[torch.Tensor], pad_value: int) -> torch.Tensor:
        max_len = max(x.size(0) for x in tensors)

        if self.pad_to_multiple_of is not None:
            m = self.pad_to_multiple_of
            if max_len % m != 0:
                max_len = ((max_len // m) + 1) * m

        out = torch.full(
            (len(tensors), max_len),
            fill_value=pad_value,
            dtype=tensors[0].dtype,
        )

        for i, x in enumerate(tensors):
            out[i, : x.size(0)] = x

        return out

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batch = {
            "input_ids": self._pad_1d([f["input_ids"] for f in features], self.pad_token_id),
            "attention_mask": self._pad_1d([f["attention_mask"] for f in features], 0),
            "labels": self._pad_1d([f["labels"] for f in features], self.label_pad_token_id),
        }

        if "cho_labels" in features[0]:
            batch["cho_labels"] = self._pad_1d(
                [f["cho_labels"] for f in features],
                self.label_pad_token_id,
            )
            batch["jung_labels"] = self._pad_1d(
                [f["jung_labels"] for f in features],
                self.label_pad_token_id,
            )
            batch["jong_labels"] = self._pad_1d(
                [f["jong_labels"] for f in features],
                self.label_pad_token_id,
            )

        return batch

class PairedSFTDataCollator:
    def __init__(
        self,
        pad_token_id: int,
        label_pad_token_id: int = IGNORE_INDEX,
        pad_to_multiple_of: Optional[int] = 8,
    ):
        self.base_collator = SFTDataCollator(
            pad_token_id=pad_token_id,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=pad_to_multiple_of,
        )

    def _strip_prefix(self, features: List[Dict[str, torch.Tensor]], prefix: str) -> List[Dict[str, torch.Tensor]]:
        out = []
        plen = len(prefix)

        for f in features:
            item = {}
            for k, v in f.items():
                if k.startswith(prefix):
                    item[k[plen:]] = v
            out.append(item)

        return out

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        clean_features = self._strip_prefix(features, "clean_")
        noisy_features = self._strip_prefix(features, "noisy_")

        clean_batch = self.base_collator(clean_features)
        noisy_batch = self.base_collator(noisy_features)

        batch = {}

        for k, v in clean_batch.items():
            batch[f"clean_{k}"] = v

        for k, v in noisy_batch.items():
            batch[f"noisy_{k}"] = v

        return batch

def debug_print_example(
    jsonl_path: str | Path,
    base_tokenizer,
    max_length: int = 1024,
    index: int = 0,
):
    path = Path(jsonl_path)
    with path.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    ex = rows[index]
    prompt, full = build_full_text(ex["instruction"], ex.get("input", ""), ex["output"])
    prompt_ids = base_tokenizer.encode(prompt, add_special_tokens=False)
    full_ids = base_tokenizer.encode(full, add_special_tokens=False, truncation=True, max_length=max_length)

    print("=" * 80)
    print("PROMPT:")
    print(prompt)
    print("-" * 80)
    print("OUTPUT:")
    print(ex["output"])
    print("-" * 80)
    print("prompt tokens:", len(prompt_ids))
    print("full tokens:", len(full_ids))
    print("supervised tokens:", max(0, len(full_ids) - len(prompt_ids)))
    print("=" * 80)
