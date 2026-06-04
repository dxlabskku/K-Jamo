# dataset.py
"""
Dataset utilities for Korean Jamo Auxiliary continued pretraining.

Input:
    plain UTF-8 text file, one utterance per line, blank lines allowed.

Output sample:
    {
        "input_ids": [...],
        "labels": [...],
        "cho_labels": [...],
        "jung_labels": [...],
        "jong_labels": [...],
    }

Usage example:
    from transformers import AutoTokenizer
    from dataset import JamoPackedTextDataset

    base_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/polyglot-ko-1.3b")
    jamo_tokenizer = AutoTokenizer.from_pretrained("./polyglot-ko-1.3b-jamo-tokenizer")

    dataset = JamoPackedTextDataset(
        text_path="/path/to/aihub_multisession_plain.cleaned.txt",
        base_tokenizer=base_tokenizer,
        jamo_tokenizer=jamo_tokenizer,
        block_size=2048,
        max_lines=None,
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from utils import make_jamo_labels


class JamoPackedTextDataset(Dataset):
    """
    Plain text -> tokenized packed blocks -> jamo auxiliary labels.

    This dataset keeps the original Polyglot tokenization unchanged.
    Jamo labels are created separately and aligned 1:1 with input_ids.

    Important:
        - input_ids are Polyglot token IDs.
        - labels are same as input_ids; causal shifting happens inside model.forward().
        - cho/jung/jong labels are jamo-token IDs from the augmented tokenizer.
        - Non-Hangul tokens get -100 labels and are ignored for jamo loss.
    """

    def __init__(
        self,
        text_path: str | Path,
        base_tokenizer: Any,
        jamo_tokenizer: Any,
        block_size: int = 2048,
        add_eos_between_lines: bool = True,
        min_tokens_per_block: int = 16,
        max_lines: Optional[int] = None,
        cache_path: Optional[str | Path] = None,
        overwrite_cache: bool = False,
    ):
        self.text_path = Path(text_path)
        self.base_tokenizer = base_tokenizer
        self.jamo_tokenizer = jamo_tokenizer
        self.block_size = block_size
        self.add_eos_between_lines = add_eos_between_lines
        self.min_tokens_per_block = min_tokens_per_block
        self.max_lines = max_lines

        if cache_path is not None:
            self.cache_path = Path(cache_path)
        else:
            suffix = f".block{block_size}"
            if max_lines is not None:
                suffix += f".max{max_lines}"
            self.cache_path = self.text_path.with_suffix(self.text_path.suffix + suffix + ".pt")

        if self.cache_path.exists() and not overwrite_cache:
            obj = torch.load(self.cache_path, map_location="cpu")
            self.examples = obj["examples"]
            self.meta = obj.get("meta", {})
        else:
            self.examples = self._build_examples()
            self.meta = {
                "text_path": str(self.text_path),
                "block_size": self.block_size,
                "num_examples": len(self.examples),
                "add_eos_between_lines": self.add_eos_between_lines,
                "min_tokens_per_block": self.min_tokens_per_block,
                "max_lines": self.max_lines,
            }
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"examples": self.examples, "meta": self.meta}, self.cache_path)

    def _iter_lines(self):
        with self.text_path.open("r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                if self.max_lines is not None and i >= self.max_lines:
                    break

                line = line.strip()
                if not line:
                    continue

                yield line

    def _build_examples(self) -> List[Dict[str, List[int]]]:
        eos_id = self.base_tokenizer.eos_token_id
        if eos_id is None:
            raise ValueError("base_tokenizer.eos_token_id is None. Polyglot tokenizer should have eos_token_id.")

        examples: List[Dict[str, List[int]]] = []
        buffer_ids: List[int] = []

        for line in tqdm(self._iter_lines(), desc=f"Tokenizing {self.text_path.name}"):
            ids = self.base_tokenizer.encode(
                line,
                add_special_tokens=False,
            )

            if not ids:
                continue

            buffer_ids.extend(ids)

            if self.add_eos_between_lines:
                buffer_ids.append(eos_id)

            while len(buffer_ids) >= self.block_size:
                block_ids = buffer_ids[: self.block_size]
                buffer_ids = buffer_ids[self.block_size :]

                examples.append(self._make_example(block_ids))

        if len(buffer_ids) >= self.min_tokens_per_block:
            examples.append(self._make_example(buffer_ids))

        return examples

    def _make_example(self, input_ids: List[int]) -> Dict[str, List[int]]:
        cho_labels, jung_labels, jong_labels = make_jamo_labels(
            input_ids,
            self.base_tokenizer,
            self.jamo_tokenizer,
        )

        assert len(input_ids) == len(cho_labels)
        assert len(input_ids) == len(jung_labels)
        assert len(input_ids) == len(jong_labels)

        return {
            "input_ids": input_ids,
            "labels": input_ids.copy(),
            "cho_labels": cho_labels,
            "jung_labels": jung_labels,
            "jong_labels": jong_labels,
        }

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        return self.examples[idx]


def inspect_dataset_sample(
    dataset: JamoPackedTextDataset,
    base_tokenizer: Any,
    jamo_tokenizer: Any,
    index: int = 0,
    num_tokens: int = 30,
):
    """
    Debug helper to inspect token/jamo alignment.
    """
    ex = dataset[index]
    input_ids = ex["input_ids"][:num_tokens]
    cho_labels = ex["cho_labels"][:num_tokens]
    jung_labels = ex["jung_labels"][:num_tokens]
    jong_labels = ex["jong_labels"][:num_tokens]

    print(f"Dataset size: {len(dataset)}")
    print(f"Sample index: {index}")
    print(f"Showing first {len(input_ids)} tokens")
    print("-" * 120)

    for i, token_id in enumerate(input_ids):
        tok = base_tokenizer.decode([token_id], clean_up_tokenization_spaces=False)

        cho_id = cho_labels[i]
        jung_id = jung_labels[i]
        jong_id = jong_labels[i]

        cho = "IGNORE" if cho_id == -100 else jamo_tokenizer.decode([cho_id])
        jung = "IGNORE" if jung_id == -100 else jamo_tokenizer.decode([jung_id])
        jong = "IGNORE" if jong_id == -100 else jamo_tokenizer.decode([jong_id])

        print(
            f"{i:>4} | "
            f"id={token_id:>6} | "
            f"tok={repr(tok):>12} | "
            f"CHO={cho:>18} | "
            f"JUNG={jung:>18} | "
            f"JONG={jong:>18}"
        )


if __name__ == "__main__":
    from transformers import AutoTokenizer

    BASE_MODEL = "EleutherAI/polyglot-ko-1.3b"
    JAMO_TOKENIZER_PATH = "./polyglot-ko-1.3b-jamo-tokenizer"

    TRAIN_TEXT = ".txt"

    base_tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    jamo_tok = AutoTokenizer.from_pretrained(JAMO_TOKENIZER_PATH, local_files_only=True)

    dataset = JamoPackedTextDataset(
        text_path=TRAIN_TEXT,
        base_tokenizer=base_tok,
        jamo_tokenizer=jamo_tok,
        block_size=2048,
        max_lines=10000,  # sanity check. Set None for full data.
        cache_path="",
        overwrite_cache=True,
    )

    inspect_dataset_sample(
        dataset=dataset,
        base_tokenizer=base_tok,
        jamo_tokenizer=jamo_tok,
        index=0,
        num_tokens=50,
    )
