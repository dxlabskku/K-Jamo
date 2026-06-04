# collator.py

import torch
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class JamoCausalLMCollator:
    tokenizer: Any
    pad_to_multiple_of: int | None = 8

    def __call__(self, features: List[Dict[str, Any]]):
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]
        cho_labels = [f["cho_labels"] for f in features]
        jung_labels = [f["jung_labels"] for f in features]
        jong_labels = [f["jong_labels"] for f in features]

        max_len = max(len(x) for x in input_ids)

        if self.pad_to_multiple_of is not None:
            if max_len % self.pad_to_multiple_of != 0:
                max_len = (
                    (max_len // self.pad_to_multiple_of) + 1
                ) * self.pad_to_multiple_of

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        batch_cho_labels = []
        batch_jung_labels = []
        batch_jong_labels = []

        for ids, lab, cho, jung, jong in zip(
            input_ids, labels, cho_labels, jung_labels, jong_labels
        ):
            pad_len = max_len - len(ids)

            batch_input_ids.append(ids + [pad_id] * pad_len)
            batch_attention_mask.append([1] * len(ids) + [0] * pad_len)

            batch_labels.append(lab + [-100] * pad_len)
            batch_cho_labels.append(cho + [-100] * pad_len)
            batch_jung_labels.append(jung + [-100] * pad_len)
            batch_jong_labels.append(jong + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "cho_labels": torch.tensor(batch_cho_labels, dtype=torch.long),
            "jung_labels": torch.tensor(batch_jung_labels, dtype=torch.long),
            "jong_labels": torch.tensor(batch_jong_labels, dtype=torch.long),
        }