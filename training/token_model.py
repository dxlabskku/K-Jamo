import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast


class PolyglotKoWithJamoAux(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        jamo_vocab_size: int,
        cho_loss_weight: float = 0.1,
        jung_loss_weight: float = 0.1,
        jong_loss_weight: float = 0.1,
        torch_dtype=torch.float16,
    ):
        super().__init__()

        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )

        hidden_size = self.base_model.config.hidden_size

        self.cho_head = nn.Linear(hidden_size, jamo_vocab_size, bias=False).to(dtype=torch_dtype)
        self.jung_head = nn.Linear(hidden_size, jamo_vocab_size, bias=False).to(dtype=torch_dtype)
        self.jong_head = nn.Linear(hidden_size, jamo_vocab_size, bias=False).to(dtype=torch_dtype)

        self.cho_loss_weight = cho_loss_weight
        self.jung_loss_weight = jung_loss_weight
        self.jong_loss_weight = jong_loss_weight

        self.config = self.base_model.config
        self.cho_head.to(dtype=torch_dtype)
        self.jung_head.to(dtype=torch_dtype)
        self.jong_head.to(dtype=torch_dtype)
    def gradient_checkpointing_enable(self, **kwargs):
        self.base_model.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        self.base_model.gradient_checkpointing_disable()

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        return self.base_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.base_model.get_output_embeddings()

    def resize_token_embeddings(self, new_num_tokens: int):
        return self.base_model.resize_token_embeddings(new_num_tokens)

    def save_pretrained(self, save_directory: str, **kwargs):
        self.base_model.save_pretrained(save_directory, **kwargs)

        torch.save(
            {
                "cho_head": self.cho_head.state_dict(),
                "jung_head": self.jung_head.state_dict(),
                "jong_head": self.jong_head.state_dict(),
                "cho_loss_weight": self.cho_loss_weight,
                "jung_loss_weight": self.jung_loss_weight,
                "jong_loss_weight": self.jong_loss_weight,
            },
            f"{save_directory}/jamo_aux_heads.pt",
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        cho_labels=None,
        jung_labels=None,
        jong_labels=None,
        **kwargs,
    ):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )

        logits = outputs.logits
        hidden_states = outputs.hidden_states[-1]

        loss = None
        lm_loss = None
        cho_loss = None
        jung_loss = None
        jong_loss = None

        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            lm_loss = F.cross_entropy(
                shift_logits.float().view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )


            loss = lm_loss

        cho_logits = self.cho_head(hidden_states)
        jung_logits = self.jung_head(hidden_states)
        jong_logits = self.jong_head(hidden_states)

        if cho_labels is not None:
            shift_cho_logits = cho_logits[:, :-1, :].contiguous()
            shift_cho_labels = cho_labels[:, 1:].contiguous()

            cho_loss = F.cross_entropy(
                shift_cho_logits.float().view(-1, shift_cho_logits.size(-1)),
                shift_cho_labels.view(-1),
                ignore_index=-100,
            )

            loss = cho_loss * self.cho_loss_weight if loss is None else loss + cho_loss * self.cho_loss_weight

        if jung_labels is not None:
            shift_jung_logits = jung_logits[:, :-1, :].contiguous()
            shift_jung_labels = jung_labels[:, 1:].contiguous()

            jung_loss = F.cross_entropy(
                shift_jung_logits.float().view(-1, shift_jung_logits.size(-1)),
                shift_jung_labels.view(-1),
                ignore_index=-100,
            )

            loss = jung_loss * self.jung_loss_weight if loss is None else loss + jung_loss * self.jung_loss_weight

        if jong_labels is not None:
            shift_jong_logits = jong_logits[:, :-1, :].contiguous()
            shift_jong_labels = jong_labels[:, 1:].contiguous()

            jong_loss = F.cross_entropy(
                shift_jong_logits.float().view(-1, shift_jong_logits.size(-1)),
                shift_jong_labels.view(-1),
                ignore_index=-100,
            )

            loss = jong_loss * self.jong_loss_weight if loss is None else loss + jong_loss * self.jong_loss_weight

        return {
            "loss": loss,
            "lm_loss": lm_loss,
            "cho_loss": cho_loss,
            "jung_loss": jung_loss,
            "jong_loss": jong_loss,
            "logits": logits,
            "cho_logits": cho_logits,
            "jung_logits": jung_logits,
            "jong_logits": jong_logits,
        }

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        return self.base_model.generate(*args, **kwargs)