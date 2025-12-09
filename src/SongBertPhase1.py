import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig


class SongBertModelPhase1(nn.Module):
    def __init__(self, bert_model, loss_fn=None):
        super().__init__()
        self.bert = bert_model
        self.loss_fn = loss_fn

        hidden_dim = bert_model.config.hidden_size

        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.model_type = "songbert-phase1"

    # ----------------------------------------------------------------------
    # Forward pass
    # ----------------------------------------------------------------------
    def forward(
        self,
        target_input_ids=None,
        target_attention_mask=None,
        context_input_ids=None,
        context_attention_mask=None,
        labels=None,
        **kwargs
    ):

        # ---- Target embedding ----
        target_out = self.bert(
            input_ids=target_input_ids,
            attention_mask=target_attention_mask
        )
        cls_target = target_out.last_hidden_state[:, 0, :]

        # ---- Context embedding ----
        B, N, L = context_input_ids.shape
        context_input_ids_flat = context_input_ids.view(B * N, L)
        context_attention_flat = context_attention_mask.view(B * N, L)

        context_out = self.bert(
            input_ids=context_input_ids_flat,
            attention_mask=context_attention_flat
        )
        cls_context = context_out.last_hidden_state[:, 0, :]
        cls_context = cls_context.view(B, N, -1)

        # ---- Combine features ----
        x = torch.cat([
            cls_target.unsqueeze(1).expand(-1, N, -1),
            cls_context,
            torch.abs(cls_target.unsqueeze(1) - cls_context),
            cls_target.unsqueeze(1) * cls_context
        ], dim=-1)

        logits = self.scorer(x).squeeze(-1)

        if labels is not None and self.loss_fn is not None:
            loss = self.loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}

        return {"logits": logits}

    # ----------------------------------------------------------------------
    # Save model in HF-compatible format
    # ----------------------------------------------------------------------
    def save_pretrained(self, save_directory):
        save_directory = str(save_directory)

        # 1. Save model weights
        torch.save(self.state_dict(), f"{save_directory}/pytorch_model.bin")

        # 2. Save BERT’s config
        self.bert.config.save_pretrained(save_directory)

        # 3. Save custom model info
        config = {
            "model_type": self.model_type,
            "bert_model_name": self.bert.name_or_path,
        }
        with open(f"{save_directory}/songbert_config.json", "w") as f:
            json.dump(config, f)

        print(f"SongBert model saved to {save_directory}")

    # ----------------------------------------------------------------------
    # Load model from HF format
    # ----------------------------------------------------------------------
    @classmethod
    def from_pretrained(cls, load_directory, loss_fn=None):
        load_directory = str(load_directory)

        # 1. Load custom config
        with open(f"{load_directory}/songbert_config.json", "r") as f:
            cfg = json.load(f)

        bert_name = cfg["bert_model_name"]

        # 2. Load BERT encoder
        bert = AutoModel.from_pretrained(load_directory)

        # 3. Create instance
        model = cls(bert_model=bert, loss_fn=loss_fn)

        # 4. Load weights
        state = torch.load(f"{load_directory}/pytorch_model.bin", map_location="cpu")
        model.load_state_dict(state)

        print(f"Loaded SongBert Phase 1 from {load_directory}")
        return model
