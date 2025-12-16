import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

class SongBertModelPhase3(nn.Module):
    def __init__(self, bert_model, loss_fn=None):
        super().__init__()
        self.bert = bert_model
        self.loss_fn = loss_fn

        hidden_dim = bert_model.config.hidden_size

        self.model_type = "songbert-phase3"

    # ----------------------------------------------------------------------
    # Forward pass
    # ----------------------------------------------------------------------
    def forward(
        self,
        target_input_ids=None,
        target_attention_mask=None,
        context_input_ids=None,
        context_attention_mask=None,
        labels=None
    ):

        target_out = self.bert(
            input_ids=target_input_ids,
            attention_mask=target_attention_mask
        )
        cls_target = target_out.last_hidden_state[:, 0, :]

        # context was a list where each lyric is an element
        # context was list of lists, where each sublist was a songid
        B, Sa, So, L = context_input_ids.shape # batch, samples, songs, lyrics

        context_input_ids_flat = context_input_ids.view(B * Sa * So, L)
        context_attention_flat = context_attention_mask.view(B * Sa * So, L)

        context_out = self.bert(
            input_ids=context_input_ids_flat,
            attention_mask=context_attention_flat
        )
        cls_songs = context_out.last_hidden_state[:, 0, :]
        # reconstruct it
        cls_songs = cls_songs.view(B, Sa, So, -1)

        cls_sample = cls_songs.mean(dim=2)
        cls_target = F.normalize(cls_target, dim=-1)
        cls_sample = F.normalize(cls_sample, dim=-1)
        logits = torch.einsum("bd,bnd->bn", cls_target, cls_sample)
  

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

        print(f"SongBert Phase 3 model saved to {save_directory}")

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

        print(f"Loaded SongBert for Phase 3 from {load_directory}")
        return model
