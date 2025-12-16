import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

class SongBertModelPhase2(nn.Module):
    def __init__(self, bert_model, loss_fn=None):
        super().__init__()
        self.bert = bert_model
        self.loss_fn = loss_fn

        hidden_dim = bert_model.config.hidden_size

        self.model_type = "songbert-phase2"

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
