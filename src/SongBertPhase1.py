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
        
        self.model_type = "songbert-phase1"

    def forward(
        self,
        target_input_ids=None,
        target_attention_mask=None,
        context_input_ids=None,
        context_attention_mask=None,
        labels=None,
        **kwargs
    ):
        device = next(self.parameters()).device

        target_input_ids = target_input_ids.to(device)
        target_attention_mask = target_attention_mask.to(device)

        target_out = self.bert(
            input_ids=target_input_ids,
            attention_mask=target_attention_mask
        )
        cls_target = target_out.last_hidden_state[:, 0, :]


        B, N, L = context_input_ids.shape
        context_input_ids_flat = context_input_ids.view(B * N, L)
        context_attention_flat = context_attention_mask.view(B * N, L)

        context_out = self.bert(
            input_ids=context_input_ids_flat,
            attention_mask=context_attention_flat
        )
        cls_context = context_out.last_hidden_state[:, 0, :]
        cls_context = cls_context.view(B, N, -1)

        cls_target = F.normalize(cls_target, dim=-1)
        cls_context = F.normalize(cls_context, dim=-1)
        logits = torch.einsum("bd,bnd->bn", cls_target, cls_context)

        if labels is not None and self.loss_fn is not None:
            loss = self.loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}

        return {"logits": logits}
