import os
from math import sqrt

import numpy as np
import torch
import torch.nn as nn

from utils.masking import TriangularCausalMask


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None,
                 attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, epoch=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            mask = attn_mask.mask if hasattr(attn_mask, "mask") else attn_mask
            scores = scores.masked_fill(mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        return V.contiguous(), None


class GradualProbSparseAttention(nn.Module):
    """
    Runnable G-PSA implementation that keeps the original intended design:
      1) warm-up with all queries active;
      2) gradual query pruning after warm-up;
      3) accumulated PRE-SOFTMAX query compatibility scores are used for pruning;
      4) removed-query outputs are replaced by mean(V);
      5) only active queries perform QK attention computation.

    Gradual ProbSparse Attention implementation used by Gformer.
    Query importance is accumulated during training and used for gradual pruning.
    """

    def __init__(self, mask_flag=True, factor=5, scale=None,
                 attention_dropout=0.1, output_attention=False,
                 warmup_epochs=2, removal_rate=0.1):
        super(GradualProbSparseAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.factor = factor
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

        self.warmup_epochs = int(warmup_epochs)
        self.removal_rate = float(removal_rate)

        # Dynamic state. Kept outside parameters because its length depends on
        # the sequence length of each attention layer.
        self.active_mask = None       # [L_Q], bool
        self.score_sum = None         # [L_Q], float
        self.score_count = None       # [L_Q], float
        self.last_epoch = None

        self.debug = os.environ.get("GFORMER_DEBUG", "0") == "1"

    def get_extra_state(self):
        # Allows EarlyStopping/model.state_dict() to preserve pruning state.
        return {
            "active_mask": None if self.active_mask is None else self.active_mask.detach().cpu(),
            "score_sum": None if self.score_sum is None else self.score_sum.detach().cpu(),
            "score_count": None if self.score_count is None else self.score_count.detach().cpu(),
            "last_epoch": self.last_epoch,
        }

    def set_extra_state(self, state):
        if not state:
            return
        self.active_mask = state.get("active_mask", None)
        self.score_sum = state.get("score_sum", None)
        self.score_count = state.get("score_count", None)
        self.last_epoch = state.get("last_epoch", None)

    def _ensure_state(self, L_Q, device):
        needs_init = (
            self.active_mask is None
            or self.active_mask.numel() != L_Q
        )

        if needs_init:
            self.active_mask = torch.ones(L_Q, dtype=torch.bool, device=device)
            self.score_sum = torch.zeros(L_Q, dtype=torch.float32, device=device)
            self.score_count = torch.zeros(L_Q, dtype=torch.float32, device=device)
            self.last_epoch = None
        else:
            self.active_mask = self.active_mask.to(device)
            self.score_sum = self.score_sum.to(device)
            self.score_count = self.score_count.to(device)

    def _importance(self):
        # Cumulative average score for every query position.
        denom = torch.clamp(self.score_count, min=1.0)
        return self.score_sum / denom

    def _prune_once(self):
        active_idx = torch.nonzero(self.active_mask, as_tuple=False).squeeze(-1)
        n_active = int(active_idx.numel())

        if n_active <= 1:
            return 0

        k = max(int(self.removal_rate * n_active), 1)
        k = min(k, n_active - 1)

        importance = self._importance()[active_idx]
        prune_local = torch.topk(importance, k=k, largest=False).indices
        prune_idx = active_idx[prune_local]

        self.active_mask[prune_idx] = False
        return int(k)

    def _handle_epoch_transition(self, epoch):
        if not self.training or epoch is None:
            return

        epoch = int(epoch)

        if self.last_epoch is None:
            self.last_epoch = epoch
            if self.debug:
                print(
                    f"[G-PSA] epoch={epoch} | warm-up "
                    f"| active={int(self.active_mask.sum())}/{self.active_mask.numel()}"
                )
            return

        if epoch == self.last_epoch:
            return

        # The first pruning is applied at the beginning of the first epoch
        # after warm-up, using the accumulated warm-up statistics.
        removed = 0
        if epoch > self.warmup_epochs:
            removed = self._prune_once()

        self.last_epoch = epoch

        if self.debug:
            phase = "warm-up" if epoch <= self.warmup_epochs else "pruning"
            print(
                f"[G-PSA] epoch={epoch} | {phase} | removed={removed} "
                f"| active={int(self.active_mask.sum())}/{self.active_mask.numel()}"
            )

    def _get_mask(self, attn_mask, B, L_Q, active_idx, device):
        if not self.mask_flag:
            return None

        if attn_mask is None:
            attn_mask = TriangularCausalMask(B, L_Q, device=device)

        mask = attn_mask.mask if hasattr(attn_mask, "mask") else attn_mask

        # Standard Gformer masks are [B, 1, L_Q, L_K].
        if mask.dim() == 4:
            return mask[:, :, active_idx, :]
        if mask.dim() == 3:
            return mask[:, active_idx, :]
        raise ValueError(f"Unsupported attention mask shape: {tuple(mask.shape)}")

    def forward(self, queries, keys, values, attn_mask, epoch=None):
        # AttentionLayer already projected Q/K/V:
        # queries: [B, L_Q, H, E]
        # keys:    [B, L_K, H, E]
        # values:  [B, L_K, H, D]
        B, L_Q, H, E = queries.shape
        _, L_K, _, D = values.shape

        self._ensure_state(L_Q, queries.device)
        self._handle_epoch_transition(epoch)

        active_idx = torch.nonzero(self.active_mask, as_tuple=False).squeeze(-1)
        if active_idx.numel() == 0:
            raise RuntimeError("G-PSA removed all queries; at least one query must remain active.")

        q_active = queries[:, active_idx, :, :]  # [B, A, H, E]

        # Compute QK only for active queries: [B, H, A, L_K]
        raw_scores = torch.einsum("bahe,bshe->bhas", q_active, keys)

        # Accumulate query importance only during training.
        # This preserves the intent of the original code, which maintained
        # pre-softmax attention-score history.
        if self.training and epoch is not None:
            batch_query_score = raw_scores.detach().mean(dim=(0, 1, 3))  # [A]
            self.score_sum[active_idx] += batch_query_score
            self.score_count[active_idx] += 1.0

        scale = self.scale or 1. / sqrt(E)
        scores = scale * raw_scores

        active_mask = self._get_mask(
            attn_mask, B, L_Q, active_idx, queries.device
        )
        if active_mask is not None:
            scores = scores.masked_fill(active_mask, -np.inf)

        A_active = self.dropout(torch.softmax(scores, dim=-1))
        V_active = torch.einsum("bhas,bshd->bahd", A_active, values)

        # Removed queries receive mean(V), preserving sequence length.
        V_mean = values.mean(dim=1, keepdim=True)  # [B, 1, H, D]
        output = V_mean.expand(B, L_Q, H, D).clone()
        output[:, active_idx, :, :] = V_active

        attn = None
        if self.output_attention:
            # For visualization only; removed-query rows are zero.
            attn = torch.zeros(
                B, H, L_Q, L_K,
                dtype=A_active.dtype,
                device=A_active.device,
            )
            attn[:, :, active_idx, :] = A_active

        return output.contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads,
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask, epoch=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            epoch=epoch,
        )

        if self.mix:
            out = out.transpose(2, 1).contiguous()

        out = out.view(B, L, -1)
        return self.out_projection(out), attn
