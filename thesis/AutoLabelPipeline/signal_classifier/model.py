"""
SignalClassifier: TimeSformer-style temporal classifier on cached DINOv2 features.

Architecture:
    linear_projection
    -> learned temporal + spatial positional encodings
    -> N x DividedAttentionBlock (divided space-time attention)
    -> mean pool over T x P tokens
    -> 2-layer MLP head -> 4-class logits

Labels: 0=none, 1=left, 2=right, 3=hazard

Attention uses torch.nn.MultiheadAttention, which automatically dispatches to
FlashAttention 2 on supported hardware (CUDA, PyTorch ≥ 2.0) with no extra install.
"""

import torch
import torch.nn as nn


LABEL_NAMES = ["none", "left", "right", "hazard"]
LABEL_MAP = {name: idx for idx, name in enumerate(LABEL_NAMES)}


class DividedAttentionBlock(nn.Module):
    """
    One layer of divided space-time attention (pre-norm style).

    Temporal attention: for each spatial position p, attend across all T frames.
    Spatial attention:  for each time step t, attend across all P patches.

    Both use residual connections and separate LayerNorms.  A shared FFN follows.

    Parameters
    ----------
    d_model   : token embedding dimension
    num_heads : number of attention heads (d_model must be divisible by num_heads)
    dropout   : applied inside attention and FFN
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # Temporal attention
        self.temporal_norm = nn.LayerNorm(d_model)
        self.temporal_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )

        # Spatial attention
        self.spatial_norm = nn.LayerNorm(d_model)
        self.spatial_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )

        # Feed-forward network
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : FloatTensor [B, T, P, d_model]

        Returns
        -------
        FloatTensor [B, T, P, d_model]
        """
        B, T, P, D = x.shape

        # ── Temporal attention ─────────────────────────────────────────────
        # Treat each (batch, spatial-patch) pair as an independent temporal sequence.
        # Reshape [B, T, P, D] -> [B*P, T, D], attend over T, reshape back.
        x_norm = self.temporal_norm(x)
        x_t = x_norm.permute(0, 2, 1, 3).reshape(B * P, T, D)
        attn_t, _ = self.temporal_attn(x_t, x_t, x_t)
        attn_t = attn_t.reshape(B, P, T, D).permute(0, 2, 1, 3)   # [B, T, P, D]
        x = x + attn_t

        # ── Spatial attention ──────────────────────────────────────────────
        # Treat each (batch, time-step) pair as an independent spatial sequence.
        # Reshape [B, T, P, D] -> [B*T, P, D], attend over P, reshape back.
        x_norm = self.spatial_norm(x)
        x_s = x_norm.reshape(B * T, P, D)
        attn_s, _ = self.spatial_attn(x_s, x_s, x_s)
        attn_s = attn_s.reshape(B, T, P, D)
        x = x + attn_s

        # ── FFN ────────────────────────────────────────────────────────────
        x = x + self.ffn(self.ffn_norm(x))

        return x


class SignalClassifier(nn.Module):
    """
    Turn signal classifier over pre-extracted DINOv2 feature windows.

    Input  : [B, T, P, d_dino]  — batch of cached DINO windows
    Output : [B, 4]             — logits over {none, left, right, hazard}

    Parameters
    ----------
    T         : window size in frames (fixed at training time)
    P         : spatial tokens per frame after downsampling
    d_dino    : DINO feature dimension (384 for ViT-S/14)
    d_model   : internal embedding dimension
    d_hidden  : classification MLP hidden dimension
    num_layers: number of DividedAttentionBlocks
    num_heads : attention heads per block
    dropout   : dropout rate
    """

    def __init__(
        self,
        T: int,
        P: int,
        d_dino: int = 384,
        d_model: int = 256,
        d_hidden: int = 128,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.T = T
        self.P = P

        # Project raw DINO features (d_dino) into model dimension (d_model)
        self.linear_projection = nn.Linear(d_dino, d_model)

        # Learned positional encodings, broadcast over the complementary dimension.
        # temporal_pos[t] is added to all P patch tokens at time step t.
        # spatial_pos[p]  is added to all T time tokens for patch position p.
        self.temporal_pos = nn.Parameter(torch.zeros(T, 1, d_model))
        self.spatial_pos  = nn.Parameter(torch.zeros(1, P, d_model))
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)
        nn.init.trunc_normal_(self.spatial_pos,  std=0.02)

        # Transformer backbone
        self.layers = nn.ModuleList(
            [DividedAttentionBlock(d_model, num_heads, dropout) for _ in range(num_layers)]
        )

        # Classification head
        self.head_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : FloatTensor [B, T, P, d_dino]

        Returns
        -------
        logits : FloatTensor [B, 4]
        """
        # Project to d_model
        x = self.linear_projection(x)          # [B, T, P, d_model]

        # Add positional encodings (broadcasting handles the missing dims)
        x = x + self.temporal_pos              # [B, T, P, d_model]
        x = x + self.spatial_pos               # [B, T, P, d_model]

        # Divided space-time attention layers
        for layer in self.layers:
            x = layer(x)                       # [B, T, P, d_model]

        # Mean pool over all T * P tokens -> single vector per sample
        x = self.head_norm(x)
        x = x.mean(dim=(1, 2))                 # [B, d_model]

        return self.head(x)                    # [B, 4]
