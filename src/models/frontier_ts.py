"""AI Frontier Time-Series Baselines:
6. TimeXer (Wang et al., NeurIPS 2024 — Exogenous-to-Endogenous Cross-Attention Transformer)
7. TimePro (Ma et al., ICML 2025 — Variable- and Time-Aware Hyper-State Black-Box Selective SSM / Mamba)
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..ssm_scan import parallel_first_order_scan


class TimeXerBlock(nn.Module):
    """Core TimeXer layer: Patch Self-Attention + Global-Token-to-Exogenous-Variate Cross-Attention + FFN."""

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, patch_and_glb: torch.Tensor, exog_var_tokens: torch.Tensor, n_glb: int) -> torch.Tensor:
        # 1. Temporal self-attention across patch tokens + global structural query tokens
        sa_out, _ = self.self_attn(patch_and_glb, patch_and_glb, patch_and_glb, need_weights=False)
        x = self.norm1(patch_and_glb + sa_out)

        # 2. Exogenous-to-Endogenous Cross-Attention on global structural tokens
        patches, glb = x[:, :-n_glb, :], x[:, -n_glb:, :]
        ca_out, _ = self.cross_attn(glb, exog_var_tokens, exog_var_tokens, need_weights=False)
        glb = self.norm2(glb + ca_out)

        x = torch.cat([patches, glb], dim=1)
        return self.norm3(x + self.ffn(x))


class TimeXer(nn.Module):
    """Wang et al. (NeurIPS, 2024): Empowering Transformers with Exogenous Variables."""

    def __init__(
        self,
        in_dim: int,
        lookback: int,
        num_targets: int,
        patch_len: int = 12,
        d_model: int = 96,
        num_layers: int = 3,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.lookback = lookback
        self.num_targets = num_targets
        self.patch_len = patch_len
        self.num_patches = lookback // patch_len

        # Patch embedding of multivariate exogenous sequence
        self.patch_proj = nn.Linear(patch_len * in_dim, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.03)

        # Variate-level embedding of each entire exogenous time series (F tokens of dim d_model)
        self.exog_variate_proj = nn.Sequential(
            nn.Linear(lookback, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.exog_var_id = nn.Parameter(torch.randn(1, in_dim, d_model) * 0.03)

        # Global endogenous structural query tokens (1 per target structural channel)
        self.struct_glb_tokens = nn.Parameter(torch.randn(1, num_targets, d_model) * 0.04)

        self.layers = nn.ModuleList([TimeXerBlock(d_model, n_heads=4, dropout=0.1) for _ in range(num_layers)])
        self.inst_proj = nn.Linear(in_dim, d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, batch: Dict[str, torch.Tensor], horizon_min: int = 0) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x_seq = batch["x_seq"]  # (B, Lw, F)
        bsz, lw, f_dim = x_seq.shape
        used_len = self.num_patches * self.patch_len
        x_cut = x_seq[:, lw - used_len :, :]

        p_tok = self.patch_proj(x_cut.reshape(bsz, self.num_patches, self.patch_len * f_dim)) + self.pos_emb
        exog_tok = self.exog_variate_proj(x_seq.transpose(1, 2)) + self.exog_var_id  # (B, F, D)
        glb_tok = self.struct_glb_tokens.expand(bsz, self.num_targets, -1)

        seq = torch.cat([p_tok, glb_tok], dim=1)
        for layer in self.layers:
            seq = layer(seq, exog_tok, self.num_targets)

        glb_final = seq[:, -self.num_targets :, :]  # (B, C, D)
        inst_ctx = self.inst_proj(x_seq[:, -1, :]).unsqueeze(1).expand_as(glb_final)
        out = self.head(torch.cat([glb_final, inst_ctx], dim=-1)).squeeze(-1)
        return out, None


class BlackBoxSelectiveSSMBlock(nn.Module):
    """Unconstrained Mamba / TimePro Hyper-State Selective Scan Block (SiLU gating + input-dependent step Delta)."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=3, padding=2, groups=d_model)
        self.dt_proj = nn.Linear(d_model, d_model)
        self.b_proj = nn.Linear(d_model, d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        self.a_log = nn.Parameter(torch.log(torch.linspace(1.0, 16.0, d_model)))
        self.out_proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D)."""
        residual = x
        u_norm = self.norm(x)
        xz = self.in_proj(u_norm)
        x_branch, z_gate = xz.chunk(2, dim=-1)

        # Causal depthwise local conv
        x_conv = F.silu(self.conv1d(x_branch.transpose(1, 2))[:, :, : x.shape[1]].transpose(1, 2))

        delta = F.softplus(self.dt_proj(x_conv))  # unconstrained input-dependent step size
        a_neg = -torch.exp(self.a_log).view(1, 1, -1)
        alpha = torch.exp(torch.clamp(delta * a_neg, min=-30.0, max=-1e-5))
        b_forcing = delta * self.b_proj(x_conv) * x_conv
        h0 = torch.zeros(x.shape[0], x.shape[2], device=x.device, dtype=x.dtype)
        h_seq = parallel_first_order_scan(alpha, b_forcing, h0)

        y = h_seq * self.c_proj(x_conv) * F.silu(z_gate)
        return residual + self.drop(self.out_proj(y))


class TimePro(nn.Module):
    """Ma et al. (ICML, 2025): Variable- and Time-Aware Hyper-State Selective SSM (Black-box SSM counterpart)."""

    def __init__(
        self,
        in_dim: int,
        lookback: int,
        num_targets: int,
        patch_len: int = 6,
        d_model: int = 96,
        num_layers: int = 3,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.patch_len = patch_len
        self.num_patches = lookback // patch_len
        self.num_targets = num_targets

        # Time-aware intra-patch multi-delay hyper-state projection
        self.hyper_patch = nn.Sequential(
            nn.Linear(patch_len * in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )
        # Variable-aware selective mixer across exogenous channels
        self.var_hyper_proj = nn.Linear(lookback, d_model)
        self.var_ssm = BlackBoxSelectiveSSMBlock(d_model, dropout=0.1)

        self.time_blocks = nn.ModuleList([BlackBoxSelectiveSSMBlock(d_model, dropout=0.1) for _ in range(num_layers)])
        self.cross_gate = nn.Linear(d_model * 2 + in_dim, d_model)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, num_targets),
        )

    def forward(self, batch: Dict[str, torch.Tensor], horizon_min: int = 0) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x_seq = batch["x_seq"]  # (B, Lw, F)
        bsz, lw, f_dim = x_seq.shape
        used_len = self.num_patches * self.patch_len
        x_cut = x_seq[:, lw - used_len :, :]

        t_hyper = self.hyper_patch(x_cut.reshape(bsz, self.num_patches, self.patch_len * f_dim))
        for blk in self.time_blocks:
            t_hyper = blk(t_hyper)
        h_time = t_hyper[:, -1, :]

        v_tokens = self.var_hyper_proj(x_seq.transpose(1, 2))  # (B, F, D)
        v_hyper = self.var_ssm(v_tokens).mean(dim=1)           # (B, D)

        fused = self.cross_gate(torch.cat([h_time, v_hyper, x_seq[:, -1, :]], dim=-1))
        return self.head(fused), None
