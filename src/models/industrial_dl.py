"""Antenna & Industrial Journal Deep Baselines:
3. VMD-BiLSTM (Wu et al., Heliyon 2024 — Radio Telescope Multi-Sensor Model)
4. Causal-Dilated TCN (Mariani et al., MSSP 2024 — Long-Series Thermal Lag TCN)
5. ST-GAT (Yang et al., IEEE TIM 2024 / Xu et al., IEEE TIM 2025 — Spatiotemporal Graph Attention Soft Sensor)
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class VMDBiLSTM(nn.Module):
    """Wu et al. (Heliyon, 2024): Strictly causal multi-band VMD sub-signals + exogenous sequence -> BiLSTM."""

    def __init__(self, in_dim: int, vmd_dim: int, num_targets: int, hidden_dim: int = 96, num_layers: int = 2):
        super().__init__()
        total_in = in_dim + vmd_dim
        # Stride-3 causal 1D stem (360 -> 120 steps) + exact last-step skip for fast & accurate cuDNN BiLSTM
        self.stem_conv = nn.Conv1d(total_in, hidden_dim, kernel_size=3, stride=3, padding=0)
        self.bilstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.attn_score = nn.Linear(hidden_dim * 2, 1)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 4 + total_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_targets),
        )

    def forward(self, batch: Dict[str, torch.Tensor], horizon_min: int = 0) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x_seq = batch["x_seq"]
        vmd_seq = batch["vmd_seq"]
        inp = torch.cat([x_seq, vmd_seq], dim=-1)  # (B, Lw, F + Kvmd)
        last_raw = inp[:, -1, :]

        # Trim front if Lw not divisible by 3 so last step aligns with t
        rem = inp.shape[1] % 3
        inp_trim = inp[:, rem:, :] if rem > 0 else inp
        tokens = F.gelu(self.stem_conv(inp_trim.transpose(1, 2))).transpose(1, 2)  # (B, L/3, H)

        lstm_out, _ = self.bilstm(tokens)  # (B, L/3, 2H)
        w_attn = torch.softmax(self.attn_score(lstm_out), dim=1)
        ctx = torch.sum(w_attn * lstm_out, dim=1)
        feat = torch.cat([lstm_out[:, -1, :], ctx, last_raw], dim=-1)
        return self.head(feat), None


class CausalChomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.chomp_size].contiguous() if self.chomp_size > 0 else x


class DilatedResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation),
            CausalChomp1d(pad),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation),
            CausalChomp1d(pad),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x + self.net(x)
        return self.norm(out.transpose(1, 2)).transpose(1, 2)


class CausalDilatedTCN(nn.Module):
    """Mariani et al. (MSSP, 2024): Long-horizon Causal Dilated TCN for structural thermal-lag strain/disp."""

    def __init__(self, in_dim: int, num_targets: int, channels: int = 80, kernel_size: int = 5):
        super().__init__()
        self.in_proj = nn.Conv1d(in_dim, channels, kernel_size=1)
        # Receptive field = 1 + 2*(5-1)*(1+2+4+8+16+32+64) = 1017 > 360 min
        dilations = [1, 2, 4, 8, 16, 32, 64]
        self.blocks = nn.ModuleList(
            [DilatedResidualBlock(channels, kernel_size=kernel_size, dilation=d, dropout=0.1) for d in dilations]
        )
        self.head = nn.Sequential(
            nn.Linear(channels + in_dim, channels),
            nn.GELU(),
            nn.Linear(channels, num_targets),
        )

    def forward(self, batch: Dict[str, torch.Tensor], horizon_min: int = 0) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x_seq = batch["x_seq"]  # (B, Lw, F)
        h = self.in_proj(x_seq.transpose(1, 2))
        for blk in self.blocks:
            h = blk(h)
        h_last = h[:, :, -1]
        return self.head(torch.cat([h_last, x_seq[:, -1, :]], dim=-1)), None


class DynamicGATv2Layer(nn.Module):
    """Memory-efficient O(N^2 H) Multi-head Graph Attention across exogenous + target sensor nodes."""

    def __init__(self, node_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = node_dim // num_heads
        self.w_src = nn.Linear(node_dim, node_dim, bias=False)
        self.w_dst = nn.Linear(node_dim, node_dim, bias=False)
        self.a_src = nn.Parameter(torch.randn(1, 1, num_heads, self.head_dim) * 0.08)
        self.a_dst = nn.Parameter(torch.randn(1, 1, num_heads, self.head_dim) * 0.08)
        self.out_proj = nn.Linear(node_dim, node_dim)
        self.norm = nn.LayerNorm(node_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        """nodes: (B_T, N_nodes, D)."""
        bt, n_nodes, d = nodes.shape
        src = self.w_src(nodes).view(bt, n_nodes, self.num_heads, self.head_dim)
        dst = self.w_dst(nodes).view(bt, n_nodes, self.num_heads, self.head_dim)
        e_src = torch.sum(src * self.a_src, dim=-1).unsqueeze(1)  # (bt, 1, N_src, H)
        e_dst = torch.sum(dst * self.a_dst, dim=-1).unsqueeze(2)  # (bt, N_dst, 1, H)
        scores = F.leaky_relu(e_dst + e_src, negative_slope=0.2)  # (bt, N_dst, N_src, H)
        alpha = self.drop(torch.softmax(scores, dim=2))
        agg = torch.einsum("bngh,bghd->bnhd", alpha, src).reshape(bt, n_nodes, d)
        return self.norm(nodes + self.out_proj(agg))


class STGAT(nn.Module):
    """Yang et al. (IEEE TIM, 2024) / Xu et al. (IEEE TIM, 2025): Spatiotemporal Graph Attention Network."""

    def __init__(self, in_dim: int, num_targets: int, node_dim: int = 64, patch_stride: int = 10):
        super().__init__()
        self.in_dim = in_dim
        self.num_targets = num_targets
        self.node_dim = node_dim
        self.patch_stride = patch_stride

        # Per-sensor temporal patch encoder (preserves distinct sensor nodes)
        self.sensor_patch_proj = nn.Linear(patch_stride, node_dim)
        self.sensor_id_emb = nn.Parameter(torch.randn(1, 1, in_dim, node_dim) * 0.04)
        self.target_query_nodes = nn.Parameter(torch.randn(1, 1, num_targets, node_dim) * 0.05)

        self.gat1 = DynamicGATv2Layer(node_dim, num_heads=4, dropout=0.1)
        self.gat2 = DynamicGATv2Layer(node_dim, num_heads=4, dropout=0.1)

        self.temporal_gru = nn.GRU(
            input_size=node_dim,
            hidden_size=node_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )
        self.inst_skip = nn.Linear(in_dim, node_dim)
        self.readout = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),
            nn.GELU(),
            nn.Linear(node_dim, 1),
        )

    def forward(self, batch: Dict[str, torch.Tensor], horizon_min: int = 0) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x_seq = batch["x_seq"]  # (B, Lw, F)
        bsz, lw, f_dim = x_seq.shape
        p = self.patch_stride
        rem = lw % p
        x_trim = x_seq[:, rem:, :] if rem > 0 else x_seq
        n_steps = x_trim.shape[1] // p

        # (B, T_p, P, F) -> (B, T_p, F, P) -> (B, T_p, F, D)
        x_patched = x_trim.view(bsz, n_steps, p, f_dim).permute(0, 1, 3, 2)
        exog_nodes = self.sensor_patch_proj(x_patched) + self.sensor_id_emb  # (B, T_p, F, D)
        tgt_nodes = self.target_query_nodes.expand(bsz, n_steps, self.num_targets, self.node_dim)
        all_nodes = torch.cat([exog_nodes, tgt_nodes], dim=2)  # (B, T_p, F + C, D)

        flat_nodes = all_nodes.reshape(bsz * n_steps, f_dim + self.num_targets, self.node_dim)
        flat_nodes = self.gat2(self.gat1(flat_nodes))
        tgt_sp = flat_nodes[:, f_dim:, :].view(bsz, n_steps, self.num_targets, self.node_dim)

        # Temporal GRU across patches for each structural gauge node
        tgt_seq = tgt_sp.permute(0, 2, 1, 3).reshape(bsz * self.num_targets, n_steps, self.node_dim)
        gru_out, _ = self.temporal_gru(tgt_seq)
        h_final = gru_out[:, -1, :].view(bsz, self.num_targets, self.node_dim)

        inst_ctx = self.inst_skip(x_seq[:, -1, :]).unsqueeze(1).expand_as(h_final)
        pred = self.readout(torch.cat([h_final, inst_ctx], dim=-1)).squeeze(-1)
        return pred, None
