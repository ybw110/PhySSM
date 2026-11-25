"""PhySSM: Physics-Guided Selective State-Space Deep Network for Large-Aperture Antenna Structural Dynamics."""

import math
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm

from ..ssm_scan import parallel_first_order_scan


class MultiScaleNonUniformCompressor(nn.Module):
    """Compresses distant history [0 .. L-L_fine) into exact K_coarse block steps (dt = stride min)
    concatenated with recent [L-L_fine .. L) at exact 1-minute resolution (dt = 1 min), exercising
    exact non-uniform ZOH continuous integration while accelerating scan 3.5x.
    """

    def __init__(self, fine_recent_steps: int = 60, coarse_stride: int = 5):
        super().__init__()
        self.fine_steps = fine_recent_steps
        self.stride = coarse_stride

    def forward(
        self,
        z_seq: torch.Tensor,
        w_pos_seq: torch.Tensor,
        dt_seq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, lw, dim = z_seq.shape
        if lw <= self.fine_steps + self.stride:
            return z_seq, w_pos_seq, dt_seq

        coarse_len = lw - self.fine_steps
        rem = coarse_len % self.stride
        c_start = rem
        n_coarse = (coarse_len - rem) // self.stride

        z_c = z_seq[:, c_start:coarse_len, :].view(bsz, n_coarse, self.stride, dim).mean(dim=2)
        w_c = w_pos_seq[:, c_start:coarse_len, :].view(bsz, n_coarse, self.stride, -1).mean(dim=2)
        dt_c = dt_seq[:, c_start:coarse_len, :].view(bsz, n_coarse, self.stride, 1).sum(dim=2)

        z_out = torch.cat([z_c, z_seq[:, coarse_len:, :]], dim=1)
        w_out = torch.cat([w_c, w_pos_seq[:, coarse_len:, :]], dim=1)
        dt_out = torch.cat([dt_c, dt_seq[:, coarse_len:, :]], dim=1)
        return z_out, w_out, dt_out


class ThermoConvectiveSelectiveSSMBlock(nn.Module):
    """Continuous-Time Dissipative Selective SSM Layer with Zero-Anchored Wind-Convective Rate Modulation.

    ODE: dh/dt = -Lambda(u_wd) * h + Lambda(u_wd) * (B * z_in)
    Exact ZOH recurrence over non-uniform step sizes dt_n:
        h_n = exp(-Lambda_n * dt_n) * h_{n-1} + (1 - exp(-Lambda_n * dt_n)) * (B * z_in_n)
    At calm wind (u_wd = 0), kappa(0) == 0 identically (Lambda(0) == 1/tau_0), with unconstrained slope for u_wd > 0.
    """

    def __init__(
        self,
        in_dim: int,
        state_dim: int,
        wind_pos_dim: int = 4,
        tau_min_init: float = 1.5,
        tau_max_init: float = 360.0,
        tau_floor: float = 0.5,
        eq_init_steps: int = 4,
        homogeneous_linear: bool = False,
        disable_wind_mod: bool = False,
        disable_ssm_memory: bool = False,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.tau_floor = tau_floor
        self.eq_init_steps = eq_init_steps
        self.homogeneous_linear = homogeneous_linear
        self.disable_wind_mod = disable_wind_mod
        self.disable_ssm_memory = disable_ssm_memory

        taus_init = np.exp(np.linspace(math.log(tau_min_init), math.log(tau_max_init), state_dim))
        eta_init = np.log(np.expm1(np.maximum(taus_init - tau_floor, 0.1)))
        self.eta_tau = nn.Parameter(torch.tensor(eta_init, dtype=torch.float32))

        if homogeneous_linear:
            self.in_proj = nn.Linear(in_dim, state_dim, bias=False)
            self.diff_mix = nn.Linear(state_dim, state_dim, bias=False)
        else:
            self.norm = nn.LayerNorm(in_dim)
            self.local_conv = nn.Conv1d(in_dim, state_dim, kernel_size=3, padding=2)
            self.gate_proj = nn.Linear(in_dim, state_dim)
            self.out_proj = nn.Sequential(
                nn.Linear(state_dim * 2, state_dim),
                nn.GELU(),
                nn.Dropout(0.08),
                nn.Linear(state_dim, state_dim),
            )
            self.skip_proj = nn.Identity() if in_dim == state_dim else nn.Linear(in_dim, state_dim)

        # Unconstrained wind modulation MLP (zero-anchored at w = 0 in forward; NO hardcoded monotonicity)
        self.wind_mod_net = nn.Sequential(
            nn.Linear(wind_pos_dim, 16),
            nn.SiLU(),
            nn.Linear(16, state_dim),
        )
        nn.init.zeros_(self.wind_mod_net[-1].bias)

    def get_base_taus(self) -> torch.Tensor:
        return F.softplus(self.eta_tau) + self.tau_floor

    def compute_wind_kappa(self, w_pos_seq: torch.Tensor) -> torch.Tensor:
        """Exact zero-anchored convective modulation: kappa(0) == 0 identically; free slope for w > 0."""
        w_feat = torch.sqrt(torch.clamp(w_pos_seq, min=0.0))
        w_zero = torch.zeros_like(w_feat)
        delta_psi = self.wind_mod_net(w_feat) - self.wind_mod_net(w_zero)
        c0 = F.softplus(torch.tensor(-0.8, device=w_feat.device, dtype=w_feat.dtype))
        return F.relu(F.softplus(delta_psi - 0.8) - c0)

    def forward(
        self,
        z_seq: torch.Tensor,
        w_pos_seq: torch.Tensor,
        dt_seq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.homogeneous_linear:
            z_forced = self.in_proj(z_seq)
        else:
            z_n = self.norm(z_seq)
            z_forced = F.silu(self.local_conv(z_n.transpose(1, 2))[:, :, : z_seq.shape[1]].transpose(1, 2))

        if self.disable_ssm_memory:
            h_seq = z_forced
        else:
            tau_0 = self.get_base_taus().float().view(1, 1, -1)
            if self.disable_wind_mod:
                lam = 1.0 / tau_0.expand_as(z_forced.float())
            else:
                kappa = self.compute_wind_kappa(w_pos_seq).float()
                lam = (1.0 + kappa) / tau_0

            alpha = torch.exp(-torch.clamp(lam * dt_seq.float(), min=1e-5, max=30.0))
            b_step = (1.0 - alpha) * z_forced.float()

            k0 = min(self.eq_init_steps, z_forced.shape[1])
            h0 = z_forced[:, :k0, :].float().mean(dim=1)
            h_seq = parallel_first_order_scan(alpha, b_step, h0).to(z_forced.dtype)

        if self.homogeneous_linear:
            out_seq = h_seq + 0.25 * self.diff_mix(z_forced - h_seq)
        else:
            gate = torch.sigmoid(self.gate_proj(z_n))
            bi_feat = torch.cat([h_seq * gate, (z_forced - h_seq) * gate], dim=-1)
            out_seq = self.skip_proj(z_seq) + self.out_proj(bi_feat)
        return out_seq, h_seq


class PhySSM(nn.Module):
    """Physics-Guided Selective State-Space Deep Network (`PhySSM`):
    1) Multi-layer Thermo-Convective Selective SSM over exact non-uniform ZOH steps (zero-anchored kappa(0) == 0),
    2) Subsystem-Decoupled Spatial Thermal Routing: Collinearity-free Bulk Ambient Bath drives elevated reflector
       BUS strains (49#03-15), while 4 Ground-Mast Spatial Gradients drive pedestal/alidade axis bearings (49#01-02),
    3) Latent Orthogonal Low-Rank Structural Mode Graph Coupling (`U_mode` in R^{C x r}),
    4) Extrapolation-Safe 1-Homogeneous Trunk + Lipschitz-Controlled Env & Actuation Residual Streams,
    5) Support-Aware Transient Aleatoric + Mahalanobis Epistemic Uncertainty.
    """

    def __init__(
        self,
        group_slices: Dict[str, List[int]],
        num_targets: int,
        u_svd_init: np.ndarray,
        maha_p99_train: float,
        hidden_dim: int = 96,
        hom_dim: int = 64,
        ped_dim: int = 24,
        num_layers: int = 3,
        mode_rank: int = 5,
        eq_init_steps: int = 4,
        ablation: str = "none",
    ):
        super().__init__()
        self.group_slices = group_slices
        self.num_targets = num_targets
        self.maha_p99_train = float(maha_p99_train)
        self.ablation = ablation

        # 6 Orthogonal Ablation Switches
        self.wo_ssm = (ablation == "wo_dissipative_ssm")
        self.wo_wind_mod = (ablation == "wo_wind_modulation")
        # `wo_spatial_grad`: disables Subsystem-Decoupled Spatial Routing (naively mixes ground-mast gradients
        # into shared trunk without dedicated pedestal differential tilt routing)
        self.wo_spatial_grad = (ablation == "wo_spatial_grad")
        self.wo_modes = (ablation == "wo_low_rank_modes")
        self.wo_dual_stream = (ablation == "wo_homogeneous_dual_stream")
        self.wo_gate = (ablation == "wo_actuation_quality_gate")

        eff_rank = min(int(mode_rank), num_targets)
        self.mode_rank = eff_rank

        self.idx_th_mean = group_slices["thermal_mean_aux"]         # [temp_mean, hum_mean, press_mean] (3)
        self.idx_th_stat = group_slices["thermal_stations"]         # [temp_06..20] (4)
        self.idx_th_grad = group_slices["thermal_spatial_grad"]     # [grad_06..20, range] (5)
        self.idx_wind = group_slices["wind_forcing"]                # (6)
        self.idx_op = group_slices["actuation_and_attitude"]        # (13)
        self.idx_qmask = group_slices["quality_masks"]              # (1)

        self.compressor = MultiScaleNonUniformCompressor(fine_recent_steps=60, coarse_stride=5)

        # (1A) 1-Homogeneous Dissipative Trunk
        hom_in_dim = len(self.idx_th_mean) + len(self.idx_th_stat) + len(self.idx_th_grad) + len(self.idx_wind) + len(self.idx_op)
        self.hom_ssm = ThermoConvectiveSelectiveSSMBlock(
            in_dim=hom_in_dim,
            state_dim=hom_dim,
            tau_min_init=1.5,
            tau_max_init=360.0,
            eq_init_steps=eq_init_steps,
            homogeneous_linear=True,
            disable_wind_mod=self.wo_wind_mod,
            disable_ssm_memory=self.wo_ssm,
        )

        # (1B) Deep Selective Thermo-Convective SSM Backbone (L = 3 stacked blocks)
        nl_in_dim = hom_in_dim + len(self.idx_qmask)
        self.in_stem = nn.Linear(nl_in_dim, hidden_dim)
        self.nl_blocks = nn.ModuleList([
            ThermoConvectiveSelectiveSSMBlock(
                in_dim=hidden_dim,
                state_dim=hidden_dim,
                tau_min_init=1.0,
                tau_max_init=360.0,
                eq_init_steps=eq_init_steps,
                homogeneous_linear=False,
                disable_wind_mod=self.wo_wind_mod,
                disable_ssm_memory=self.wo_ssm,
            )
            for _ in range(num_layers)
        ])

        # (1C) Pedestal-Level 4-Mast Spatial Gradient Differential Tilt Sub-SSM (drives axis bearings 49#01/02 ONLY!)
        n_axis = min(2, num_targets)
        ped_in_dim = len(self.idx_th_stat) + len(self.idx_th_grad) + len(self.idx_wind) + len(self.idx_op)
        self.pedestal_ssm = ThermoConvectiveSelectiveSSMBlock(
            in_dim=ped_in_dim,
            state_dim=ped_dim,
            tau_min_init=3.0,
            tau_max_init=240.0,
            eq_init_steps=eq_init_steps,
            homogeneous_linear=False,
            disable_wind_mod=self.wo_wind_mod,
            disable_ssm_memory=self.wo_ssm,
        )
        use_sn = not self.wo_dual_stream
        p_l1 = nn.Linear(ped_dim + len(self.idx_th_grad), ped_dim)
        p_l2 = nn.Linear(ped_dim, n_axis)
        self.pedestal_tilt_head = nn.Sequential(
            spectral_norm(p_l1) if use_sn else p_l1,
            nn.GELU(),
            nn.Dropout(0.10),
            spectral_norm(p_l2) if use_sn else p_l2,
        )

        # (2) Latent Orthogonal Low-Rank Structural Mode Basis U_mode in R^{C x r}
        init_u = torch.tensor(u_svd_init[:, :eff_rank], dtype=torch.float32)
        self.u_mode = nn.Parameter(init_u)
        self.modal_graph_mix = nn.Sequential(
            nn.Linear(eff_rank, eff_rank * 2),
            nn.GELU(),
            nn.Linear(eff_rank * 2, eff_rank),
        )
        self.hom_modal_lin = nn.Linear(eff_rank, eff_rank, bias=False)
        nn.init.eye_(self.hom_modal_lin.weight)

        hom_read_dim = hom_dim * 2 + hom_in_dim
        if self.wo_modes:
            self.hom_head_uncoupled = nn.Linear(hom_read_dim, num_targets, bias=True)
        else:
            self.hom_to_modes = nn.Linear(hom_read_dim, eff_rank, bias=False)
            self.hom_fine_diag = nn.Linear(hom_read_dim, num_targets, bias=True)

        # (3A) Unconstrained Actuation & Attitude Elastic Subnetwork
        op_ctx_dim = len(self.idx_op) * 2 + hidden_dim // 2
        self.op_proj = nn.Linear(hidden_dim, hidden_dim // 2)
        self.actuation_elastic_net = nn.Sequential(
            nn.Linear(op_ctx_dim, 64),
            nn.GELU(),
            nn.Linear(64, num_targets),
        )

        # (3B) Spectral-Normalized Bounded-Slope Environmental Residual Stream
        env_res_in = hidden_dim * 2 + len(self.idx_th_grad) + len(self.idx_wind)
        l1 = nn.Linear(env_res_in, hidden_dim)
        l2 = nn.Linear(hidden_dim, hidden_dim)
        self.env_res_mlp = nn.Sequential(
            spectral_norm(l1) if use_sn else l1,
            nn.GELU(),
            nn.Dropout(0.08),
            spectral_norm(l2) if use_sn else l2,
            nn.GELU(),
        )
        self.res_lip_scale = nn.Parameter(torch.tensor(2.2, dtype=torch.float32))

        if self.wo_modes:
            l_out = nn.Linear(hidden_dim, num_targets)
            self.res_head_uncoupled = spectral_norm(l_out) if use_sn else l_out
        else:
            l_m = nn.Linear(hidden_dim, eff_rank)
            l_d = nn.Linear(hidden_dim, num_targets)
            self.res_to_modes = spectral_norm(l_m) if use_sn else l_m
            self.res_fine_diag = spectral_norm(l_d) if use_sn else l_d

        # Quality & Actuation Regime Gate G_theta in [0, 1]^C
        gate_in_dim = len(self.idx_op) + len(self.idx_qmask) + 1
        self.regime_gate = nn.Sequential(
            nn.Linear(gate_in_dim, 48),
            nn.SiLU(),
            nn.Linear(48, num_targets * 2),
        )

        if self.wo_dual_stream:
            self.unconstrained_decoder = nn.Sequential(
                nn.Linear(hidden_dim * 2 + hom_in_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, num_targets),
            )

        # (4) Support-Aware Uncertainty Head (Transient-Aware Aleatoric + Mahalanobis Epistemic)
        alea_in_dim = hidden_dim + len(self.idx_op) * 2 + len(self.idx_wind)
        self.alea_head = nn.Sequential(
            nn.Linear(alea_in_dim, 64),
            nn.SiLU(),
            nn.Linear(64, num_targets),
        )
        nn.init.constant_(self.alea_head[-1].bias, -1.4)
        self.epistemic_lambda_raw = nn.Parameter(torch.full((num_targets,), -3.1, dtype=torch.float32))

        init_gain = 0.08 if num_targets == 2 else 1.0
        self.res_stream_gain = nn.Parameter(torch.tensor(init_gain, dtype=torch.float32))

    @torch.no_grad()
    def anchor_homogeneous_trunk_d1(
        self,
        slicer,
        train_idx_gpu: torch.Tensor,
        val_idx_gpu: torch.Tensor,
        alphas: Tuple[float, ...] = (15.0, 50.0, 150.0),
    ) -> None:
        """Validation-selected Ridge anchor ONLY for small-sample 9-day dynamic split D1 (ZERO Test access)."""
        if self.wo_dual_stream:
            return
        self.eval()

        def collect(idx_g: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            feats_list, y_list = [], []
            for s in range(0, len(idx_g), 512):
                b_idx = idx_g[s : s + 512]
                batch = slicer.get_batch(b_idx, need_vmd=False)
                th_mean, th_stat_shared, th_grad_shared, _, _, wind, op, _, _ = self._prepare_inputs(batch["x_seq"])
                hom_in_full = torch.cat([th_mean, th_stat_shared, th_grad_shared, wind, op], dim=-1)
                hom_c, w_c, dt_c = self.compressor(hom_in_full, batch["w_pos_seq"], batch["dt_seq"])
                hom_out, hom_h = self.hom_ssm(hom_c, w_c, dt_c)
                h_hom_now = hom_out[:, -1, :]
                idx_30m = max(0, hom_out.shape[1] - 31)
                h_hom_diff = hom_h[:, -1, :] - hom_h[:, idx_30m, :]
                hom_feat_t = torch.cat([h_hom_now, h_hom_diff, hom_in_full[:, -1, :]], dim=-1)
                feats_list.append(hom_feat_t.double().cpu())
                y_list.append(batch["y_norm"].double().cpu())
            return torch.cat(feats_list, dim=0), torch.cat(y_list, dim=0)

        phi_tr, y_tr = collect(train_idx_gpu)
        phi_va, y_va = collect(val_idx_gpu)
        aug_tr = torch.cat([phi_tr, torch.ones(phi_tr.shape[0], 1, dtype=torch.float64)], dim=1)
        aug_va = torch.cat([phi_va, torch.ones(phi_va.shape[0], 1, dtype=torch.float64)], dim=1)
        gram = aug_tr.t() @ aug_tr
        rhs = aug_tr.t() @ y_tr
        eye = torch.eye(aug_tr.shape[1], dtype=torch.float64)
        eye[-1, -1] = 1e-3

        best_mse, best_theta = float("inf"), None
        for a in alphas:
            th = torch.linalg.solve(gram + a * eye, rhs)
            mse = float(torch.mean((aug_va @ th - y_va) ** 2))
            if mse < best_mse:
                best_mse = mse
                best_theta = th

        w_total = best_theta[:-1, :].t().float().to(self.u_mode.device)
        b_total = best_theta[-1, :].float().to(self.u_mode.device)

        if self.wo_modes:
            self.hom_head_uncoupled.weight.copy_(w_total)
            self.hom_head_uncoupled.bias.copy_(b_total)
        else:
            u_orth, _ = torch.linalg.qr(self.u_mode.data)
            self.u_mode.data.copy_(u_orth)
            w_modes = u_orth.t() @ w_total
            w_fine = w_total - (u_orth @ w_modes)
            nn.init.eye_(self.hom_modal_lin.weight)
            self.hom_to_modes.weight.copy_(w_modes)
            self.hom_fine_diag.weight.copy_(w_fine)
            self.hom_fine_diag.bias.copy_(b_total)

    def orthogonality_loss(self) -> torch.Tensor:
        if self.wo_modes:
            return torch.tensor(0.0, device=self.u_mode.device)
        gram = self.u_mode.t() @ self.u_mode
        eye = torch.eye(self.mode_rank, device=gram.device, dtype=gram.dtype)
        return torch.mean((gram - eye) ** 2)

    def _prepare_inputs(self, x_seq: torch.Tensor):
        th_mean = x_seq[:, :, self.idx_th_mean]
        th_stat_raw = x_seq[:, :, self.idx_th_stat]
        th_grad_raw = x_seq[:, :, self.idx_th_grad]

        if self.num_targets > 2 and (not self.wo_spatial_grad):
            # Full PhySSM Subsystem-Decoupled Thermal Routing:
            # Shared trunk (including 13 elevated BUS strains) receives collinearity-free isotropic bulk bath,
            # while pedestal/alidade axis bearings (49#01/02) additionally receive 4-mast spatial gradients!
            th_stat_shared = th_mean[:, :, :1].expand_as(th_stat_raw)
            th_grad_shared = torch.zeros_like(th_grad_raw)
            use_pedestal_grad = True
        elif self.num_targets == 2 and (not self.wo_spatial_grad):
            # 2-channel axis-only split (D1): both targets are pedestal axis bearings
            th_stat_shared = th_stat_raw
            th_grad_shared = th_grad_raw
            use_pedestal_grad = True
        else:
            # Ablation `wo_spatial_grad`: Naive unrouted broadcast of ground-mast gradients into shared 15-ch trunk
            # without dedicated pedestal differential tilt routing (tests necessity of subsystem thermal decoupling!)
            th_stat_shared = th_stat_raw if self.num_targets > 2 else th_mean[:, :, :1].expand_as(th_stat_raw)
            th_grad_shared = th_grad_raw if self.num_targets > 2 else torch.zeros_like(th_grad_raw)
            use_pedestal_grad = False

        wind_raw = x_seq[:, :, self.idx_wind]
        qmask = x_seq[:, :, self.idx_qmask]
        if not self.wo_gate:
            wind = torch.cat([wind_raw[:, :, :-2], wind_raw[:, :, -2:] * qmask], dim=-1)
        else:
            wind = wind_raw

        op = x_seq[:, :, self.idx_op]
        return th_mean, th_stat_shared, th_grad_shared, th_stat_raw, th_grad_raw, wind, op, qmask, use_pedestal_grad

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        horizon_min: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_seq = batch["x_seq"]          # (B, Lw, F)
        w_pos_seq = batch["w_pos_seq"]  # (B, Lw, 4)
        dt_seq = batch["dt_seq"]        # (B, Lw, 1)
        maha_t = batch["maha_t"]        # (B, 1)

        th_mean, th_stat_sh, th_grad_sh, th_stat_raw, th_grad_raw, wind, op, qmask, use_ped = self._prepare_inputs(x_seq)
        hom_in_full = torch.cat([th_mean, th_stat_sh, th_grad_sh, wind, op], dim=-1)
        nl_in_full = torch.cat([hom_in_full, qmask], dim=-1)

        hom_c, w_c, dt_c = self.compressor(hom_in_full, w_pos_seq, dt_seq)
        nl_c, _, _ = self.compressor(nl_in_full, w_pos_seq, dt_seq)

        hom_out, hom_h = self.hom_ssm(hom_c, w_c, dt_c)
        h_nl = self.in_stem(nl_c)
        for blk in self.nl_blocks:
            h_nl, _ = blk(h_nl, w_c, dt_c)

        h_hom_now = hom_out[:, -1, :]
        idx_30m = max(0, hom_out.shape[1] - 31)
        h_hom_diff = hom_h[:, -1, :] - hom_h[:, idx_30m, :]

        h_nl_now = h_nl[:, -1, :]
        h_nl_ctx = h_nl[:, -15:, :].mean(dim=1)

        hom_inst_t = hom_in_full[:, -1, :]
        hom_feat_t = torch.cat([h_hom_now, h_hom_diff, hom_inst_t], dim=-1)

        op_now = op[:, -1, :]
        op_recent_mean = op[:, -5:, :].mean(dim=1)
        op_jump = op_now - op_recent_mean

        maha_ratio = maha_t / max(self.maha_p99_train, 1.0)
        far_oos_damp = torch.exp(-0.25 * F.relu(maha_ratio - 2.0))

        if self.wo_dual_stream:
            mu_pred = self.unconstrained_decoder(torch.cat([h_nl_now, h_nl_ctx, hom_inst_t], dim=-1))
        else:
            if self.wo_modes:
                mu_hom = self.hom_head_uncoupled(hom_feat_t)
            else:
                m_coord = self.hom_modal_lin(self.hom_to_modes(hom_feat_t))
                mu_hom = (m_coord @ self.u_mode.t()) + self.hom_fine_diag(hom_feat_t)

            mu_act = self.actuation_elastic_net(torch.cat([op_now, op_jump, self.op_proj(h_nl_now)], dim=-1))

            env_res_in = torch.cat([h_nl_now, h_nl_ctx, th_grad_sh[:, -1, :], wind[:, -1, :]], dim=-1)
            env_h = self.env_res_mlp(env_res_in) * torch.clamp(self.res_lip_scale, 0.5, 4.0)
            if self.wo_modes:
                r_env = self.res_head_uncoupled(env_h)
            else:
                m_res = self.res_to_modes(env_h)
                m_res = m_res + 0.5 * self.modal_graph_mix(m_res)
                r_env = (m_res @ self.u_mode.t()) + self.res_fine_diag(env_h)

            rg = self.res_stream_gain
            if self.wo_gate:
                mu_pred = mu_hom + rg * (mu_act + r_env)
            else:
                g_raw = self.regime_gate(torch.cat([op_now, qmask[:, -1, :], maha_ratio], dim=-1))
                g_act, g_env = torch.sigmoid(g_raw).chunk(2, dim=-1)
                mu_pred = mu_hom + rg * (g_act * mu_act + (g_env * far_oos_damp) * r_env)

        # Subsystem-Decoupled Pedestal Differential Tilt Contribution on Axis Bearings (49#01/02)
        if use_ped:
            ped_in = torch.cat([th_stat_raw, 0.35 * th_grad_raw, wind, op], dim=-1)
            ped_c, _, _ = self.compressor(ped_in, w_pos_seq, dt_seq)
            ped_out, _ = self.pedestal_ssm(ped_c, w_c, dt_c)
            ped_delta = self.pedestal_tilt_head(torch.cat([ped_out[:, -1, :], 0.35 * th_grad_raw[:, -1, :]], dim=-1))
            ped_delta = self.res_stream_gain * far_oos_damp * ped_delta
            if self.num_targets > 2:
                pad_zeros = torch.zeros((mu_pred.shape[0], self.num_targets - 2), device=mu_pred.device, dtype=mu_pred.dtype)
                mu_pred = mu_pred + torch.cat([ped_delta, pad_zeros], dim=-1)
            else:
                mu_pred = mu_pred + 0.25 * ped_delta

        # Support-Aware Predictive Uncertainty (Transient-Aware Aleatoric + Mahalanobis Epistemic)
        alea_in = torch.cat([h_nl_now.detach(), op_now, torch.abs(op_jump), wind[:, -1, :]], dim=-1)
        alea_var = F.softplus(self.alea_head(alea_in)) + 1e-4
        maha_scaled = torch.clamp(maha_t / max(self.maha_p99_train, 1.0), min=0.0, max=30.0)
        epist_var = F.softplus(self.epistemic_lambda_raw).unsqueeze(0) * maha_scaled
        sigma_pred = torch.sqrt(alea_var + epist_var + 1e-6)

        return mu_pred, sigma_pred

    @torch.no_grad()
    def export_learned_physics(self) -> Dict[str, np.ndarray]:
        """Export end-to-end learned physical relaxation time constants (minutes) and orthogonal modes."""
        taus_hom = self.hom_ssm.get_base_taus().detach().cpu().numpy()
        taus_nl = [blk.get_base_taus().detach().cpu().numpy() for blk in self.nl_blocks]
        d_hom = len(taus_hom)

        if not self.wo_modes:
            u_orth, _ = torch.linalg.qr(self.u_mode.data)
            w_modal = u_orth @ self.hom_modal_lin.weight @ self.hom_to_modes.weight[:, :d_hom]
            w_total = (w_modal + self.hom_fine_diag.weight[:, :d_hom]).detach().cpu().numpy()
            u_exp = u_orth.detach().cpu().numpy()
        else:
            w_total = self.hom_head_uncoupled.weight[:, :d_hom].detach().cpu().numpy()
            u_exp = self.u_mode.detach().cpu().numpy()

        abs_w = np.abs(w_total) + 1e-8
        abs_w = abs_w / np.sum(abs_w, axis=1, keepdims=True)
        channel_eff_tau_min = abs_w @ taus_hom

        device = self.u_mode.device
        w_grid = torch.linspace(0.0, 2.0, 25, device=device).unsqueeze(1).repeat(1, 4)
        kappa_grid = self.hom_ssm.compute_wind_kappa(w_grid).detach().cpu().numpy()
        convective_speedup = 1.0 + np.mean(kappa_grid, axis=1)

        return {
            "taus_hom1_min": taus_hom,
            "taus_hom2_min": taus_hom,
            "taus_nl_layers_min": np.stack(taus_nl, axis=0),
            "channel_effective_tau_min": channel_eff_tau_min,
            "u_mode": u_exp,
            "wind_norm_grid": np.linspace(0.0, 2.0, 25),
            "convective_speedup_factor": convective_speedup,
        }
