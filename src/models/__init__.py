"""Unified Model Registry for Paper 1 (7 Literature Baselines + PhySSM)."""

from typing import Union
import torch.nn as nn

from ..data_pipeline import SplitBundle
from .physsm import PhySSM
from .traditional_and_sysid import CompositeGPRModel, N4SIDSubspaceModel
from .industrial_dl import VMDBiLSTM, CausalDilatedTCN, STGAT
from .frontier_ts import TimeXer, TimePro

NON_TORCH_MODELS = {"Composite-GPR", "N4SID"}


def build_model(
    model_name: str,
    bundle: SplitBundle,
    seed: int = 42,
    ablation: str = "none",
    mode_rank: int = 5,
) -> Union[nn.Module, CompositeGPRModel, N4SIDSubspaceModel]:
    in_dim = len(bundle.feature_names)
    num_targets = len(bundle.target_names)
    lw = bundle.lookback

    if model_name == "Composite-GPR":
        return CompositeGPRModel(num_inducing=1800, seed=seed)
    if model_name == "N4SID":
        return N4SIDSubspaceModel(horizon_blocks=12, downsample_ident=3)
    if model_name == "VMD-BiLSTM":
        return VMDBiLSTM(in_dim=in_dim, vmd_dim=12, num_targets=num_targets, hidden_dim=96, num_layers=2)
    if model_name == "Causal-Dilated TCN":
        return CausalDilatedTCN(in_dim=in_dim, num_targets=num_targets, channels=80, kernel_size=5)
    if model_name == "ST-GAT":
        return STGAT(in_dim=in_dim, num_targets=num_targets, node_dim=64, patch_stride=12)
    if model_name == "TimeXer":
        return TimeXer(in_dim=in_dim, lookback=lw, num_targets=num_targets, patch_len=12, d_model=96, num_layers=3)
    if model_name == "TimePro":
        return TimePro(in_dim=in_dim, lookback=lw, num_targets=num_targets, patch_len=6, d_model=96, num_layers=3)
    if model_name == "PhySSM":
        return PhySSM(
            group_slices=bundle.group_slices,
            num_targets=num_targets,
            u_svd_init=bundle.u_svd_init,
            maha_p99_train=bundle.maha_p99_train,
            hidden_dim=96,
            hom_dim=64,
            num_layers=3,
            mode_rank=mode_rank,
            eq_init_steps=4,
            ablation=ablation,
        )
    raise ValueError(f"Unknown model_name: {model_name}")
