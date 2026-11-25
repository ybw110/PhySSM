"""Zero-leakage causal sliding-window data pipeline for Paper 1 (Splits M1, M2-F1..3, D1 & Multi-Horizon)."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class SplitBundle:
    """Container holding leak-free normalized contiguous segments, index maps, scalers, and physical priors."""
    split_name: str
    horizon_min: int
    lookback: int
    feature_names: List[str]
    group_slices: Dict[str, List[int]]
    target_names: List[str]

    # Contiguous timeline arrays (padded by `lookback - 1` at segment starts for edge-safe indexing)
    x_norm_Pad: np.ndarray          # (N_pad, F) float32
    wind_pos_Pad: np.ndarray        # (N_pad, 4) non-negative wind magnitude features >= 0
    dt_min_Pad: np.ndarray          # (N_pad, 1) step size in minutes
    maha_sq_Pad: np.ndarray         # (N_pad, 1) Train-fitted environmental Mahalanobis distance^2
    y_norm_All: np.ndarray          # (N_pad, C) float32
    y_phys_All: np.ndarray          # (N_pad, C) float64 physical units

    # Window end-step indices `t_idx` into `*_Pad` (causal window is [t_idx - lookback + 1 : t_idx + 1], target at t_idx + horizon_min)
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray

    # Scalers (fitted strictly on Train)
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    maha_p99_train: float
    u_svd_init: np.ndarray          # (C, C) orthogonal SVD columns fitted strictly on Train targets

    # Metadata & regime masks aligned with `test_indices`
    test_dates: np.ndarray
    test_timestamps: np.ndarray
    test_temp_mean: np.ndarray
    test_wind_speed: np.ndarray
    test_temp_range: np.ndarray
    test_regimes: Dict[str, np.ndarray]

    # Auxiliary Train-fitted environmental & wind scaling stats for standalone checkpoint inference
    env_cols: Optional[List[str]] = None
    env_mu: Optional[np.ndarray] = None
    env_cov_inv: Optional[np.ndarray] = None
    wind_pos_cols: Optional[List[str]] = None
    wind_pos_scale: Optional[np.ndarray] = None


class CausalWindowDataset(Dataset):
    """Fast zero-copy GPU/CPU causal window dataset over contiguous padded arrays."""

    def __init__(self, bundle: SplitBundle, indices: np.ndarray, vmd_imfs_Pad: Optional[np.ndarray] = None):
        self.bundle = bundle
        self.indices = indices.astype(np.int64)
        self.lw = bundle.lookback
        self.h = bundle.horizon_min
        self.x = bundle.x_norm_Pad
        self.w_pos = bundle.wind_pos_Pad
        self.dt = bundle.dt_min_Pad
        self.maha = bundle.maha_sq_Pad
        self.y_norm = bundle.y_norm_All
        self.vmd = vmd_imfs_Pad

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        t = int(self.indices[i])
        s = t - self.lw + 1
        e = t + 1
        item = {
            "x_seq": torch.from_numpy(self.x[s:e]),           # (Lw, F)
            "w_pos_seq": torch.from_numpy(self.w_pos[s:e]),   # (Lw, 4)
            "dt_seq": torch.from_numpy(self.dt[s:e]),         # (Lw, 1)
            "maha_t": torch.from_numpy(self.maha[t]),         # (1,)
            "y_norm": torch.from_numpy(self.y_norm[t + self.h]),  # (C,)
        }
        if self.vmd is not None:
            item["vmd_seq"] = torch.from_numpy(self.vmd[s:e])  # (Lw, K_vmd)
        return item


def compute_causal_multiscale_vmd_proxy(
    df_seg: pd.DataFrame,
    x_mean_map: Dict[str, float],
    x_std_map: Dict[str, float],
) -> np.ndarray:
    """Strictly causal 3-band variational filterbank decomposition (trend / mesoscale / gust-transient)
    over key exogenous drivers (`temp_mean`, `temp_spatial_range`, `wind_12_14_speed`, `wind_q_total_Pa`)
    for VMD-BiLSTM (Wu et al., Heliyon 2024) with ZERO future leakage across time `t`.
    """
    drivers = ["temp_mean", "temp_spatial_range", "wind_12_14_speed", "wind_q_total_Pa"]
    bands = []
    for col in drivers:
        raw = df_seg[col].to_numpy(dtype=np.float64)
        norm = (raw - x_mean_map[col]) / x_std_map[col]
        s_pd = pd.Series(norm)
        # Strictly causal exponential / variational Wiener sub-bands:
        # IMF-1: slow diurnal/synoptic thermal-wind mode (tau ~ 90 min)
        low = s_pd.ewm(span=120, adjust=False).mean().to_numpy()
        # IMF-2: intermediate mesoscale convective band (tau ~ 15..90 min)
        mid_lp = s_pd.ewm(span=18, adjust=False).mean().to_numpy()
        mid = mid_lp - low
        # IMF-3: fast turbulent gust / transient band (< 15 min)
        high = norm - mid_lp
        bands.extend([low, mid, high])
    return np.stack(bands, axis=-1).astype(np.float32)  # (N_seg, 12)


def load_split_bundle(
    config_path: Path,
    data_root: Path,
    split_name: str = "M1",
    horizon_min: int = 0,
    lookback_override: Optional[int] = None,
) -> Tuple[SplitBundle, np.ndarray]:
    """Build strictly leak-free SplitBundle and causal 3-band VMD array for any split in {M1, M2_fold1..3, D1}."""
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    lookback = int(lookback_override if lookback_override is not None else cfg["lookback_minutes"])

    is_2025 = (split_name == "D1")
    if is_2025:
        csv_path = data_root / "processed_2025_1min" / "antenna_2025_15d_1min_canonical.csv"
        split_col = "split_D1"
        target_names = list(cfg["targets_2025_2ch"])
    else:
        csv_path = data_root / "processed_2026_1min" / "antenna_2026_45d_1min_canonical.csv"
        split_col = f"split_{split_name}" if not split_name.startswith("split_") else split_name
        target_names = list(cfg["targets_2026_15ch"])

    df = pd.read_csv(csv_path)
    df = df[df["raw_count"] >= 100].copy()
    df["ts_dt"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("ts_dt").reset_index(drop=True)

    # Assemble ordered exogenous feature list and group index slices
    f_groups = cfg["feature_groups"]
    ordered_groups = [
        "thermal_mean_aux",
        "thermal_stations",
        "thermal_spatial_grad",
        "wind_forcing",
        "actuation_and_attitude",
        "quality_masks",
    ]
    feature_names: List[str] = []
    group_slices: Dict[str, List[int]] = {}
    for g in ordered_groups:
        idxs = []
        for col in f_groups[g]:
            idxs.append(len(feature_names))
            feature_names.append(col)
        group_slices[g] = idxs

    # Filter rows participating in this split (`train`, `val`, `test`)
    df = df[df[split_col].isin(["train", "val", "test"])].reset_index(drop=True)
    train_df = df[df[split_col] == "train"]
    if len(train_df) == 0:
        raise ValueError(f"Empty train split for {split_name}!")

    # Fit exogenous scalers STRICTLY on Train
    zero_preserving_cols = set(f_groups["thermal_spatial_grad"]) | {
        "motor_active_flag",
        "wind_dir_quality_mask",
        "mode30_frac",
        "mode34_frac",
        "el_sin",
        "el_cos",
        "az_sin",
        "az_cos",
    }
    identity_scale_cols = {
        "motor_active_flag",
        "wind_dir_quality_mask",
        "mode30_frac",
        "mode34_frac",
        "el_sin",
        "el_cos",
        "az_sin",
        "az_cos",
    }

    x_mean = np.zeros(len(feature_names), dtype=np.float64)
    x_std = np.ones(len(feature_names), dtype=np.float64)
    for i, col in enumerate(feature_names):
        vals = train_df[col].to_numpy(dtype=np.float64)
        if col in identity_scale_cols:
            x_mean[i] = 0.0
            x_std[i] = 1.0
        elif col in zero_preserving_cols:
            x_mean[i] = 0.0
            x_std[i] = max(float(np.sqrt(np.mean(vals ** 2))), 1e-3)
        else:
            x_mean[i] = float(np.mean(vals))
            x_std[i] = max(float(np.std(vals)), 1e-3)

    x_mean_map = {c: float(x_mean[i]) for i, c in enumerate(feature_names)}
    x_std_map = {c: float(x_std[i]) for i, c in enumerate(feature_names)}

    # Fit target scalers & SVD orthogonal structural mode basis STRICTLY on Train
    y_train_phys = train_df[target_names].to_numpy(dtype=np.float64)
    y_mean = np.mean(y_train_phys, axis=0)
    y_std = np.maximum(np.std(y_train_phys, axis=0), 1e-5)
    y_train_norm = (y_train_phys - y_mean) / y_std
    # SVD on clean-standby (or all train if few clean) for pure structural deformation modes
    clean_tr = train_df["is_clean_standby"].to_numpy(dtype=bool)
    svd_src = y_train_norm[clean_tr] if clean_tr.sum() >= 500 else y_train_norm
    _, _, vt = np.linalg.svd(svd_src - np.mean(svd_src, axis=0, keepdims=True), full_matrices=True)
    u_svd_init = vt.T.astype(np.float32)  # (C, C) columns are top singular vectors

    # Fit environmental Mahalanobis support metric STRICTLY on Train
    env_cols = ["temp_mean", "temp_grad_06", "temp_grad_11", "temp_grad_13", "wind_12_14_speed"]
    env_tr = train_df[env_cols].to_numpy(dtype=np.float64)
    env_mu = np.mean(env_tr, axis=0)
    env_cov = np.cov(env_tr, rowvar=False) + np.eye(len(env_cols)) * 1e-4
    env_cov_inv = np.linalg.inv(env_cov)
    diff_tr = env_tr - env_mu
    maha_tr = np.einsum("ni,ij,nj->n", diff_tr, env_cov_inv, diff_tr)
    maha_p99_train = float(np.percentile(maha_tr, 99.0))
    temp_p95_train = float(np.percentile(train_df["temp_mean"], 95.0))

    dev_df = df[df[split_col].isin(["train", "val"])]
    dev_temp_max = float(dev_df["temp_mean"].max())
    dev_temp_min = float(dev_df["temp_mean"].min())

    # Partition into contiguous calendar segments (break if gap > 15 min, e.g., 2025-10-21)
    dt_raw = df["ts_dt"].diff().dt.total_seconds().div(60.0).fillna(1.0).to_numpy(dtype=np.float64)
    seg_breaks = np.where(dt_raw > 15.0)[0]
    seg_starts = [0] + seg_breaks.tolist()
    seg_ends = seg_breaks.tolist() + [len(df)]

    wind_pos_cols = ["wind_12_14_speed", "wind_speed_mean24", "wind_speed_p90_24", "wind_q_total_Pa"]
    wind_pos_scale = np.array([max(float(np.percentile(train_df[c], 90.0)), 0.5) for c in wind_pos_cols], dtype=np.float64)

    x_chunks, wpos_chunks, dt_chunks, maha_chunks, yn_chunks, yp_chunks, vmd_chunks = [], [], [], [], [], [], []
    train_idx_list, val_idx_list, test_idx_list = [], [], []
    test_meta_rows = []

    pad_len = lookback - 1
    offset = 0

    for s_idx, e_idx in zip(seg_starts, seg_ends):
        seg = df.iloc[s_idx:e_idx].reset_index(drop=True)
        n_seg = len(seg)

        x_seg = (seg[feature_names].to_numpy(dtype=np.float64) - x_mean) / x_std
        wpos_seg = np.maximum(seg[wind_pos_cols].to_numpy(dtype=np.float64), 0.0) / wind_pos_scale
        dt_seg = seg["ts_dt"].diff().dt.total_seconds().div(60.0).fillna(1.0).clip(0.5, 15.0).to_numpy(dtype=np.float64)[:, None]

        env_seg = seg[env_cols].to_numpy(dtype=np.float64) - env_mu
        maha_seg = np.einsum("ni,ij,nj->n", env_seg, env_cov_inv, env_seg)[:, None]

        yp_seg = seg[target_names].to_numpy(dtype=np.float64)
        yn_seg = (yp_seg - y_mean) / y_std
        vmd_seg = compute_causal_multiscale_vmd_proxy(seg, x_mean_map, x_std_map)

        # Edge-replicate first row `pad_len` times ONLY for segment-start boundary safety
        x_pad = np.vstack([np.repeat(x_seg[:1], pad_len, axis=0), x_seg]).astype(np.float32)
        wpos_pad = np.vstack([np.repeat(wpos_seg[:1], pad_len, axis=0), wpos_seg]).astype(np.float32)
        dt_pad = np.vstack([np.ones((pad_len, 1), dtype=np.float64), dt_seg]).astype(np.float32)
        maha_pad = np.vstack([np.repeat(maha_seg[:1], pad_len, axis=0), maha_seg]).astype(np.float32)
        yp_pad = np.vstack([np.repeat(yp_seg[:1], pad_len, axis=0), yp_seg]).astype(np.float64)
        yn_pad = np.vstack([np.repeat(yn_seg[:1], pad_len, axis=0), yn_seg]).astype(np.float32)
        vmd_pad = np.vstack([np.repeat(vmd_seg[:1], pad_len, axis=0), vmd_seg]).astype(np.float32)

        splits_seg = seg[split_col].to_numpy()
        max_valid_i = n_seg - horizon_min
        for i in range(max_valid_i):
            sp = splits_seg[i]
            sp_target = splits_seg[i + horizon_min]
            # Ensure target at t + H stays inside the exact same split!
            if sp != sp_target:
                continue
            global_pos = offset + pad_len + i
            if sp == "train":
                # Require full unpadded real lookback for training samples
                if i >= lookback - 1:
                    train_idx_list.append(global_pos)
            elif sp == "val":
                val_idx_list.append(global_pos)
            elif sp == "test":
                test_idx_list.append(global_pos)
                test_meta_rows.append((seg.iloc[i + horizon_min], float(maha_seg[i + horizon_min, 0])))

        x_chunks.append(x_pad)
        wpos_chunks.append(wpos_pad)
        dt_chunks.append(dt_pad)
        maha_chunks.append(maha_pad)
        yn_chunks.append(yn_pad)
        yp_chunks.append(yp_pad)
        vmd_chunks.append(vmd_pad)
        offset += len(x_pad)

    test_df_rows = pd.DataFrame([r[0] for r in test_meta_rows]).reset_index(drop=True)
    test_maha_arr = np.array([r[1] for r in test_meta_rows], dtype=np.float64)
    test_temp = test_df_rows["temp_mean"].to_numpy(dtype=np.float64)

    if split_name == "M2_fold1":
        oos_mask = test_temp > dev_temp_max
    elif split_name == "D1":
        oos_mask = test_temp < dev_temp_min
    else:
        oos_mask = (test_temp > temp_p95_train) | (test_maha_arr > maha_p99_train)

    test_regimes = {
        "overall": np.ones(len(test_df_rows), dtype=bool),
        "in_support": ~oos_mask,
        "oos_tail": oos_mask,
        "clean_standby": test_df_rows["is_clean_standby"].to_numpy(dtype=bool),
        "motor_active": test_df_rows["motor_active_flag"].to_numpy(dtype=bool),
    }

    bundle = SplitBundle(
        split_name=split_name,
        horizon_min=horizon_min,
        lookback=lookback,
        feature_names=feature_names,
        group_slices=group_slices,
        target_names=target_names,
        x_norm_Pad=np.concatenate(x_chunks, axis=0),
        wind_pos_Pad=np.concatenate(wpos_chunks, axis=0),
        dt_min_Pad=np.concatenate(dt_chunks, axis=0),
        maha_sq_Pad=np.concatenate(maha_chunks, axis=0),
        y_norm_All=np.concatenate(yn_chunks, axis=0),
        y_phys_All=np.concatenate(yp_chunks, axis=0),
        train_indices=np.array(train_idx_list, dtype=np.int64),
        val_indices=np.array(val_idx_list, dtype=np.int64),
        test_indices=np.array(test_idx_list, dtype=np.int64),
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        maha_p99_train=maha_p99_train,
        u_svd_init=u_svd_init,
        test_dates=test_df_rows["date"].to_numpy(),
        test_timestamps=test_df_rows["timestamp"].to_numpy(),
        test_temp_mean=test_temp,
        test_wind_speed=test_df_rows["wind_12_14_speed"].to_numpy(dtype=np.float64),
        test_temp_range=test_df_rows["temp_spatial_range"].to_numpy(dtype=np.float64),
        test_regimes=test_regimes,
        env_cols=list(env_cols),
        env_mu=env_mu.astype(np.float64),
        env_cov_inv=env_cov_inv.astype(np.float64),
        wind_pos_cols=list(wind_pos_cols),
        wind_pos_scale=wind_pos_scale.astype(np.float64),
    )
    vmd_all = np.concatenate(vmd_chunks, axis=0)
    return bundle, vmd_all
