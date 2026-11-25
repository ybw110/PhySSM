"""Zero-leakage metrics (RMSE, MAE, exact Gaussian CRPS), Normalized Split-Conformal Calibration, and Multi-Seed Day-Block Bootstrap."""

import math
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from scipy.optimize import minimize_scalar

INV_SQRT_2 = 1.0 / math.sqrt(2.0)
INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
INV_SQRT_PI = 1.0 / math.sqrt(math.pi)


def gaussian_crps_torch(mu: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Exact analytical Closed-Form Gaussian CRPS (Gneiting & Raftery, JASA 2007) in PyTorch."""
    sig = torch.clamp(sigma, min=1e-5)
    z = (target - mu) / sig
    two_phi_minus_one = torch.erf(z * INV_SQRT_2)
    pdf = INV_SQRT_2PI * torch.exp(-0.5 * z * z)
    return sig * (z * two_phi_minus_one + 2.0 * pdf - INV_SQRT_PI)


def gaussian_crps_numpy(mu: np.ndarray, sigma: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Vectorized exact analytical Gaussian CRPS in NumPy (same physical units as target)."""
    from scipy.special import erf
    sig = np.maximum(sigma, 1e-6)
    z = (target - mu) / sig
    two_phi_minus_one = erf(z * INV_SQRT_2)
    pdf = INV_SQRT_2PI * np.exp(-0.5 * np.clip(z * z, 0.0, 80.0))
    return sig * (z * two_phi_minus_one + 2.0 * pdf - INV_SQRT_PI)


def calibrate_gaussian_sigma(
    val_mu: np.ndarray,
    val_sigma_raw: Optional[np.ndarray],
    val_target: np.ndarray,
) -> Tuple[np.ndarray, str]:
    """Calibrate continuous predictive scale `sigma` strictly on Validation set (ZERO Test access)."""
    n_samples, n_ch = val_target.shape
    scales = np.ones(n_ch, dtype=np.float64)

    if val_sigma_raw is None:
        resid = val_target - val_mu
        for c in range(n_ch):
            std_hat = float(np.sqrt(np.mean(resid[:, c] ** 2)))
            std_hat = max(std_hat, 1e-5)

            def obj(s: float) -> float:
                sig_c = np.full(n_samples, max(s, 1e-6), dtype=np.float64)
                return float(np.mean(gaussian_crps_numpy(val_mu[:, c], sig_c, val_target[:, c])))

            res = minimize_scalar(obj, bounds=(0.2 * std_hat, 3.0 * std_hat), method="bounded")
            scales[c] = float(res.x) if res.success else std_hat
        return scales, "homoscedastic_val_residual"

    for c in range(n_ch):
        raw_c = np.maximum(val_sigma_raw[:, c].astype(np.float64), 1e-6)

        def obj(mult: float) -> float:
            return float(np.mean(gaussian_crps_numpy(val_mu[:, c], mult * raw_c, val_target[:, c])))

        res = minimize_scalar(obj, bounds=(0.15, 6.0), method="bounded")
        scales[c] = float(res.x) if res.success else 1.0
    return scales, "heteroscedastic_val_scaled"


def apply_sigma_calibration(
    test_mu: np.ndarray,
    test_sigma_raw: Optional[np.ndarray],
    scales: np.ndarray,
    mode: str,
) -> np.ndarray:
    """Apply Validation-fitted uncertainty scale to Test predictions."""
    n_samples, n_ch = test_mu.shape
    if mode == "homoscedastic_val_residual" or test_sigma_raw is None:
        return np.broadcast_to(scales.reshape(1, n_ch), (n_samples, n_ch)).copy()
    return np.maximum(test_sigma_raw * scales.reshape(1, n_ch), 1e-6)


def calibrate_conformal_q95(
    val_mu: np.ndarray,
    val_sigma_cal: np.ndarray,
    val_target: np.ndarray,
    alpha: float = 0.05,
) -> np.ndarray:
    """Normalized Split-Conformal Prediction (SCP, Lei et al., JASA 2018; Romano et al., NeurIPS 2019).

    Computes exact finite-sample validation nonconformity quantile q_{1-alpha, c} of |y - mu| / sigma_cal
    strictly on Validation set (ZERO Test access), applied identically to ALL baselines and PhySSM.
    """
    n_val, n_ch = val_target.shape
    level = min(1.0, math.ceil((1.0 - alpha) * (n_val + 1)) / float(max(n_val, 1)))
    scores = np.abs(val_target - val_mu) / np.maximum(val_sigma_cal, 1e-6)
    q95 = np.quantile(scores, level, axis=0).astype(np.float64)
    return np.maximum(q95, 0.5)


def compute_channel_and_group_metrics(
    mu_phys: np.ndarray,
    sigma_phys: np.ndarray,
    target_phys: np.ndarray,
    target_names: List[str],
    mask: Optional[np.ndarray] = None,
    conformal_q95: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute RMSE, MAE, CRPS in physical units, plus Split-Conformal 95% PI coverage & width."""
    if mask is not None:
        idx = np.asarray(mask, dtype=bool)
        if idx.sum() == 0:
            return {"n_minutes": 0}
        mu_p = mu_phys[idx]
        sig_p = sigma_phys[idx]
        tgt_p = target_phys[idx]
    else:
        mu_p = mu_phys
        sig_p = sigma_phys
        tgt_p = target_phys

    err = mu_p - tgt_p
    rmse_ch = np.sqrt(np.mean(err ** 2, axis=0))
    mae_ch = np.mean(np.abs(err), axis=0)
    crps_ch = np.mean(gaussian_crps_numpy(mu_p, sig_p, tgt_p), axis=0)

    # Fair Normalized Split-Conformal 95% PI (or Gaussian 1.96 fallback if conformal_q95 omitted)
    q_mult = conformal_q95.reshape(1, -1) if conformal_q95 is not None else np.full((1, len(target_names)), 1.959964)
    half_width = q_mult * sig_p
    cov95_ch = np.mean(np.abs(err) <= half_width, axis=0)
    piw95_ch = np.mean(2.0 * half_width, axis=0)

    disp_idx = [i for i, n in enumerate(target_names) if n.startswith("disp_")]
    strain_idx = [i for i, n in enumerate(target_names) if n.startswith("strain_")]

    out: Dict[str, float] = {"n_minutes": int(len(tgt_p))}
    if disp_idx:
        out["disp_RMSE_mm"] = float(np.mean(rmse_ch[disp_idx]))
        out["disp_MAE_mm"] = float(np.mean(mae_ch[disp_idx]))
        out["disp_CRPS_mm"] = float(np.mean(crps_ch[disp_idx]))
        out["disp_PICP95"] = float(np.mean(cov95_ch[disp_idx]))
        out["disp_MPIW95_mm"] = float(np.mean(piw95_ch[disp_idx]))
    if strain_idx:
        out["strain_RMSE_ue"] = float(np.mean(rmse_ch[strain_idx]))
        out["strain_MAE_ue"] = float(np.mean(mae_ch[strain_idx]))
        out["strain_CRPS_ue"] = float(np.mean(crps_ch[strain_idx]))
        out["strain_PICP95"] = float(np.mean(cov95_ch[strain_idx]))
        out["strain_MPIW95_ue"] = float(np.mean(piw95_ch[strain_idx]))

    for i, name in enumerate(target_names):
        out[f"ch_{name}_RMSE"] = float(rmse_ch[i])
        out[f"ch_{name}_MAE"] = float(mae_ch[i])
        out[f"ch_{name}_CRPS"] = float(crps_ch[i])

    return out


def day_block_bootstrap_pvalue(
    dates: np.ndarray,
    err_sq_ref: np.ndarray,
    err_sq_cand: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
    is_rmse: bool = True,
) -> Dict[str, float]:
    """Paired Day-Block Bootstrap test preserving full intra-day autocorrelation structure.

    Supports both squared-error arrays (`is_rmse=True`, comparing root-mean-square) and direct loss
    arrays such as per-minute CRPS (`is_rmse=False`, comparing mean CRPS).
    Tests H0: Metric(cand) <= Metric(ref) vs H1: Metric(ref) < Metric(cand), where `ref` is PhySSM.
    """
    unique_days = np.unique(dates)
    n_days = len(unique_days)
    day_to_idx = [np.where(dates == d)[0] for d in unique_days]
    arr_ref = np.asarray(err_sq_ref, dtype=np.float64)
    arr_cand = np.asarray(err_sq_cand, dtype=np.float64)
    if arr_ref.ndim == 1:
        arr_ref = arr_ref[:, None, None]
    elif arr_ref.ndim == 2:
        arr_ref = arr_ref[:, None, :]
    if arr_cand.ndim == 1:
        arr_cand = arr_cand[:, None, None]
    elif arr_cand.ndim == 2:
        arr_cand = arr_ref[:, :0, :] if arr_cand.size == 0 else arr_cand[:, None, :]

    # Pre-aggregate per-day sums (n_days, K_seeds, C_ch) for exact Table-1 metric block resampling
    day_sum_ref = np.stack([np.sum(arr_ref[idx], axis=0) for idx in day_to_idx], axis=0)
    day_sum_cand = np.stack([np.sum(arr_cand[idx], axis=0) for idx in day_to_idx], axis=0)
    day_cnt = np.array([len(idx) for idx in day_to_idx], dtype=np.float64)

    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, n_days, size=(n_boot, n_days))
    tot_cnt = np.sum(day_cnt[sampled], axis=1)[:, None, None]
    mean_ref = np.sum(day_sum_ref[sampled], axis=1) / tot_cnt
    mean_cand = np.sum(day_sum_cand[sampled], axis=1) / tot_cnt

    if is_rmse:
        val_ref = np.mean(np.sqrt(np.maximum(mean_ref, 0.0)), axis=(1, 2))
        val_cand = np.mean(np.sqrt(np.maximum(mean_cand, 0.0)), axis=(1, 2))
    else:
        val_ref = np.mean(mean_ref, axis=(1, 2))
        val_cand = np.mean(mean_cand, axis=(1, 2))

    diffs = val_cand - val_ref
    p_val = float(np.mean(diffs <= 0.0))
    return {
        "mean_gain": float(np.mean(diffs)),
        "ci95_low": float(np.percentile(diffs, 2.5)),
        "ci95_high": float(np.percentile(diffs, 97.5)),
        "p_value_one_sided": p_val,
    }
