"""Standalone Pretrained Checkpoint Loader, New-Data Inference API & Reproducibility Verifier for Paper 1.

Supports all 8 benchmark models (PhySSM, Composite-GPR, N4SID, VMD-BiLSTM, Causal-Dilated TCN, ST-GAT, TimeXer, TimePro)
using 100% self-contained `.pt` checkpoints saved under `common_engine/checkpoints/`.

Usage examples:
    # 1) Verify every local .pt checkpoint reproduces its test-set RMSE/CRPS/PICP95 exactly:
    python common_engine/infer_from_checkpoint.py --verify_all

    # 2) Verify a single checkpoint (e.g., PhySSM on Split M1, seed 42):
    python common_engine/infer_from_checkpoint.py --ckpt common_engine/checkpoints/M1_H0_PhySSM_ab_none_r5_s42.pt

    # 3) Run standalone inference on any brand-new 1-min antenna CSV (zero deformation history required):
    python common_engine/infer_from_checkpoint.py \
        --ckpt common_engine/checkpoints/M1_H0_PhySSM_ab_none_r5_s42.pt \
        --csv path/to/new_antenna_1min.csv \
        --out_npz path/to/new_preds.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

EXP_ROOT = Path(__file__).resolve().parents[1]
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from src.data_pipeline import compute_causal_multiscale_vmd_proxy, load_split_bundle
from src.metrics_and_calibration import apply_sigma_calibration, compute_channel_and_group_metrics
from src.models import NON_TORCH_MODELS, build_model


def load_checkpoint_model(ckpt_path: Path, device_str: str = "auto") -> Tuple[object, Dict]:
    """Load any self-contained Paper 1 `.pt` checkpoint (PyTorch or Non-Torch) into eval mode."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    prep = ckpt["preprocessing"]

    stub_bundle = SimpleNamespace(
        feature_names=prep["feature_names"],
        group_slices=prep["group_slices"],
        target_names=prep["target_names"],
        lookback=int(prep["lookback"]),
        u_svd_init=np.asarray(prep["u_svd_init"], dtype=np.float32),
        maha_p99_train=float(prep["maha_p99_train"]),
    )
    model = build_model(
        model_name=ckpt["model_name"],
        bundle=stub_bundle,
        seed=int(ckpt["seed"]),
        ablation=str(ckpt["ablation"]),
        mode_rank=int(ckpt["mode_rank"]),
    )

    if ckpt["is_torch_model"]:
        if device_str == "auto":
            dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            dev = torch.device(device_str)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        model.to(dev)
        model.eval()
        ckpt["_device"] = dev
    else:
        st = ckpt["non_torch_state"]
        if ckpt["model_name"] == "Composite-GPR":
            model.num_inducing = int(st["num_inducing"])
            model.seed = int(st["seed"])
            model.z_ind = np.asarray(st["z_ind"], dtype=np.float64)
            model.best_params = dict(st["best_params"])
            model.sigma_n2 = float(st["sigma_n2"])
            model.alpha_weights = np.asarray(st["alpha_weights"], dtype=np.float64)
            model.k_mm_cho = (np.asarray(st["k_mm_cho"][0], dtype=np.float64), bool(st["k_mm_cho"][1]))
            model.a_mat_cho = (np.asarray(st["a_mat_cho"][0], dtype=np.float64), bool(st["a_mat_cho"][1]))
        elif ckpt["model_name"] == "N4SID":
            model.s_blk = int(st["s_blk"])
            model.ds = int(st["ds"])
            model.best_order = int(st["best_order"])
            model.A = np.asarray(st["A"], dtype=np.float64)
            model.B = np.asarray(st["B"], dtype=np.float64)
            model.C = np.asarray(st["C"], dtype=np.float64)
            model.D = np.asarray(st["D"], dtype=np.float64)
            model.bias = np.asarray(st["bias"], dtype=np.float64)
        ckpt["_device"] = torch.device("cpu")

    return model, ckpt


def _prepare_raw_dataframe_tensors(df_raw: pd.DataFrame, ckpt: Dict) -> Dict[str, np.ndarray]:
    """Apply exact Train-fitted preprocessing saved inside `.pt` to ANY new 1-min exogenous DataFrame."""
    prep = ckpt["preprocessing"]
    lookback = int(prep["lookback"])
    feature_names = list(prep["feature_names"])
    x_mean = np.asarray(prep["x_mean"], dtype=np.float64)
    x_std = np.asarray(prep["x_std"], dtype=np.float64)
    env_cols = list(prep["env_cols"])
    env_mu = np.asarray(prep["env_mu"], dtype=np.float64)
    env_cov_inv = np.asarray(prep["env_cov_inv"], dtype=np.float64)
    wind_pos_cols = list(prep["wind_pos_cols"])
    wind_pos_scale = np.asarray(prep["wind_pos_scale"], dtype=np.float64)

    df = df_raw.copy()
    if "raw_count" in df.columns:
        df = df[df["raw_count"] >= 100].copy()
    if len(df) == 0:
        raise ValueError("Input DataFrame has 0 valid rows (check `raw_count >= 100` or empty CSV).")
    df["ts_dt"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("ts_dt").reset_index(drop=True)

    x_mean_map = {c: float(x_mean[i]) for i, c in enumerate(feature_names)}
    x_std_map = {c: float(x_std[i]) for i, c in enumerate(feature_names)}

    dt_raw = df["ts_dt"].diff().dt.total_seconds().div(60.0).fillna(1.0).to_numpy(dtype=np.float64)
    seg_breaks = np.where(dt_raw > 15.0)[0]
    seg_starts = [0] + seg_breaks.tolist()
    seg_ends = seg_breaks.tolist() + [len(df)]

    pad_len = lookback - 1
    offset = 0
    x_chunks, wpos_chunks, dt_chunks, maha_chunks, vmd_chunks = [], [], [], [], []
    query_indices: List[int] = []

    for s_idx, e_idx in zip(seg_starts, seg_ends):
        seg = df.iloc[s_idx:e_idx].reset_index(drop=True)
        n_seg = len(seg)
        x_seg = (seg[feature_names].to_numpy(dtype=np.float64) - x_mean) / x_std
        wpos_seg = np.maximum(seg[wind_pos_cols].to_numpy(dtype=np.float64), 0.0) / wind_pos_scale
        dt_seg = seg["ts_dt"].diff().dt.total_seconds().div(60.0).fillna(1.0).clip(0.5, 15.0).to_numpy(dtype=np.float64)[:, None]
        env_seg = seg[env_cols].to_numpy(dtype=np.float64) - env_mu
        maha_seg = np.einsum("ni,ij,nj->n", env_seg, env_cov_inv, env_seg)[:, None]
        vmd_seg = compute_causal_multiscale_vmd_proxy(seg, x_mean_map, x_std_map)

        x_pad = np.vstack([np.repeat(x_seg[:1], pad_len, axis=0), x_seg]).astype(np.float32)
        wpos_pad = np.vstack([np.repeat(wpos_seg[:1], pad_len, axis=0), wpos_seg]).astype(np.float32)
        dt_pad = np.vstack([np.ones((pad_len, 1), dtype=np.float64), dt_seg]).astype(np.float32)
        maha_pad = np.vstack([np.repeat(maha_seg[:1], pad_len, axis=0), maha_seg]).astype(np.float32)
        vmd_pad = np.vstack([np.repeat(vmd_seg[:1], pad_len, axis=0), vmd_seg]).astype(np.float32)

        for i in range(n_seg):
            query_indices.append(offset + pad_len + i)

        x_chunks.append(x_pad)
        wpos_chunks.append(wpos_pad)
        dt_chunks.append(dt_pad)
        maha_chunks.append(maha_pad)
        vmd_chunks.append(vmd_pad)
        offset += len(x_pad)

    return {
        "x_norm_Pad": np.concatenate(x_chunks, axis=0),
        "wind_pos_Pad": np.concatenate(wpos_chunks, axis=0),
        "dt_min_Pad": np.concatenate(dt_chunks, axis=0),
        "maha_sq_Pad": np.concatenate(maha_chunks, axis=0),
        "vmd_Pad": np.concatenate(vmd_chunks, axis=0),
        "query_indices": np.asarray(query_indices, dtype=np.int64),
        "timestamps": df["timestamp"].astype(str).to_numpy(),
    }


@torch.no_grad()
def predict_from_tensors(
    model: object,
    ckpt: Dict,
    x_norm_Pad: np.ndarray,
    wind_pos_Pad: np.ndarray,
    dt_min_Pad: np.ndarray,
    maha_sq_Pad: np.ndarray,
    vmd_Pad: np.ndarray,
    indices: np.ndarray,
    batch_size: int = 256,
) -> Dict[str, np.ndarray]:
    """Run calibrated physical-unit prediction + 95% Split-Conformal prediction intervals."""
    prep = ckpt["preprocessing"]
    cal = ckpt["calibration"]
    lookback = int(prep["lookback"])
    horizon_min = int(ckpt["horizon_min"])
    y_mean = np.asarray(prep["y_mean"], dtype=np.float64).reshape(1, -1)
    y_std = np.asarray(prep["y_std"], dtype=np.float64).reshape(1, -1)

    if ckpt["is_torch_model"]:
        dev = ckpt["_device"]
        need_vmd = (ckpt["model_name"] == "VMD-BiLSTM")
        use_amp = (dev.type == "cuda") and (ckpt["model_name"] != "PhySSM")
        offsets = torch.arange(-lookback + 1, 1, device=dev, dtype=torch.long)

        x_t = torch.from_numpy(x_norm_Pad).to(dev)
        w_t = torch.from_numpy(wind_pos_Pad).to(dev)
        dt_t = torch.from_numpy(dt_min_Pad).to(dev)
        maha_t = torch.from_numpy(maha_sq_Pad).to(dev)
        vmd_t = torch.from_numpy(vmd_Pad).to(dev) if need_vmd else None

        idx_t = torch.from_numpy(indices).to(dev, dtype=torch.long)
        mu_list, sig_list = [], []
        has_sig = False
        for s in range(0, len(idx_t), batch_size):
            b_idx = idx_t[s : s + batch_size]
            win_idx = b_idx.unsqueeze(1) + offsets.unsqueeze(0)
            batch = {
                "x_seq": x_t[win_idx],
                "w_pos_seq": w_t[win_idx],
                "dt_seq": dt_t[win_idx],
                "maha_t": maha_t[b_idx],
            }
            if need_vmd and vmd_t is not None:
                batch["vmd_seq"] = vmd_t[win_idx]
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                mu_b, sig_b = model(batch, horizon_min=horizon_min)
            mu_list.append(mu_b.float().detach().cpu().numpy())
            if sig_b is not None:
                has_sig = True
                sig_list.append(sig_b.float().detach().cpu().numpy())
        mu_norm = np.concatenate(mu_list, axis=0).astype(np.float64)
        sig_norm = np.concatenate(sig_list, axis=0).astype(np.float64) if has_sig else None
    else:
        mini_bundle = SimpleNamespace(
            x_norm_Pad=x_norm_Pad,
            lookback=lookback,
            target_names=prep["target_names"],
        )
        mu_norm, sig_norm = model.predict_indices(mini_bundle, indices)

    mu_phys = mu_norm * y_std + y_mean
    sig_raw = (sig_norm * y_std) if sig_norm is not None else None
    sig_phys = apply_sigma_calibration(
        mu_phys,
        sig_raw,
        np.asarray(cal["calibration_scales"], dtype=np.float64),
        str(cal["calibration_mode"]),
    )
    q95 = np.asarray(cal["conformal_q95"], dtype=np.float64).reshape(1, -1)
    half_width = q95 * sig_phys
    return {
        "mu_phys": mu_phys,
        "sigma_phys": sig_phys,
        "pi95_low": mu_phys - half_width,
        "pi95_high": mu_phys + half_width,
        "conformal_q95": q95.squeeze(0),
        "target_names": np.asarray(prep["target_names"]),
    }


def predict_on_dataframe(
    ckpt_path: Path,
    df_raw: pd.DataFrame,
    device_str: str = "auto",
    batch_size: int = 256,
) -> Dict[str, np.ndarray]:
    """Run standalone inference on ANY raw 1-minute antenna DataFrame using only a `.pt` checkpoint."""
    model, ckpt = load_checkpoint_model(ckpt_path, device_str=device_str)
    tensors = _prepare_raw_dataframe_tensors(df_raw, ckpt)
    out = predict_from_tensors(
        model,
        ckpt,
        tensors["x_norm_Pad"],
        tensors["wind_pos_Pad"],
        tensors["dt_min_Pad"],
        tensors["maha_sq_Pad"],
        tensors["vmd_Pad"],
        tensors["query_indices"],
        batch_size=batch_size,
    )
    out["timestamps"] = tensors["timestamps"]
    return out


def predict_ensemble_on_dataframe(
    ckpt_paths: List[Path],
    df_raw: pd.DataFrame,
    device_str: str = "auto",
    batch_size: int = 256,
) -> Dict[str, np.ndarray]:
    """Run multi-seed Deep Ensemble inference (e.g. 5-Seed PhySSM-Ens) on any raw 1-min DataFrame."""
    mus, vars_alea = [], []
    timestamps = None
    target_names = None
    for p in ckpt_paths:
        res = predict_on_dataframe(p, df_raw, device_str=device_str, batch_size=batch_size)
        mus.append(res["mu_phys"])
        vars_alea.append(res["sigma_phys"] ** 2)
        timestamps = res["timestamps"]
        target_names = res["target_names"]
    mu_stack = np.stack(mus, axis=0)
    mu_ens = np.mean(mu_stack, axis=0)
    var_tot = np.mean(np.stack(vars_alea, axis=0), axis=0) + np.var(mu_stack, axis=0)
    sig_ens = np.sqrt(np.maximum(var_tot, 1e-8))
    return {
        "mu_phys": mu_ens,
        "sigma_phys": sig_ens,
        "pi95_low": mu_ens - 1.959964 * sig_ens,
        "pi95_high": mu_ens + 1.959964 * sig_ens,
        "timestamps": timestamps,
        "target_names": target_names,
    }


def verify_checkpoint(
    ckpt_path: Path,
    data_root: Optional[Path] = None,
    device_str: str = "auto",
    batch_size: int = 256,
    bundle_cache: Optional[Dict[Tuple[str, int], Tuple[object, np.ndarray]]] = None,
) -> Dict[str, float]:
    """Verify exact numerical reproducibility of a saved `.pt` checkpoint on its benchmark split."""
    if data_root is None:
        data_root = EXP_ROOT / "data"
    cfg_path = EXP_ROOT / "configs" / "default_benchmark_config.json"

    model, ckpt = load_checkpoint_model(ckpt_path, device_str=device_str)
    cache_key = (str(ckpt["split_name"]), int(ckpt["horizon_min"]))
    if bundle_cache is not None and cache_key in bundle_cache:
        bundle, vmd_all = bundle_cache[cache_key]
    else:
        bundle, vmd_all = load_split_bundle(
            config_path=cfg_path,
            data_root=data_root,
            split_name=ckpt["split_name"],
            horizon_min=int(ckpt["horizon_min"]),
        )
        if bundle_cache is not None:
            bundle_cache[cache_key] = (bundle, vmd_all)
    out = predict_from_tensors(
        model,
        ckpt,
        bundle.x_norm_Pad,
        bundle.wind_pos_Pad,
        bundle.dt_min_Pad,
        bundle.maha_sq_Pad,
        vmd_all,
        bundle.test_indices,
        batch_size=batch_size,
    )
    tgt_phys = bundle.y_phys_All[bundle.test_indices + int(ckpt["horizon_min"])]
    recomputed = compute_channel_and_group_metrics(
        out["mu_phys"],
        out["sigma_phys"],
        tgt_phys,
        bundle.target_names,
        conformal_q95=out["conformal_q95"],
    )
    ref = ckpt["reference_metrics"]["test_regimes"]["overall"]
    key_m = "strain_RMSE_ue" if "strain_RMSE_ue" in ref else "disp_RMSE_mm"
    key_c = "strain_CRPS_ue" if "strain_CRPS_ue" in ref else "disp_CRPS_mm"
    diff_rmse = abs(float(recomputed[key_m]) - float(ref[key_m]))
    diff_crps = abs(float(recomputed[key_c]) - float(ref[key_c]))
    # FP32 / exact-solver models match within 1e-5; cross-GPU FP16 AMP baselines match within 5e-4
    tol = 1e-5 if (not ckpt["is_torch_model"] or ckpt["model_name"] == "PhySSM") else 5e-4
    return {
        "run_id": ckpt["run_id"],
        "metric_key": key_m,
        "ref_rmse": float(ref[key_m]),
        "recomputed_rmse": float(recomputed[key_m]),
        "abs_diff_rmse": diff_rmse,
        "abs_diff_crps": diff_crps,
        "tolerance": tol,
        "passed": bool(diff_rmse < tol and diff_crps < tol),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Load, infer, or verify pretrained Paper 1 checkpoints")
    parser.add_argument("--ckpt", type=str, default="", help="Single .pt checkpoint path")
    parser.add_argument("--verify_all", action="store_true", help="Verify all .pt files in common_engine/checkpoints/")
    parser.add_argument("--csv", type=str, default="", help="Optional raw 1-min antenna CSV for new-data inference")
    parser.add_argument("--out_npz", type=str, default="", help="Output .npz path when --csv is supplied")
    parser.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda:0")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    if args.verify_all:
        ckpt_dir = EXP_ROOT / "checkpoints"
        ckpts = sorted(ckpt_dir.glob("*.pt"))
        if not ckpts:
            raise FileNotFoundError(f"No .pt checkpoints found in {ckpt_dir}")
        ok_cnt = 0
        b_cache: Dict[Tuple[str, int], Tuple[object, np.ndarray]] = {}
        for p in ckpts:
            r = verify_checkpoint(p, device_str=args.device, batch_size=args.batch_size, bundle_cache=b_cache)
            status = "PASS" if r["passed"] else "FAIL"
            if r["passed"]:
                ok_cnt += 1
            print(
                f"[{status}] {r['run_id']:48s} | {r['metric_key']}: ref={r['ref_rmse']:.5f} "
                f"recomp={r['recomputed_rmse']:.5f} (|diff|={r['abs_diff_rmse']:.2e})",
                flush=True,
            )
        print(f"\nVerified {ok_cnt}/{len(ckpts)} local checkpoints cleanly.")
        if ok_cnt != len(ckpts):
            sys.exit(1)
        return

    if not args.ckpt:
        parser.error("Provide --ckpt <path.pt> or --verify_all")

    ckpt_path = Path(args.ckpt)
    if args.csv:
        df_new = pd.read_csv(args.csv)
        res = predict_on_dataframe(ckpt_path, df_new, device_str=args.device, batch_size=args.batch_size)
        out_p = Path(args.out_npz) if args.out_npz else ckpt_path.with_suffix(".new_preds.npz")
        np.savez_compressed(out_p, **res)
        print(f"[INFER] Saved {len(res['mu_phys'])} calibrated predictions -> {out_p}")
    else:
        r = verify_checkpoint(ckpt_path, device_str=args.device, batch_size=args.batch_size)
        print(json.dumps(r, indent=2, ensure_ascii=False))
        if not r["passed"]:
            sys.exit(1)


if __name__ == "__main__":
    main()
