"""Unified single-run training, validation conformal calibration, and multi-regime evaluation engine for Paper 1."""

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F

EXP_ROOT = Path(__file__).resolve().parents[1]
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from src.data_pipeline import SplitBundle, load_split_bundle
from src.metrics_and_calibration import (
    apply_sigma_calibration,
    calibrate_conformal_q95,
    calibrate_gaussian_sigma,
    compute_channel_and_group_metrics,
    gaussian_crps_numpy,
    gaussian_crps_torch,
)
from src.models import NON_TORCH_MODELS, build_model


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GPUBatchSlicer:
    """Resident GPU contiguous timeline slicer for ultra-fast causal window batching."""

    def __init__(self, bundle: SplitBundle, vmd_all: np.ndarray, device: torch.device):
        self.device = device
        self.lw = bundle.lookback
        self.h = bundle.horizon_min
        self.x = torch.from_numpy(bundle.x_norm_Pad).to(device)
        self.w_pos = torch.from_numpy(bundle.wind_pos_Pad).to(device)
        self.dt = torch.from_numpy(bundle.dt_min_Pad).to(device)
        self.maha = torch.from_numpy(bundle.maha_sq_Pad).to(device)
        self.y_norm = torch.from_numpy(bundle.y_norm_All).to(device)
        self.vmd = torch.from_numpy(vmd_all).to(device)
        self.offsets = torch.arange(-self.lw + 1, 1, device=device, dtype=torch.long)

    def get_batch(self, idx_t: torch.Tensor, need_vmd: bool = False) -> Dict[str, torch.Tensor]:
        win_idx = idx_t.unsqueeze(1) + self.offsets.unsqueeze(0)
        out = {
            "x_seq": self.x[win_idx],
            "w_pos_seq": self.w_pos[win_idx],
            "dt_seq": self.dt[win_idx],
            "maha_t": self.maha[idx_t],
            "y_norm": self.y_norm[idx_t + self.h],
        }
        if need_vmd:
            out["vmd_seq"] = self.vmd[win_idx]
        return out


@torch.no_grad()
def predict_torch_model(
    model: torch.nn.Module,
    slicer: GPUBatchSlicer,
    indices_np: np.ndarray,
    batch_size: int,
    horizon_min: int,
    need_vmd: bool,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    model.eval()
    idx_all = torch.from_numpy(indices_np).to(slicer.device, dtype=torch.long)
    mu_list, sig_list = [], []
    has_sigma = False
    use_amp = (slicer.device.type == "cuda") and (model.__class__.__name__ != "PhySSM")

    for s in range(0, len(idx_all), batch_size):
        b_idx = idx_all[s : s + batch_size]
        batch = slicer.get_batch(b_idx, need_vmd=need_vmd)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            mu_b, sig_b = model(batch, horizon_min=horizon_min)
        mu_list.append(mu_b.float().detach().cpu().numpy())
        if sig_b is not None:
            has_sigma = True
            sig_list.append(sig_b.float().detach().cpu().numpy())

    mu_arr = np.concatenate(mu_list, axis=0).astype(np.float64)
    sig_arr = np.concatenate(sig_list, axis=0).astype(np.float64) if has_sigma else None
    return mu_arr, sig_arr


@torch.no_grad()
def update_ema_weights(ema_model: torch.nn.Module, live_model: torch.nn.Module, decay: float = 0.992) -> None:
    for p_ema, p_live in zip(ema_model.parameters(), live_model.parameters()):
        p_ema.data.mul_(decay).add_(p_live.data, alpha=1.0 - decay)
    for b_ema, b_live in zip(ema_model.buffers(), live_model.buffers()):
        b_ema.data.copy_(b_live.data)


def train_and_evaluate(
    split_name: str,
    model_name: str,
    seed: int,
    horizon_min: int = 0,
    ablation: str = "none",
    mode_rank: int = 5,
    device_str: str = "cuda:0",
    save_predictions: bool = False,
    save_checkpoint: bool = False,
    data_root: Optional[Path] = None,
    out_root: Optional[Path] = None,
) -> Dict:
    cfg_path = EXP_ROOT / "configs" / "default_benchmark_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    if data_root is None:
        data_root = EXP_ROOT / "data"
    if out_root is None:
        out_root = EXP_ROOT / "outputs" / "raw_cache"

    (out_root / "raw_runs").mkdir(parents=True, exist_ok=True)
    (out_root / "predictions").mkdir(parents=True, exist_ok=True)
    (out_root / "physics").mkdir(parents=True, exist_ok=True)
    if save_checkpoint:
        (out_root / "checkpoints").mkdir(parents=True, exist_ok=True)

    safe_model = model_name.replace(" ", "_").replace("-", "_")
    run_id = f"{split_name}_H{horizon_min}_{safe_model}_ab_{ablation}_r{mode_rank}_s{seed}"
    json_out_path = out_root / "raw_runs" / f"{run_id}.json"

    set_all_seeds(seed)
    t_start = time.time()

    bundle, vmd_all = load_split_bundle(
        config_path=cfg_path,
        data_root=data_root,
        split_name=split_name,
        horizon_min=horizon_min,
    )
    model = build_model(model_name, bundle, seed=seed, ablation=ablation, mode_rank=mode_rank)

    if model_name in NON_TORCH_MODELS:
        num_params = 0
        best_epoch = 1
        model.fit(bundle)
        val_mu_norm, val_sig_norm = model.predict_indices(bundle, bundle.val_indices)
        test_mu_norm, test_sig_norm = model.predict_indices(bundle, bundle.test_indices)
    else:
        device = torch.device(device_str if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        num_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))

        slicer = GPUBatchSlicer(bundle, vmd_all, device=device)
        need_vmd = (model_name == "VMD-BiLSTM")
        is_physsm = (model_name == "PhySSM")

        batch_size = int(cfg["batch_size"])
        eval_bs = int(cfg["eval_batch_size"])
        max_epochs = int(cfg["max_epochs"])
        patience = int(cfg["patience"])
        lr = float(cfg["learning_rate"])
        wd = float(cfg["weight_decay"])
        grad_clip = float(cfg["grad_clip_norm"])

        train_idx_gpu = torch.from_numpy(bundle.train_indices).to(device, dtype=torch.long)
        val_idx_gpu = torch.from_numpy(bundle.val_indices).to(device, dtype=torch.long)
        y_val_norm = bundle.y_norm_All[bundle.val_indices + horizon_min]
        n_tgt = len(bundle.target_names)

        if is_physsm and n_tgt == 2:
            model.anchor_homogeneous_trunk_d1(slicer, train_idx_gpu, val_idx_gpu)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr * 0.45, weight_decay=3e-4)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        ema_model = None

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=3e-5)
        use_amp = (device.type == "cuda") and (not is_physsm)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        best_val_score = float("inf")
        best_state = None
        best_epoch = 0
        no_improve = 0

        if is_physsm and n_tgt == 2:
            val_mu_0, val_sig_0 = predict_torch_model(model, slicer, bundle.val_indices, eval_bs, horizon_min, need_vmd)
            val_rmse_0 = float(np.sqrt(np.mean((val_mu_0 - y_val_norm) ** 2)))
            val_crps_0 = float(np.mean(gaussian_crps_numpy(val_mu_0, val_sig_0, y_val_norm)))
            best_val_score = 0.92 * val_rmse_0 + 0.08 * val_crps_0
            best_state = copy.deepcopy(model.state_dict())

        for epoch in range(1, max_epochs + 1):
            model.train()
            sub_train_idx = train_idx_gpu if is_physsm else train_idx_gpu[(epoch % 2)::2]
            perm = torch.randperm(len(sub_train_idx), device=device)
            shuf_idx = sub_train_idx[perm]

            for s in range(0, len(shuf_idx), batch_size):
                b_idx = shuf_idx[s : s + batch_size]
                batch = slicer.get_batch(b_idx, need_vmd=need_vmd)
                y_tgt = batch["y_norm"]

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    mu_b, sig_b = model(batch, horizon_min=horizon_min)

                mu_f = mu_b.float()
                y_f = y_tgt.float()
                if is_physsm and sig_b is not None:
                    sig_f = sig_b.float()
                    point_l = 0.65 * F.mse_loss(mu_f, y_f) + 0.35 * F.smooth_l1_loss(mu_f, y_f, beta=0.25)
                    crps_l = torch.mean(gaussian_crps_torch(mu_f.detach(), sig_f, y_f))
                    orth_l = model.orthogonality_loss().float()
                    loss = point_l + 0.25 * crps_l + 0.03 * orth_l
                else:
                    loss = 0.65 * F.mse_loss(mu_f, y_f) + 0.35 * F.smooth_l1_loss(mu_f, y_f, beta=0.25)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()

                if ema_model is not None:
                    update_ema_weights(ema_model, model, decay=0.992)

            scheduler.step()

            eval_net = model
            val_mu_n, val_sig_n = predict_torch_model(model, slicer, bundle.val_indices, eval_bs, horizon_min, need_vmd)
            val_rmse_n = float(np.sqrt(np.mean((val_mu_n - y_val_norm) ** 2)))
            if is_physsm and val_sig_n is not None:
                val_crps_n = float(np.mean(gaussian_crps_numpy(val_mu_n, val_sig_n, y_val_norm)))
                val_score = 0.90 * val_rmse_n + 0.10 * val_crps_n
                if ema_model is not None and epoch >= 4:
                    e_mu, e_sig = predict_torch_model(ema_model, slicer, bundle.val_indices, eval_bs, horizon_min, need_vmd)
                    e_score = 0.90 * float(np.sqrt(np.mean((e_mu - y_val_norm) ** 2))) + 0.10 * float(
                        np.mean(gaussian_crps_numpy(e_mu, e_sig, y_val_norm))
                    )
                    if e_score < val_score:
                        val_score = e_score
                        eval_net = ema_model
            else:
                val_score = val_rmse_n

            if val_score < best_val_score - 1e-5:
                best_val_score = val_score
                best_state = copy.deepcopy(eval_net.state_dict())
                best_epoch = epoch
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        val_mu_norm, val_sig_norm = predict_torch_model(model, slicer, bundle.val_indices, eval_bs, horizon_min, need_vmd)
        test_mu_norm, test_sig_norm = predict_torch_model(model, slicer, bundle.test_indices, eval_bs, horizon_min, need_vmd)

        if is_physsm and ablation == "none" and horizon_min == 0:
            phys_dict = model.export_learned_physics()
            np.savez_compressed(
                out_root / "physics" / f"physsm_{split_name}_r{mode_rank}_s{seed}_physics.npz",
                target_names=np.array(bundle.target_names),
                **phys_dict,
            )

    y_mean = bundle.y_mean.reshape(1, -1)
    y_std = bundle.y_std.reshape(1, -1)

    val_mu_phys = val_mu_norm * y_std + y_mean
    val_sig_phys_raw = (val_sig_norm * y_std) if val_sig_norm is not None else None
    val_tgt_phys = bundle.y_phys_All[bundle.val_indices + horizon_min]

    test_mu_phys = test_mu_norm * y_std + y_mean
    test_sig_phys_raw = (test_sig_norm * y_std) if test_sig_norm is not None else None
    test_tgt_phys = bundle.y_phys_All[bundle.test_indices + horizon_min]

    cal_scales, cal_mode = calibrate_gaussian_sigma(val_mu_phys, val_sig_phys_raw, val_tgt_phys)
    val_sig_phys = apply_sigma_calibration(val_mu_phys, val_sig_phys_raw, cal_scales, cal_mode)
    test_sig_phys = apply_sigma_calibration(test_mu_phys, test_sig_phys_raw, cal_scales, cal_mode)

    # Fair Normalized Split-Conformal Quantile Calibration strictly on Validation set (applied identically to ALL models)
    conf_q95 = calibrate_conformal_q95(val_mu_phys, val_sig_phys, val_tgt_phys, alpha=0.05)

    val_metrics = compute_channel_and_group_metrics(
        val_mu_phys, val_sig_phys, val_tgt_phys, bundle.target_names, conformal_q95=conf_q95
    )
    regime_metrics = {
        reg_name: compute_channel_and_group_metrics(
            test_mu_phys, test_sig_phys, test_tgt_phys, bundle.target_names, mask=reg_mask, conformal_q95=conf_q95
        )
        for reg_name, reg_mask in bundle.test_regimes.items()
    }

    elapsed = round(time.time() - t_start, 2)
    summary = {
        "run_id": run_id,
        "split_name": split_name,
        "horizon_min": horizon_min,
        "model_name": model_name,
        "ablation": ablation,
        "mode_rank": mode_rank,
        "seed": seed,
        "num_params": num_params,
        "best_epoch": best_epoch,
        "train_eval_seconds": elapsed,
        "calibration_mode": cal_mode,
        "calibration_scales": [round(float(x), 5) for x in cal_scales],
        "conformal_q95": [round(float(x), 5) for x in conf_q95],
        "val_overall": val_metrics,
        "test_regimes": regime_metrics,
    }
    json_out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if save_predictions:
        np.savez_compressed(
            out_root / "predictions" / f"{run_id}_preds.npz",
            mu_phys=test_mu_phys.astype(np.float32),
            sigma_phys=test_sig_phys.astype(np.float32),
            target_phys=test_tgt_phys.astype(np.float32),
            val_mu_phys=val_mu_phys.astype(np.float32),
            val_sigma_phys=val_sig_phys.astype(np.float32),
            val_target_phys=val_tgt_phys.astype(np.float32),
            conformal_q95=conf_q95.astype(np.float32),
            dates=bundle.test_dates.astype(str),
            timestamps=bundle.test_timestamps.astype(str),
            temp_mean=bundle.test_temp_mean.astype(np.float32),
            wind_speed=bundle.test_wind_speed.astype(np.float32),
            temp_range=bundle.test_temp_range.astype(np.float32),
            is_oos=bundle.test_regimes["oos_tail"],
            is_clean=bundle.test_regimes["clean_standby"],
            is_motor=bundle.test_regimes["motor_active"],
            target_names=np.array(bundle.target_names),
        )

    if save_checkpoint:
        ckpt_path = out_root / "checkpoints" / f"{run_id}.pt"
        is_torch_m = model_name not in NON_TORCH_MODELS
        if is_torch_m:
            model_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            non_torch_state = None
        elif model_name == "Composite-GPR":
            model_state = None
            non_torch_state = {
                "num_inducing": int(model.num_inducing),
                "seed": int(model.seed),
                "z_ind": model.z_ind.astype(np.float64),
                "best_params": dict(model.best_params),
                "sigma_n2": float(model.sigma_n2),
                "alpha_weights": model.alpha_weights.astype(np.float64),
                "k_mm_cho": (model.k_mm_cho[0].astype(np.float64), bool(model.k_mm_cho[1])),
                "a_mat_cho": (model.a_mat_cho[0].astype(np.float64), bool(model.a_mat_cho[1])),
            }
        elif model_name == "N4SID":
            model_state = None
            non_torch_state = {
                "s_blk": int(model.s_blk),
                "ds": int(model.ds),
                "best_order": int(model.best_order),
                "A": model.A.astype(np.float64),
                "B": model.B.astype(np.float64),
                "C": model.C.astype(np.float64),
                "D": model.D.astype(np.float64),
                "bias": model.bias.astype(np.float64),
            }
        else:
            model_state = None
            non_torch_state = None

        ckpt_payload = {
            "format_version": "physsm_ckpt_v1",
            "run_id": run_id,
            "split_name": split_name,
            "model_name": model_name,
            "seed": int(seed),
            "horizon_min": int(horizon_min),
            "ablation": ablation,
            "mode_rank": int(mode_rank),
            "num_params": int(num_params),
            "best_epoch": int(best_epoch),
            "train_eval_seconds": float(elapsed),
            "is_torch_model": bool(is_torch_m),
            "state_dict": model_state,
            "non_torch_state": non_torch_state,
            "preprocessing": {
                "lookback": int(bundle.lookback),
                "feature_names": list(bundle.feature_names),
                "group_slices": {k: list(v) for k, v in bundle.group_slices.items()},
                "target_names": list(bundle.target_names),
                "x_mean": bundle.x_mean.astype(np.float64),
                "x_std": bundle.x_std.astype(np.float64),
                "y_mean": bundle.y_mean.astype(np.float64),
                "y_std": bundle.y_std.astype(np.float64),
                "u_svd_init": bundle.u_svd_init.astype(np.float32),
                "env_cols": list(bundle.env_cols) if bundle.env_cols is not None else [],
                "env_mu": bundle.env_mu.astype(np.float64) if bundle.env_mu is not None else None,
                "env_cov_inv": bundle.env_cov_inv.astype(np.float64) if bundle.env_cov_inv is not None else None,
                "maha_p99_train": float(bundle.maha_p99_train),
                "wind_pos_cols": list(bundle.wind_pos_cols) if bundle.wind_pos_cols is not None else [],
                "wind_pos_scale": bundle.wind_pos_scale.astype(np.float64) if bundle.wind_pos_scale is not None else None,
            },
            "calibration": {
                "calibration_mode": cal_mode,
                "calibration_scales": np.asarray(cal_scales, dtype=np.float64),
                "conformal_q95": np.asarray(conf_q95, dtype=np.float64),
            },
            "reference_metrics": {
                "val_overall": val_metrics,
                "test_regimes": regime_metrics,
            },
        }
        torch.save(ckpt_payload, ckpt_path)

    s_rmse = regime_metrics["overall"].get("strain_RMSE_ue", -1.0)
    d_rmse = regime_metrics["overall"].get("disp_RMSE_mm", -1.0)
    oos_s = regime_metrics["oos_tail"].get("strain_RMSE_ue", -1.0)
    ov_cov = regime_metrics["overall"].get("strain_PICP95", regime_metrics["overall"].get("disp_PICP95", -1.0))
    print(
        f"[DONE] {run_id} | {elapsed:.1f}s | ep={best_epoch} | "
        f"Overall Strain={s_rmse:.3f} (OOS={oos_s:.3f}, PICP95={ov_cov:.3f}) | Disp={d_rmse:.4f} mm",
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run single Paper 1 benchmark experiment")
    parser.add_argument("--split", type=str, default="M1")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon", type=int, default=0)
    parser.add_argument("--ablation", type=str, default="none")
    parser.add_argument("--mode_rank", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--save_checkpoint", action="store_true")
    parser.add_argument("--data_root", type=str, default="")
    parser.add_argument("--out_root", type=str, default="")
    args = parser.parse_args()

    train_and_evaluate(
        split_name=args.split,
        model_name=args.model,
        seed=args.seed,
        horizon_min=args.horizon,
        ablation=args.ablation,
        mode_rank=args.mode_rank,
        device_str=args.device,
        save_predictions=args.save_predictions,
        save_checkpoint=args.save_checkpoint,
        data_root=Path(args.data_root) if args.data_root else None,
        out_root=Path(args.out_root) if args.out_root else None,
    )


if __name__ == "__main__":
    main()
