"""[Main] Summarize Main Experiments:
- Table 1: Split M1 8-Model 5-Seed Benchmark (with fair Split-Conformal PICP95/MPIW95)
- Table 2: Split M2 (3 Rolling Folds + +5.84°C Strict OOS) & Split D1 (2025 Dynamic Attitude + <4.38°C Cold OOS)
- Table 3: Multi-Horizon Lookahead Forecasting (H = 0, 10, 30, 60 min)
- Table 4: 5-Seed Pooled Paired Day-Block Bootstrap (B = 2,000) & Daily Wilcoxon Signed-Rank p-values (RMSE & CRPS)
- Table 5: Fair Probabilistic & Uncertainty Quantification Benchmark (Single + SCP vs 5-Seed Deep Ensembles + SCP)
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

EXP_ROOT = Path(__file__).resolve().parents[2]
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from src.metrics_and_calibration import (
    apply_sigma_calibration,
    calibrate_conformal_q95,
    calibrate_gaussian_sigma,
    compute_channel_and_group_metrics,
    day_block_bootstrap_pvalue,
    gaussian_crps_numpy,
)

CACHE_DIR = EXP_ROOT / "outputs" / "raw_cache"
OUT_DIR = EXP_ROOT / "outputs" / "main_results"


def load_raw_records() -> pd.DataFrame:
    rows = []
    for jf in sorted((CACHE_DIR / "raw_runs").glob("*.json")):
        d = json.loads(jf.read_text(encoding="utf-8"))
        if d["ablation"] != "none":
            continue
        expected_r = 2 if d["split_name"] == "D1" else 5
        if d["mode_rank"] != expected_r:
            continue
        for reg_name, m_dict in d["test_regimes"].items():
            r = {
                "split": d["split_name"],
                "horizon": d["horizon_min"],
                "model": d["model_name"],
                "seed": d["seed"],
                "num_params": d["num_params"],
                "seconds": d["train_eval_seconds"],
                "regime": reg_name,
            }
            for k, v in m_dict.items():
                if not k.startswith("ch_"):
                    r[k] = v
            rows.append(r)
    return pd.DataFrame(rows)


def aggregate_seeds(df: pd.DataFrame, models_order: list) -> pd.DataFrame:
    metric_cols = [
        c for c in [
            "strain_RMSE_ue", "strain_MAE_ue", "strain_CRPS_ue", "strain_PICP95", "strain_MPIW95_ue",
            "disp_RMSE_mm", "disp_MAE_mm", "disp_CRPS_mm", "disp_PICP95", "disp_MPIW95_mm",
        ]
        if c in df.columns
    ]
    grp_cols = ["split", "horizon", "model", "regime"]
    agg_rows = []
    for keys, sub in df.groupby(grp_cols, dropna=False):
        entry = dict(zip(grp_cols, keys))
        entry["n_seeds"] = len(sub)
        entry["num_params"] = int(sub["num_params"].iloc[0])
        entry["mean_seconds"] = round(float(sub["seconds"].mean()), 2)
        for mc in metric_cols:
            vals = sub[mc].dropna().to_numpy(dtype=np.float64)
            if len(vals) > 0:
                entry[f"{mc}_mean"] = float(np.mean(vals))
                entry[f"{mc}_std"] = float(np.std(vals, ddof=0))
        agg_rows.append(entry)

    out = pd.DataFrame(agg_rows)
    ord_map = {m: i for i, m in enumerate(models_order)}
    out["m_ord"] = out["model"].map(ord_map)
    return out.sort_values(["split", "horizon", "regime", "m_ord"]).drop(columns=["m_ord"]).reset_index(drop=True)


def load_model_m1_seed_preds(pred_dir: Path, model_name: str, seeds: list) -> list:
    safe_m = model_name.replace(" ", "_").replace("-", "_")
    loaded = []
    for s in seeds:
        p = pred_dir / f"M1_H0_{safe_m}_ab_none_r5_s{s}_preds.npz"
        if p.exists():
            loaded.append((s, np.load(p, allow_pickle=True)))
    if not loaded:
        p42 = pred_dir / f"M1_H0_{safe_m}_ab_none_r5_s42_preds.npz"
        if p42.exists():
            loaded.append((42, np.load(p42, allow_pickle=True)))
    return loaded


def compute_pooled_significance_and_ensembles(pred_dir: Path, models_order: list, seeds_5: list) -> None:
    phy_list = load_model_m1_seed_preds(pred_dir, "PhySSM", seeds_5)
    if not phy_list:
        return

    ref0 = phy_list[0][1]
    dates = ref0["dates"]
    tgt_phys = ref0["target_phys"].astype(np.float64)
    tgt_names = [str(x) for x in ref0["target_names"]]
    s_idx = [i for i, n in enumerate(tgt_names) if n.startswith("strain_")]
    d_idx = [i for i, n in enumerate(tgt_names) if n.startswith("disp_")]
    oos_m = ref0["is_oos"].astype(bool)
    mot_m = ref0["is_motor"].astype(bool) if "is_motor" in ref0 else np.zeros(len(dates), dtype=bool)
    cln_m = ref0["is_clean"].astype(bool)

    regimes_map = {
        "overall": np.ones(len(dates), dtype=bool),
        "in_support": ~oos_m,
        "oos_tail": oos_m,
        "clean_standby": cln_m,
        "motor_active": mot_m,
    }

    # Exact (N, K_seeds, C_ch) per-minute squared error & CRPS matching Table 1 channel-averaged definition
    phy_sq_s = np.stack([((z["mu_phys"].astype(np.float64) - tgt_phys)[:, s_idx] ** 2) for _, z in phy_list], axis=1)
    phy_sq_d = np.stack([((z["mu_phys"].astype(np.float64) - tgt_phys)[:, d_idx] ** 2) for _, z in phy_list], axis=1)
    phy_crps_s = np.stack(
        [gaussian_crps_numpy(z["mu_phys"].astype(np.float64)[:, s_idx], z["sigma_phys"].astype(np.float64)[:, s_idx], tgt_phys[:, s_idx]) for _, z in phy_list],
        axis=1,
    )

    unique_days = np.unique(dates)
    boot_rows = []
    uq_rows = []

    for m in models_order:
        m_list = load_model_m1_seed_preds(pred_dir, m, seeds_5)
        if not m_list:
            continue

        # 1. Multi-seed pooled Day-Block Bootstrap & Daily Wilcoxon Signed-Rank Test vs PhySSM
        if m != "PhySSM":
            c_sq_s = np.stack([((z["mu_phys"].astype(np.float64) - tgt_phys)[:, s_idx] ** 2) for _, z in m_list], axis=1)
            c_sq_d = np.stack([((z["mu_phys"].astype(np.float64) - tgt_phys)[:, d_idx] ** 2) for _, z in m_list], axis=1)
            c_crps_s = np.stack(
                [gaussian_crps_numpy(z["mu_phys"].astype(np.float64)[:, s_idx], z["sigma_phys"].astype(np.float64)[:, s_idx], tgt_phys[:, s_idx]) for _, z in m_list],
                axis=1,
            )

            bs_all = day_block_bootstrap_pvalue(dates, phy_sq_s, c_sq_s, n_boot=2000, seed=42, is_rmse=True)
            bs_oos = day_block_bootstrap_pvalue(dates[oos_m], phy_sq_s[oos_m], c_sq_s[oos_m], n_boot=2000, seed=42, is_rmse=True)
            bc_all = day_block_bootstrap_pvalue(dates, phy_crps_s, c_crps_s, n_boot=2000, seed=42, is_rmse=False)
            bc_oos = day_block_bootstrap_pvalue(dates[oos_m], phy_crps_s[oos_m], c_crps_s[oos_m], n_boot=2000, seed=42, is_rmse=False)
            bd_all = day_block_bootstrap_pvalue(dates, phy_sq_d, c_sq_d, n_boot=2000, seed=42, is_rmse=True)

            # Paired daily channel-averaged RMSE Wilcoxon across 14 held-out test days
            day_phy = np.array([np.mean(np.sqrt(np.mean(phy_sq_s[dates == d], axis=0))) for d in unique_days])
            day_cand = np.array([np.mean(np.sqrt(np.mean(c_sq_s[dates == d], axis=0))) for d in unique_days])
            try:
                _, wil_p = wilcoxon(day_cand - day_phy, alternative="greater")
            except Exception:
                wil_p = 1.0

            boot_rows.append({
                "baseline": m,
                "n_pooled_seeds": len(m_list),
                "strain_overall_rmse_gain_ue": bs_all["mean_gain"],
                "strain_overall_ci95_low": bs_all["ci95_low"],
                "strain_overall_ci95_high": bs_all["ci95_high"],
                "strain_overall_p_value": bs_all["p_value_one_sided"],
                "daily_wilcoxon_p_value": float(wil_p),
                "strain_oos_rmse_gain_ue": bs_oos["mean_gain"],
                "strain_oos_p_value": bs_oos["p_value_one_sided"],
                "strain_overall_crps_gain_ue": bc_all["mean_gain"],
                "strain_overall_crps_p_value": bc_all["p_value_one_sided"],
                "strain_oos_crps_gain_ue": bc_oos["mean_gain"],
                "strain_oos_crps_p_value": bc_oos["p_value_one_sided"],
                "disp_overall_rmse_gain_mm": bd_all["mean_gain"],
                "disp_overall_p_value": bd_all["p_value_one_sided"],
            })

        # 2. Single-Model (1x) + Normalized Split-Conformal Calibration (averaged across seeds)
        for reg_name, r_mask in regimes_map.items():
            if r_mask.sum() == 0:
                continue
            per_seed_mets = []
            for _, z in m_list:
                q_s = z["conformal_q95"].astype(np.float64) if "conformal_q95" in z else None
                per_seed_mets.append(
                    compute_channel_and_group_metrics(
                        z["mu_phys"].astype(np.float64),
                        z["sigma_phys"].astype(np.float64),
                        tgt_phys,
                        tgt_names,
                        mask=r_mask,
                        conformal_q95=q_s,
                    )
                )
            uq_rows.append({
                "method": f"{m} (1x+SCP)",
                "base_model": m,
                "paradigm": "Single Model + Split Conformal",
                "inference_cost_mult": 1,
                "regime": reg_name,
                "strain_RMSE_ue": float(np.mean([x["strain_RMSE_ue"] for x in per_seed_mets])),
                "strain_CRPS_ue": float(np.mean([x["strain_CRPS_ue"] for x in per_seed_mets])),
                "strain_PICP95": float(np.mean([x["strain_PICP95"] for x in per_seed_mets])),
                "strain_MPIW95_ue": float(np.mean([x["strain_MPIW95_ue"] for x in per_seed_mets])),
                "disp_RMSE_mm": float(np.mean([x["disp_RMSE_mm"] for x in per_seed_mets])),
                "disp_CRPS_mm": float(np.mean([x["disp_CRPS_mm"] for x in per_seed_mets])),
                "disp_PICP95": float(np.mean([x["disp_PICP95"] for x in per_seed_mets])),
                "disp_MPIW95_mm": float(np.mean([x["disp_MPIW95_mm"] for x in per_seed_mets])),
            })

        # 3. Build 5-Seed Deep Ensemble + Normalized Split-Conformal Calibration (if >= 3 seeds with val_mu_phys exist)
        valid_ens = [z for _, z in m_list if "val_mu_phys" in z]
        if len(valid_ens) >= 3:
            val_tgt = valid_ens[0]["val_target_phys"].astype(np.float64)
            val_mus = np.stack([z["val_mu_phys"].astype(np.float64) for z in valid_ens], axis=0)
            val_sigs = np.stack([z["val_sigma_phys"].astype(np.float64) for z in valid_ens], axis=0)
            test_mus = np.stack([z["mu_phys"].astype(np.float64) for z in valid_ens], axis=0)
            test_sigs = np.stack([z["sigma_phys"].astype(np.float64) for z in valid_ens], axis=0)

            ens_val_mu = np.mean(val_mus, axis=0)
            ens_val_var = np.mean(val_sigs ** 2, axis=0) + np.var(val_mus, axis=0, ddof=0)
            ens_val_sig_raw = np.sqrt(np.maximum(ens_val_var, 1e-8))

            ens_test_mu = np.mean(test_mus, axis=0)
            ens_test_var = np.mean(test_sigs ** 2, axis=0) + np.var(test_mus, axis=0, ddof=0)
            ens_test_sig_raw = np.sqrt(np.maximum(ens_test_var, 1e-8))

            scales, cal_mode = calibrate_gaussian_sigma(ens_val_mu, ens_val_sig_raw, val_tgt)
            ens_val_sig = apply_sigma_calibration(ens_val_mu, ens_val_sig_raw, scales, cal_mode)
            ens_test_sig = apply_sigma_calibration(ens_test_mu, ens_test_sig_raw, scales, cal_mode)
            conf_q95 = calibrate_conformal_q95(ens_val_mu, ens_val_sig, val_tgt, alpha=0.05)

            for reg_name, r_mask in regimes_map.items():
                if r_mask.sum() == 0:
                    continue
                met = compute_channel_and_group_metrics(
                    ens_test_mu, ens_test_sig, tgt_phys, tgt_names, mask=r_mask, conformal_q95=conf_q95
                )
                uq_rows.append({
                    "method": f"{m}-DeepEns (5x)",
                    "base_model": m,
                    "paradigm": "5-Seed Deep Ensemble + Split Conformal",
                    "inference_cost_mult": len(valid_ens),
                    "regime": reg_name,
                    "strain_RMSE_ue": met["strain_RMSE_ue"],
                    "strain_CRPS_ue": met["strain_CRPS_ue"],
                    "strain_PICP95": met["strain_PICP95"],
                    "strain_MPIW95_ue": met["strain_MPIW95_ue"],
                    "disp_RMSE_mm": met["disp_RMSE_mm"],
                    "disp_CRPS_mm": met["disp_CRPS_mm"],
                    "disp_PICP95": met["disp_PICP95"],
                    "disp_MPIW95_mm": met["disp_MPIW95_mm"],
                })

    if boot_rows:
        pd.DataFrame(boot_rows).to_csv(OUT_DIR / "table4_day_block_bootstrap_pvalues.csv", index=False)

    if uq_rows:
        pd.DataFrame(uq_rows).to_csv(OUT_DIR / "table5_probabilistic_uq_fairness_ensembles.csv", index=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((EXP_ROOT / "configs" / "default_benchmark_config.json").read_text(encoding="utf-8"))
    models_order = cfg["models_order"]
    seeds_5 = cfg["primary_seeds"]

    df_seed = load_raw_records()
    if len(df_seed) == 0:
        print("No raw run JSONs found in common_engine/raw_cache/raw_runs.")
        return

    df_seed.to_csv(OUT_DIR / "main_all_seed_level_runs.csv", index=False)
    df_agg = aggregate_seeds(df_seed, models_order)

    # Table 1: Split M1 Main Benchmark (H = 0)
    t1 = df_agg[(df_agg["split"] == "M1") & (df_agg["horizon"] == 0)].copy()
    t1.to_csv(OUT_DIR / "table1_split_M1_8models_5seeds.csv", index=False)

    # Table 2: Rolling-Origin M2 (Folds 1..3 + 3-Fold Mean) & Dynamic D1
    t2 = df_agg[(df_agg["split"].isin(["M2_fold1", "M2_fold2", "M2_fold3", "D1"])) & (df_agg["horizon"] == 0)].copy()
    m2_sub = t2[t2["split"].str.startswith("M2_")].copy()
    if len(m2_sub) > 0:
        num_cols = [c for c in m2_sub.columns if c.endswith("_mean") or c.endswith("_std") or c in ("num_params", "mean_seconds")]
        m2_avg = m2_sub.groupby(["horizon", "model", "regime"], as_index=False)[num_cols].mean()
        m2_avg["split"] = "M2_3fold_avg"
        m2_avg["n_seeds"] = 3
        ord_map = {m: i for i, m in enumerate(models_order)}
        t2 = pd.concat([t2, m2_avg], ignore_index=True)
        t2["m_ord"] = t2["model"].map(ord_map)
        t2 = t2.sort_values(["split", "regime", "m_ord"]).drop(columns=["m_ord"])
    t2.to_csv(OUT_DIR / "table2_rolling_M2_and_dynamic_D1.csv", index=False)

    # Table 3: Multi-Horizon Lookahead (H = 0, 10, 30, 60 min on Split M1)
    rep_models = ["Composite-GPR", "N4SID", "Causal-Dilated TCN", "TimeXer", "TimePro", "PhySSM"]
    t3 = df_agg[
        (df_agg["split"] == "M1")
        & (df_agg["model"].isin(rep_models))
        & (df_agg["regime"].isin(["overall", "oos_tail"]))
    ].copy()
    t3.to_csv(OUT_DIR / "table3_multi_horizon_forecasting.csv", index=False)

    # Tables 4 & 5: Multi-Seed Pooled Day-Block Bootstrap & 5-Seed Deep Ensemble UQ Fairness
    compute_pooled_significance_and_ensembles(CACHE_DIR / "predictions", models_order, seeds_5)
    print(f"[Main] Summary tables (table1..table5) written to {OUT_DIR}")


if __name__ == "__main__":
    main()
