"""[Ablation] Summarize Cross-Protocol Orthogonal Mechanistic Ablations (M1 5-Seed + M2 3-Fold + D1)
& Latent Mode Rank Sensitivity (r = 1, 3, 5, 8, 15).
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd

EXP_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = EXP_ROOT / "outputs" / "raw_cache"
OUT_DIR = EXP_ROOT / "outputs" / "ablation_results"

ABLATION_ORDER = [
    ("none", "Full PhySSM (Ours)"),
    ("wo_dissipative_ssm", "w/o Dissipative SSM"),
    ("wo_homogeneous_dual_stream", "w/o Homog. Dual-Stream"),
    ("wo_actuation_quality_gate", "w/o Actuation & Qual. Gate"),
    ("wo_spatial_grad", "w/o Subsystem Routing"),
    ("wo_low_rank_modes", "w/o Low-Rank Modes"),
    ("wo_wind_modulation", "w/o Wind-Convective Mod."),
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ab_keys = [k for k, _ in ABLATION_ORDER]
    ab_lbl_map = dict(ABLATION_ORDER)
    ab_ord_map = {k: i for i, (k, _) in enumerate(ABLATION_ORDER)}

    rows_m1 = []
    rows_all = []
    for jf in sorted((CACHE_DIR / "raw_runs").glob("*.json")):
        d = json.loads(jf.read_text(encoding="utf-8"))
        if d["horizon_min"] != 0 or d["model_name"] != "PhySSM":
            continue
        sp = d["split_name"]
        ab = d["ablation"]
        rk = int(d["mode_rank"])
        for reg_name, m_dict in d["test_regimes"].items():
            r = {
                "split": sp,
                "ablation": ab,
                "mode_rank": rk,
                "seed": int(d["seed"]),
                "num_params": int(d["num_params"]),
                "seconds": float(d["train_eval_seconds"]),
                "regime": reg_name,
            }
            for k, v in m_dict.items():
                if not k.startswith("ch_"):
                    r[k] = v
            if sp == "M1":
                rows_m1.append(r)
            if ab in ab_keys and ((sp == "M1" and rk == 5) or (sp.startswith("M2_") and rk == 5) or (sp == "D1" and rk == 2)):
                rows_all.append(r)

    if not rows_m1:
        print("No PhySSM ablation JSONs found.")
        return

    df_m1 = pd.DataFrame(rows_m1)
    df_m1.to_csv(OUT_DIR / "ablation_all_seed_runs.csv", index=False)

    metric_cols = [
        c for c in [
            "strain_RMSE_ue", "strain_MAE_ue", "strain_CRPS_ue", "strain_PICP95", "strain_MPIW95_ue",
            "disp_RMSE_mm", "disp_MAE_mm", "disp_CRPS_mm", "disp_PICP95", "disp_MPIW95_mm",
        ]
        if c in df_m1.columns
    ]

    agg_rows = []
    for (ab, rk, reg), sub in df_m1.groupby(["ablation", "mode_rank", "regime"]):
        entry = {
            "ablation": ab,
            "mode_rank": int(rk),
            "regime": reg,
            "n_seeds": len(sub),
            "num_params": int(sub["num_params"].iloc[0]),
            "mean_seconds": round(float(sub["seconds"].mean()), 2),
        }
        for mc in metric_cols:
            vals = sub[mc].dropna().to_numpy(dtype=np.float64)
            if len(vals) > 0:
                entry[f"{mc}_mean"] = float(np.mean(vals))
                entry[f"{mc}_std"] = float(np.std(vals, ddof=0))
        agg_rows.append(entry)

    df_agg = pd.DataFrame(agg_rows)

    # Table 1: M1 6 Orthogonal Ablations (kept for full ledger backward compatibility)
    t1 = df_agg[(df_agg["mode_rank"] == 5) & (df_agg["ablation"].isin(ab_keys))].copy()
    t1["variant_label"] = t1["ablation"].map(ab_lbl_map)
    t1["ab_ord"] = t1["ablation"].map(ab_ord_map)
    t1 = t1.sort_values(["regime", "ab_ord"]).drop(columns=["ab_ord"]).reset_index(drop=True)

    delta_s_rmse, pct_s_rmse, pct_s_crps, pct_d_rmse = [], [], [], []
    for _, row in t1.iterrows():
        ref = t1[(t1["regime"] == row["regime"]) & (t1["ablation"] == "none")]
        r_s = float(ref["strain_RMSE_ue_mean"].iloc[0])
        r_c = float(ref["strain_CRPS_ue_mean"].iloc[0])
        r_d = float(ref["disp_RMSE_mm_mean"].iloc[0])
        cur_s = float(row["strain_RMSE_ue_mean"])
        cur_c = float(row["strain_CRPS_ue_mean"])
        cur_d = float(row["disp_RMSE_mm_mean"])
        delta_s_rmse.append(cur_s - r_s)
        pct_s_rmse.append(100.0 * (cur_s - r_s) / max(r_s, 1e-6))
        pct_s_crps.append(100.0 * (cur_c - r_c) / max(r_c, 1e-6))
        pct_d_rmse.append(100.0 * (cur_d - r_d) / max(r_d, 1e-6))
    t1["delta_strain_RMSE_ue"] = delta_s_rmse
    t1["degradation_strain_RMSE_pct"] = pct_s_rmse
    t1["degradation_strain_CRPS_pct"] = pct_s_crps
    t1["degradation_disp_RMSE_pct"] = pct_d_rmse
    t1.to_csv(OUT_DIR / "ablation_table1_6components_M1.csv", index=False)

    # Table 2: Latent Mode Rank Sensitivity r in {1, 3, 5, 8, 15}
    t2 = df_agg[df_agg["ablation"] == "none"].sort_values(["regime", "mode_rank"]).reset_index(drop=True)
    t2.to_csv(OUT_DIR / "ablation_table2_mode_rank_sensitivity.csv", index=False)

    # Table 3: Unified Cross-Protocol Ablation Matrix (M1 5-Seed + M2 3-Fold Warm-OOD + D1 Cold-OOD)
    df_all = pd.DataFrame(rows_all)
    df_all.to_csv(OUT_DIR / "ablation_cross_protocol_all_seed_runs.csv", index=False)

    cp_rows = []
    pf_rows = []
    for ab in ab_keys:
        sub_m1_all = df_all[(df_all["split"] == "M1") & (df_all["ablation"] == ab) & (df_all["regime"] == "overall")]
        sub_m1_ood = df_all[(df_all["split"] == "M1") & (df_all["ablation"] == ab) & (df_all["regime"] == "oos_tail")]
        sub_m2_all = df_all[(df_all["split"].str.startswith("M2_")) & (df_all["ablation"] == ab) & (df_all["regime"] == "overall")]
        sub_m2_ood = df_all[(df_all["split"].str.startswith("M2_")) & (df_all["ablation"] == ab) & (df_all["regime"] == "oos_tail")]
        sub_d1_all = df_all[(df_all["split"] == "D1") & (df_all["ablation"] == ab) & (df_all["regime"] == "overall")]
        sub_d1_ood = df_all[(df_all["split"] == "D1") & (df_all["ablation"] == ab) & (df_all["regime"] == "oos_tail")]

        # M2 3-fold seed-level mean (matching Main Table 2 aggregation!)
        m2_all_seed = sub_m2_all.groupby("seed")["strain_RMSE_ue"].mean()
        m2_ood_rmse_seed = sub_m2_ood.groupby("seed")["strain_RMSE_ue"].mean()
        m2_ood_crps_seed = sub_m2_ood.groupby("seed")["strain_CRPS_ue"].mean()
        m2_ood_picp_seed = sub_m2_ood.groupby("seed")["strain_PICP95"].mean()
        m2_ood_mpiw_seed = sub_m2_ood.groupby("seed")["strain_MPIW95_ue"].mean()

        cp_rows.append({
            "ablation": ab,
            "variant_label": ab_lbl_map[ab],
            "num_params_M1": int(sub_m1_all["num_params"].iloc[0]),
            "M1_All_RMSE_mean": float(sub_m1_all["strain_RMSE_ue"].mean()),
            "M1_All_RMSE_std": float(sub_m1_all["strain_RMSE_ue"].std(ddof=0)),
            "M1_All_CRPS_mean": float(sub_m1_all["strain_CRPS_ue"].mean()),
            "M1_All_CRPS_std": float(sub_m1_all["strain_CRPS_ue"].std(ddof=0)),
            "M1_OOD_RMSE_mean": float(sub_m1_ood["strain_RMSE_ue"].mean()),
            "M1_OOD_RMSE_std": float(sub_m1_ood["strain_RMSE_ue"].std(ddof=0)),
            "M2_All_RMSE_mean": float(m2_all_seed.mean()),
            "M2_All_RMSE_std": float(m2_all_seed.std(ddof=0)),
            "M2_WarmOOD_RMSE_mean": float(m2_ood_rmse_seed.mean()),
            "M2_WarmOOD_RMSE_std": float(m2_ood_rmse_seed.std(ddof=0)),
            "M2_WarmOOD_CRPS_mean": float(m2_ood_crps_seed.mean()),
            "M2_WarmOOD_CRPS_std": float(m2_ood_crps_seed.std(ddof=0)),
            "M2_WarmOOD_PICP95_mean": float(m2_ood_picp_seed.mean()),
            "M2_WarmOOD_MPIW95_mean": float(m2_ood_mpiw_seed.mean()),
            "D1_All_Disp_RMSE_mean": float(sub_d1_all["disp_RMSE_mm"].mean()),
            "D1_All_Disp_RMSE_std": float(sub_d1_all["disp_RMSE_mm"].std(ddof=0)),
            "D1_All_Disp_CRPS_mean": float(sub_d1_all["disp_CRPS_mm"].mean()),
            "D1_All_Disp_CRPS_std": float(sub_d1_all["disp_CRPS_mm"].std(ddof=0)),
            "D1_ColdOOD_Disp_RMSE_mean": float(sub_d1_ood["disp_RMSE_mm"].mean()),
            "D1_ColdOOD_Disp_RMSE_std": float(sub_d1_ood["disp_RMSE_mm"].std(ddof=0)),
        })

        # Per-fold M2 OOD table
        pf = {"ablation": ab, "variant_label": ab_lbl_map[ab]}
        for fk in ["M2_fold1", "M2_fold2", "M2_fold3"]:
            sf = df_all[(df_all["split"] == fk) & (df_all["ablation"] == ab) & (df_all["regime"] == "oos_tail")]
            pf[f"{fk}_OOD_RMSE_mean"] = float(sf["strain_RMSE_ue"].mean())
            pf[f"{fk}_OOD_RMSE_std"] = float(sf["strain_RMSE_ue"].std(ddof=0))
            pf[f"{fk}_OOD_PICP95_mean"] = float(sf["strain_PICP95"].mean())
        pf_rows.append(pf)

    t3 = pd.DataFrame(cp_rows)
    ref3 = t3[t3["ablation"] == "none"].iloc[0]
    t3["deg_M1_All_pct"] = 100.0 * (t3["M1_All_RMSE_mean"] - ref3["M1_All_RMSE_mean"]) / ref3["M1_All_RMSE_mean"]
    t3["deg_M2_WarmOOD_pct"] = 100.0 * (t3["M2_WarmOOD_RMSE_mean"] - ref3["M2_WarmOOD_RMSE_mean"]) / ref3["M2_WarmOOD_RMSE_mean"]
    t3["deg_M2_OOD_CRPS_pct"] = 100.0 * (t3["M2_WarmOOD_CRPS_mean"] - ref3["M2_WarmOOD_CRPS_mean"]) / ref3["M2_WarmOOD_CRPS_mean"]
    t3["deg_D1_ColdOOD_pct"] = 100.0 * (t3["D1_ColdOOD_Disp_RMSE_mean"] - ref3["D1_ColdOOD_Disp_RMSE_mean"]) / ref3["D1_ColdOOD_Disp_RMSE_mean"]
    t3.to_csv(OUT_DIR / "ablation_table3_cross_protocol_M1_M2_D1.csv", index=False)

    t4 = pd.DataFrame(pf_rows)
    t4.to_csv(OUT_DIR / "ablation_table4_M2_per_fold.csv", index=False)
    print(f"[Ablation] Cross-protocol summary tables written to {OUT_DIR}")


if __name__ == "__main__":
    main()
