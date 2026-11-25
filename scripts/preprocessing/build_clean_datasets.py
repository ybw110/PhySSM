"""Build clean, real-timestamp-aligned 1-minute canonical datasets and split manifests for Paper 1 (PhySSM).

Fixes all legacy preprocessing issues:
1. Pairs exact calendar timestamp lines ('YYYY-MM-DD HH:MM:SS.mmm') with value lines (no row-index drift).
2. Preserves all 15 structural channels (2 displacements in mm + 13 individual strains in microstrain).
3. Computes 4-station spatial temperature gradients, humidity/pressure, 24-station wind statistics,
   zero-leakage (Train-only calibrated) 12/14 directional wind dynamic pressure proxies, and motor-activity masks.
4. Generates exact Split M1, Split M2 (3-fold rolling-origin), and Split D1 (2025 dynamic attitude) manifests
   and comprehensive CSV/JSON audit tables.
"""

from __future__ import annotations

import concurrent.futures
import io
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
RAW_2026_DIR = Path(os.environ.get("PHYSSM_RAW_2026_DIR", str(PROJECT_DIR / "raw_data" / "campaign_2026")))
RAW_2025_DIR = Path(os.environ.get("PHYSSM_RAW_2025_DIR", str(PROJECT_DIR / "raw_data" / "campaign_2025")))
DATA_DIR = PROJECT_DIR / "data"
OUT_2026_DIR = DATA_DIR / "processed_2026_1min"
OUT_2025_DIR = DATA_DIR / "processed_2025_1min"
META_DIR = DATA_DIR / "splits_and_metadata"

AIR_DENSITY = 1.225
ZENITH_AZ_DEG = 198.937
ZENITH_EL_DEG = 89.997
ANGLE_TOL_DEG = 0.02

DISP_COLS = ["49#01", "49#02"]
STRAIN_COLS = [f"49#{i:02d}" for i in range(3, 16)]
TEMP_STATIONS = ["06", "11", "13", "20"]
TEMP_COLS = [f"{s}_temperature" for s in TEMP_STATIONS]
HUM_COLS = [f"{s}_humidity" for s in TEMP_STATIONS]
PRESS_COLS = [f"{s}_airPressure" for s in TEMP_STATIONS]

TWO_ELEM_STATIONS = [
    "01", "02", "03", "05", "07", "08", "09", "10", "12", "14",
    "15", "16", "17", "18", "19", "21", "22", "23", "24", "25",
]
ALL_WIND_STATIONS = TWO_ELEM_STATIONS + TEMP_STATIONS
WIND_SPEED_COLS = [f"{s}_speed" for s in ALL_WIND_STATIONS]
KEY_WIND_STATIONS = ["12", "14", "15"]

CURRENT_COLS = ["az1_current", "az2_current", "el1_current", "el2_current"]
RELIABLE_SPEED_COLS = ["az2_speed", "el1_speed", "el2_speed"]

MODE_MAP = {
    "30": 30,
    "32": 32,
    "34": 34,
    "38": 38,
    "3a": 58,
    "3A": 58,
}

ELIGIBLE_2025_DATES = (
    "20251012", "20251013", "20251014", "20251015", "20251016",
    "20251017", "20251018", "20251019", "20251020", "20251022",
    "20251023", "20251024", "20251025", "20251026", "20251027",
)


def parse_mode_code(val) -> int:
    s = str(val).strip()
    if s in MODE_MAP:
        return MODE_MAP[s]
    try:
        return int(float(s))
    except Exception:
        try:
            return int(s, 16)
        except Exception:
            return -1


def wrap_to_180(deg: np.ndarray) -> np.ndarray:
    return (np.asarray(deg, dtype=np.float64) + 180.0) % 360.0 - 180.0


def parse_single_day_file(args: Tuple[str, str]) -> Tuple[pd.DataFrame, Dict]:
    """Stream-parse one daily raw TXT file by pairing exact timestamp lines and value lines."""
    file_path_str, batch_tag = args
    path = Path(file_path_str)
    date_compact = path.stem.replace("data", "")
    date_iso = f"{date_compact[:4]}-{date_compact[4:6]}-{date_compact[6:8]}"

    t0 = time.perf_counter()
    raw_pairs = 0
    malformed_lines = 0

    with path.open("r", encoding="utf-8", errors="replace") as f:
        header_line = f.readline().rstrip("\r\n")
        header_cols = header_line.split("\t")
        n_expected_cols = len(header_cols)

        # Target columns to keep per chunk (by exact header name)
        needed_cols = (
            ["work_mode", "az_angle", "el_angle", "az_com_angle", "el_com_angle"]
            + CURRENT_COLS
            + RELIABLE_SPEED_COLS
            + TEMP_COLS
            + HUM_COLS
            + PRESS_COLS
            + WIND_SPEED_COLS
            + [f"{s}_direction" for s in KEY_WIND_STATIONS]
            + DISP_COLS
            + STRAIN_COLS
        )
        col_indices = {c: header_cols.index(c) for c in needed_cols}
        use_indices = sorted(col_indices.values())
        use_names = [header_cols[i] for i in use_indices]

        # Read timestamp + value pairs safely in chunks
        CHUNK_PAIRS = 150_000
        ts_sec_list: List[np.ndarray] = []
        min_bin_list: List[np.ndarray] = []
        df_chunks: List[pd.DataFrame] = []

        buf_sec: List[float] = []
        buf_min: List[int] = []
        buf_val_lines: List[str] = []

        def flush_buffer():
            nonlocal buf_sec, buf_min, buf_val_lines
            if not buf_val_lines:
                return
            text_block = "".join(buf_val_lines)
            sub_df = pd.read_csv(
                io.StringIO(text_block),
                sep="\t",
                header=None,
                usecols=use_indices,
                names=use_names,
                converters={"work_mode": parse_mode_code},
                dtype={c: np.float64 for c in use_names if c != "work_mode"},
                engine="c",
                na_values=["", "nan", "NaN", "NULL"],
            )
            ts_sec_list.append(np.asarray(buf_sec, dtype=np.float64))
            min_bin_list.append(np.asarray(buf_min, dtype=np.int32))
            df_chunks.append(sub_df)
            buf_sec = []
            buf_min = []
            buf_val_lines = []

        pending_ts = None
        for line in f:
            # Timestamp line format: 'YYYY-MM-DD HH:MM:SS.mmm' (len ~ 23, no tabs)
            if "\t" not in line:
                s = line.strip()
                if len(s) >= 19 and s[4] == "-" and s[7] == "-" and s[13] == ":":
                    pending_ts = s
                elif s:
                    malformed_lines += 1
                    pending_ts = None
                continue
            else:
                if pending_ts is None:
                    malformed_lines += 1
                    continue
                # Verify column count quickly via tab count
                if line.count("\t") + 1 != n_expected_cols:
                    malformed_lines += 1
                    pending_ts = None
                    continue
                try:
                    hh = int(pending_ts[11:13])
                    mm = int(pending_ts[14:16])
                    ss = float(pending_ts[17:])
                except ValueError:
                    malformed_lines += 1
                    pending_ts = None
                    continue

                raw_pairs += 1
                buf_min.append(hh * 60 + mm)
                buf_sec.append(hh * 3600.0 + mm * 60.0 + ss)
                buf_val_lines.append(line)
                pending_ts = None

                if len(buf_val_lines) >= CHUNK_PAIRS:
                    flush_buffer()

        flush_buffer()

    all_sec = np.concatenate(ts_sec_list)
    all_min = np.concatenate(min_bin_list)
    raw_df = pd.concat(df_chunks, ignore_index=True)
    del df_chunks, ts_sec_list, min_bin_list

    # Compute sample-to-sample dt within the day (first sample dt = 0)
    dt_arr = np.zeros_like(all_sec)
    if len(all_sec) > 1:
        dt_arr[1:] = np.maximum(0.0, np.diff(all_sec))
    raw_df["minute_of_day"] = all_min
    raw_df["dt_sec"] = dt_arr
    raw_df["sec_of_day"] = all_sec

    # Precompute circular components for azimuth and key wind directions
    az_rad = np.deg2rad(raw_df["az_angle"].to_numpy(dtype=np.float64))
    raw_df["az_sin"] = np.sin(az_rad)
    raw_df["az_cos"] = np.cos(az_rad)

    for st in KEY_WIND_STATIONS:
        wdir_rad = np.deg2rad(raw_df[f"{st}_direction"].to_numpy(dtype=np.float64))
        raw_df[f"w{st}_sin"] = np.sin(wdir_rad)
        raw_df[f"w{st}_cos"] = np.cos(wdir_rad)

    # Motor max abs current & reliable speed per raw row
    cur_abs = raw_df[CURRENT_COLS].abs().max(axis=1)
    spd_abs = raw_df[RELIABLE_SPEED_COLS].abs().max(axis=1)
    raw_df["row_cur_abs_max"] = cur_abs
    raw_df["row_spd_abs_max"] = spd_abs
    raw_df["is_mode30"] = (raw_df["work_mode"] == 30).astype(np.float64)
    raw_df["is_mode34"] = (raw_df["work_mode"] == 34).astype(np.float64)
    raw_df["is_mode32"] = (raw_df["work_mode"] == 32).astype(np.float64)

    # In 2025 batch, strain froze after 2025-10-17 11:53:49.687 (42829.687 s)
    if batch_tag == "2025":
        if date_compact < "20251017":
            raw_df["strain_valid"] = 1.0
        elif date_compact == "20251017":
            raw_df["strain_valid"] = (raw_df["sec_of_day"] < 42829.0).astype(np.float64)
        else:
            raw_df["strain_valid"] = 0.0
    else:
        raw_df["strain_valid"] = 1.0

    # Group by exact calendar minute_of_day (0..1439)
    grp = raw_df.groupby("minute_of_day", sort=True)

    full_index = pd.Index(np.arange(1440, dtype=np.int32), name="minute_of_day")
    out = pd.DataFrame(index=full_index)
    out["date"] = date_iso
    out["timestamp"] = pd.date_range(start=f"{date_iso} 00:00:00", periods=1440, freq="1min")
    out["raw_count"] = grp.size().reindex(full_index, fill_value=0).astype(np.int32)
    out["max_dt_sec"] = grp["dt_sec"].max().reindex(full_index)

    # Mode fractions & dominant mode
    out["mode30_frac"] = grp["is_mode30"].mean().reindex(full_index)
    out["mode34_frac"] = grp["is_mode34"].mean().reindex(full_index)
    out["mode32_frac"] = grp["is_mode32"].mean().reindex(full_index)
    dom_mode = grp["work_mode"].agg(lambda x: int(x.mode().iloc[0]) if len(x) else -1)
    out["dominant_work_mode"] = dom_mode.reindex(full_index, fill_value=-1).astype(np.int32)

    # Attitude & kinematics
    az_sin_m = grp["az_sin"].mean().reindex(full_index)
    az_cos_m = grp["az_cos"].mean().reindex(full_index)
    out["az_sin"] = az_sin_m
    out["az_cos"] = az_cos_m
    out["az_angle_deg"] = (np.rad2deg(np.arctan2(az_sin_m, az_cos_m)) + 360.0) % 360.0
    out["el_angle_deg"] = grp["el_angle"].mean().reindex(full_index)
    el_rad_m = np.deg2rad(out["el_angle_deg"])
    out["el_sin"] = np.sin(el_rad_m)
    out["el_cos"] = np.cos(el_rad_m)

    # Within-minute motion range (handle azimuth wrap via circular spread or max-min)
    out["el_range_deg"] = (grp["el_angle"].max() - grp["el_angle"].min()).reindex(full_index)
    az_R = np.hypot(az_sin_m, az_cos_m)
    out["az_circular_R"] = az_R

    # Motor current & speed aggregates
    for c in CURRENT_COLS:
        out[f"{c}_mean"] = grp[c].mean().reindex(full_index)
        out[f"{c}_abs_max"] = grp[c].apply(lambda s: s.abs().max()).reindex(full_index)
    out["motor_current_abs_max"] = grp["row_cur_abs_max"].max().reindex(full_index)
    out["motor_speed_abs_max"] = grp["row_spd_abs_max"].max().reindex(full_index)
    out["motor_active_flag"] = (out["motor_current_abs_max"] > 0.05).fillna(False)

    # Clean fixed-attitude standby flag
    if batch_tag == "2026":
        az_diff = np.abs(wrap_to_180(out["az_angle_deg"] - ZENITH_AZ_DEG))
        el_diff = np.abs(out["el_angle_deg"] - ZENITH_EL_DEG)
        out["is_clean_standby"] = (
            (out["raw_count"] >= 100)
            & (out["mode30_frac"] >= 0.999)
            & (out["motor_current_abs_max"] <= 1e-6)
            & (out["motor_speed_abs_max"] <= 1e-6)
            & (az_diff <= ANGLE_TOL_DEG)
            & (el_diff <= ANGLE_TOL_DEG)
        )
    else:
        out["is_clean_standby"] = (
            (out["raw_count"] >= 100)
            & (out["mode30_frac"] >= 0.999)
            & (out["motor_current_abs_max"] <= 1e-6)
            & (out["motor_speed_abs_max"] <= 1e-6)
        )

    # 4-station Temperature, Humidity, Pressure & Spatial Gradients
    for s in TEMP_STATIONS:
        out[f"temp_{s}"] = grp[f"{s}_temperature"].mean().reindex(full_index)
        out[f"hum_{s}"] = grp[f"{s}_humidity"].mean().reindex(full_index)
        out[f"press_{s}"] = grp[f"{s}_airPressure"].mean().reindex(full_index)

    t_cols_clean = [f"temp_{s}" for s in TEMP_STATIONS]
    out["temp_mean"] = out[t_cols_clean].mean(axis=1)
    out["temp_spatial_std"] = out[t_cols_clean].std(axis=1, ddof=0)
    out["temp_spatial_range"] = out[t_cols_clean].max(axis=1) - out[t_cols_clean].min(axis=1)
    for s in TEMP_STATIONS:
        out[f"temp_grad_{s}"] = out[f"temp_{s}"] - out["temp_mean"]

    out["hum_mean"] = out[[f"hum_{s}" for s in TEMP_STATIONS]].mean(axis=1)
    out["press_mean"] = out[[f"press_{s}" for s in TEMP_STATIONS]].mean(axis=1)

    # 24-station Wind Field & Key Stations 12, 14, 15
    ws_means = grp[WIND_SPEED_COLS].mean().reindex(full_index)
    ws_mat = ws_means.to_numpy(dtype=np.float64)
    with np.errstate(all="ignore"):
        out["wind_speed_mean24"] = np.nanmean(ws_mat, axis=1)
        out["wind_speed_median24"] = np.nanmedian(ws_mat, axis=1)
        out["wind_speed_p90_24"] = np.nanpercentile(ws_mat, 90, axis=1)
        out["wind_speed_max24"] = np.nanmax(ws_mat, axis=1)
        out["wind_valid_stations"] = np.sum(np.isfinite(ws_mat), axis=1).astype(np.int32)

    for st in KEY_WIND_STATIONS:
        out[f"wind_{st}_speed"] = ws_means[f"{st}_speed"]
        out[f"wind_{st}_speed_sq"] = grp[f"{st}_speed"].apply(lambda x: np.mean(np.square(x))).reindex(full_index)
        s_m = grp[f"w{st}_sin"].mean().reindex(full_index)
        c_m = grp[f"w{st}_cos"].mean().reindex(full_index)
        out[f"wind_{st}_dir_deg"] = (np.rad2deg(np.arctan2(s_m, c_m)) + 360.0) % 360.0
        out[f"wind_{st}_dir_R"] = np.hypot(s_m, c_m)

    # 15 Target Structural Channels (2 Displacements mm + 13 Strains microstrain)
    out["disp_49_01"] = grp["49#01"].mean().reindex(full_index)
    out["disp_49_02"] = grp["49#02"].mean().reindex(full_index)
    out["disp_49_01_std"] = grp["49#01"].std().reindex(full_index)
    out["disp_49_02_std"] = grp["49#02"].std().reindex(full_index)

    for sc in STRAIN_COLS:
        cname = f"strain_{sc.replace('#', '_')}"
        out[cname] = grp[sc].mean().reindex(full_index)
        out[f"{cname}_std"] = grp[sc].std().reindex(full_index)

    strain_clean_cols = [f"strain_{sc.replace('#', '_')}" for sc in STRAIN_COLS]
    out["strain_median_13"] = out[strain_clean_cols].median(axis=1)
    out["strain_valid_frac"] = grp["strain_valid"].mean().reindex(full_index, fill_value=0.0)
    # Mask frozen strain minutes in 2025 to NaN in modeling columns if invalid
    invalid_strain = out["strain_valid_frac"] < 0.99
    if invalid_strain.any():
        out.loc[invalid_strain, strain_clean_cols + ["strain_median_13"]] = np.nan

    out = out.reset_index()
    elapsed = time.perf_counter() - t0

    valid_min = int((out["raw_count"] >= 100).sum())
    clean_standby_min = int(out["is_clean_standby"].sum())
    motor_active_min = int(out["motor_active_flag"].sum())

    summary = {
        "batch": batch_tag,
        "date": date_iso,
        "file_name": path.name,
        "n_columns": n_expected_cols,
        "raw_valid_records": int(raw_pairs),
        "malformed_lines": int(malformed_lines),
        "calendar_minutes": 1440,
        "valid_minutes_ge100": valid_min,
        "missing_or_sparse_minutes": 1440 - valid_min,
        "clean_standby_minutes": clean_standby_min,
        "motor_active_minutes": motor_active_min,
        "mode30_minutes": int((out["mode30_frac"] >= 0.5).sum()),
        "mode34_minutes": int((out["mode34_frac"] >= 0.5).sum()),
        "mode32_minutes": int((out["mode32_frac"] >= 0.5).sum()),
        "temp_min_degC": float(out["temp_mean"].min()),
        "temp_median_degC": float(out["temp_mean"].median()),
        "temp_max_degC": float(out["temp_mean"].max()),
        "temp_diurnal_range_degC": float(out["temp_mean"].max() - out["temp_mean"].min()),
        "temp_spatial_range_p95_degC": float(out["temp_spatial_range"].quantile(0.95)),
        "wind_median24_p50": float(out["wind_speed_median24"].median()),
        "wind_median24_p95": float(out["wind_speed_median24"].quantile(0.95)),
        "wind_12_14_max": float(
            0.5 * (out["wind_12_speed"] + out["wind_14_speed"]).max()
        ),
        "disp_01_median_mm": float(out["disp_49_01"].median()),
        "disp_02_median_mm": float(out["disp_49_02"].median()),
        "strain_valid_minutes": int((out["strain_valid_frac"] >= 0.99).sum()),
        "parse_seconds": round(elapsed, 2),
    }
    return out, summary


def add_zero_leakage_wind_proxy(df: pd.DataFrame, train_mask: pd.Series) -> Tuple[pd.DataFrame, float]:
    """Calibrate station 12 vs 14 angular offset strictly on Train split, then compute directional dynamic pressure."""
    df = df.copy()
    cal_mask = (
        train_mask
        & (df["raw_count"] >= 100)
        & (df["wind_12_speed"] >= 1.0)
        & (df["wind_14_speed"] >= 1.0)
        & (df["wind_12_dir_R"] >= 0.70)
        & (df["wind_14_dir_R"] >= 0.70)
    )
    raw_diff = wrap_to_180(df.loc[cal_mask, "wind_14_dir_deg"] - df.loc[cal_mask, "wind_12_dir_deg"])
    vec = np.exp(1j * np.deg2rad(raw_diff))
    offset_14_to_12_deg = float(np.rad2deg(np.angle(np.mean(vec))))

    dir12 = df["wind_12_dir_deg"].to_numpy(dtype=np.float64)
    dir14_corr = (df["wind_14_dir_deg"].to_numpy(dtype=np.float64) - offset_14_to_12_deg + 360.0) % 360.0
    disagreement = np.abs(wrap_to_180(dir14_corr - dir12))

    # Speed-weighted vector fusion of station 12 & calibrated station 14
    w12 = np.clip(df["wind_12_speed"].to_numpy(dtype=np.float64), 0.01, None)
    w14 = np.clip(df["wind_14_speed"].to_numpy(dtype=np.float64), 0.01, None)
    fx = w12 * np.cos(np.deg2rad(dir12)) + w14 * np.cos(np.deg2rad(dir14_corr))
    fy = w12 * np.sin(np.deg2rad(dir12)) + w14 * np.sin(np.deg2rad(dir14_corr))
    fused_dir = (np.rad2deg(np.arctan2(fy, fx)) + 360.0) % 360.0

    fused_speed = 0.5 * (df["wind_12_speed"] + df["wind_14_speed"])
    fused_speed_sq = 0.5 * (df["wind_12_speed_sq"] + df["wind_14_speed_sq"])

    rel_wind_deg = wrap_to_180(fused_dir - df["az_angle_deg"].to_numpy(dtype=np.float64))
    q_mag = 0.5 * AIR_DENSITY * fused_speed_sq

    wind_quality_ok = (
        (df["raw_count"] >= 100)
        & (fused_speed >= 1.0)
        & (df["wind_12_dir_R"] >= 0.70)
        & (df["wind_14_dir_R"] >= 0.70)
        & (disagreement <= 30.0)
    )

    df["wind_12_14_speed"] = fused_speed
    df["wind_12_14_dir_deg"] = fused_dir
    df["wind_12_14_disagreement_deg"] = disagreement
    df["wind_rel_az_deg"] = rel_wind_deg
    df["wind_q_total_Pa"] = q_mag
    df["wind_q_longitudinal_Pa"] = q_mag * np.cos(np.deg2rad(rel_wind_deg))
    df["wind_q_lateral_Pa"] = q_mag * np.sin(np.deg2rad(rel_wind_deg))
    df["wind_dir_quality_mask"] = wind_quality_ok.fillna(False)

    return df, offset_14_to_12_deg


def main():
    t_start = time.perf_counter()
    for d in (OUT_2026_DIR, OUT_2025_DIR, META_DIR):
        d.mkdir(parents=True, exist_ok=True)

    files_2026 = sorted(RAW_2026_DIR.glob("data2026*.txt"))
    files_2025 = [RAW_2025_DIR / f"data{dt}.txt" for dt in ELIGIBLE_2025_DATES]

    assert len(files_2026) == 45, f"Expected 45 files in 2026, got {len(files_2026)}"
    assert all(p.exists() for p in files_2025), "Some 2025 eligible files missing!"

    tasks = [(str(p), "2026") for p in files_2026] + [(str(p), "2025") for p in files_2025]
    max_workers = min(8, os.cpu_count() or 4)
    print(f"Launching {max_workers} parallel workers to parse {len(tasks)} daily files (~24 GB)...")

    results_2026: List[pd.DataFrame] = []
    results_2025: List[pd.DataFrame] = []
    daily_summaries: List[Dict] = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as ex:
        for idx, (df_day, summ) in enumerate(ex.map(parse_single_day_file, tasks), 1):
            daily_summaries.append(summ)
            if summ["batch"] == "2026":
                results_2026.append(df_day)
            else:
                results_2025.append(df_day)
            if idx % 10 == 0 or idx == len(tasks):
                print(
                    f"  [{idx:02d}/{len(tasks)}] {summ['date']} ({summ['batch']}) "
                    f"raw={summ['raw_valid_records']:,}, valid_min={summ['valid_minutes_ge100']}/1440 "
                    f"({summ['parse_seconds']}s)",
                    flush=True,
                )

    df26 = pd.concat(results_2026, ignore_index=True).sort_values(["date", "minute_of_day"]).reset_index(drop=True)
    df25 = pd.concat(results_2025, ignore_index=True).sort_values(["date", "minute_of_day"]).reset_index(drop=True)

    # =========================================================================
    # 1. Assign Split M1 & Split M2 on 2026 (45 days = 64,800 calendar minutes)
    # =========================================================================
    df26["split_M1"] = np.select(
        [
            df26["date"] <= "2026-05-16",
            (df26["date"] >= "2026-05-17") & (df26["date"] <= "2026-05-22"),
            df26["date"] >= "2026-05-23",
        ],
        ["train", "val", "test"],
        default="unknown",
    )

    # Calibrate wind proxy strictly on Split M1 Train
    df26, offset26_deg = add_zero_leakage_wind_proxy(df26, df26["split_M1"] == "train")

    # Compute exact Train and Dev (Train+Val) temperature support upper/lower bounds on valid minutes
    valid26 = df26["raw_count"] >= 100
    train_temp_max = float(df26.loc[valid26 & (df26["split_M1"] == "train"), "temp_mean"].max())
    train_temp_min = float(df26.loc[valid26 & (df26["split_M1"] == "train"), "temp_mean"].min())
    dev_temp_max = float(df26.loc[valid26 & df26["split_M1"].isin(["train", "val"]), "temp_mean"].max())
    dev_temp_min = float(df26.loc[valid26 & df26["split_M1"].isin(["train", "val"]), "temp_mean"].min())

    df26["exceeds_train_temp_max"] = valid26 & (df26["temp_mean"] > train_temp_max)
    df26["exceeds_dev_temp_max"] = valid26 & (df26["temp_mean"] > dev_temp_max)
    df26["test_support_regime"] = np.where(
        df26["split_M1"] != "test",
        df26["split_M1"],
        np.where(df26["exceeds_dev_temp_max"], "test_out_of_support", "test_in_support"),
    )

    # Split M2: 3-Fold Expanding Rolling-Origin Protocol
    m2_folds = [
        ("fold1", "2026-04-22", "2026-05-11", "2026-05-12", "2026-05-15", "2026-05-16", "2026-05-22"),
        ("fold2", "2026-04-22", "2026-05-18", "2026-05-19", "2026-05-22", "2026-05-23", "2026-05-29"),
        ("fold3", "2026-04-22", "2026-05-22", "2026-05-23", "2026-05-26", "2026-05-27", "2026-06-05"),
    ]
    for fname, tr_s, tr_e, va_s, va_e, te_s, te_e in m2_folds:
        df26[f"split_M2_{fname}"] = np.select(
            [
                (df26["date"] >= tr_s) & (df26["date"] <= tr_e),
                (df26["date"] >= va_s) & (df26["date"] <= va_e),
                (df26["date"] >= te_s) & (df26["date"] <= te_e),
            ],
            ["train", "val", "test"],
            default="unused",
        )

    # =========================================================================
    # 2. Assign Split D1 on 2025 (15 eligible days = 21,600 calendar minutes)
    # =========================================================================
    df25["split_D1"] = np.select(
        [
            df25["date"] <= "2025-10-20",
            (df25["date"] >= "2025-10-22") & (df25["date"] <= "2025-10-23"),
            df25["date"] >= "2025-10-24",
        ],
        ["train", "val", "test"],
        default="excluded",
    )
    df25, offset25_deg = add_zero_leakage_wind_proxy(df25, df25["split_D1"] == "train")

    valid25 = df25["raw_count"] >= 100
    dev25_temp_min = float(df25.loc[valid25 & df25["split_D1"].isin(["train", "val"]), "temp_mean"].min())
    df25["below_dev_temp_min"] = valid25 & (df25["temp_mean"] < dev25_temp_min)

    # =========================================================================
    # 3. Save Canonical Datasets & Metadata Summaries
    # =========================================================================
    csv_2026 = OUT_2026_DIR / "antenna_2026_45d_1min_canonical.csv"
    npz_2026 = OUT_2026_DIR / "antenna_2026_45d_1min_canonical.npz"
    df26.to_csv(csv_2026, index=False, encoding="utf-8-sig")
    np.savez_compressed(
        npz_2026,
        **{col: df26[col].astype(str).to_numpy() if df26[col].dtype == object or "datetime" in str(df26[col].dtype) else df26[col].to_numpy() for col in df26.columns}
    )

    csv_2025 = OUT_2025_DIR / "antenna_2025_15d_1min_canonical.csv"
    npz_2025 = OUT_2025_DIR / "antenna_2025_15d_1min_canonical.npz"
    df25.to_csv(csv_2025, index=False, encoding="utf-8-sig")
    np.savez_compressed(
        npz_2025,
        **{col: df25[col].astype(str).to_numpy() if df25[col].dtype == object or "datetime" in str(df25[col].dtype) else df25[col].to_numpy() for col in df25.columns}
    )

    daily_df = pd.DataFrame(daily_summaries).sort_values(["batch", "date"]).reset_index(drop=True)
    daily_df.to_csv(META_DIR / "daily_audit_summary_60d.csv", index=False, encoding="utf-8-sig")

    # Build Split Summary Table
    split_stats_rows = []
    for sp in ["train", "val", "test"]:
        sub = df26[df26["split_M1"] == sp]
        vsub = sub[sub["raw_count"] >= 100]
        csub = sub[sub["is_clean_standby"]]
        split_stats_rows.append({
            "protocol": "Split_M1_2026",
            "subset": sp,
            "days": int(sub["date"].nunique()),
            "date_range": f"{sub['date'].min()} ~ {sub['date'].max()}",
            "calendar_minutes": int(len(sub)),
            "valid_minutes": int(len(vsub)),
            "clean_standby_minutes": int(len(csub)),
            "motor_active_minutes": int(vsub["motor_active_flag"].sum()),
            "raw_records_sum": int(sub["raw_count"].sum()),
            "temp_min_C": round(float(vsub["temp_mean"].min()), 3),
            "temp_p50_C": round(float(vsub["temp_mean"].median()), 3),
            "temp_p95_C": round(float(vsub["temp_mean"].quantile(0.95)), 3),
            "temp_max_C": round(float(vsub["temp_mean"].max()), 3),
            "out_of_dev_support_minutes": int(vsub["exceeds_dev_temp_max"].sum()),
            "out_of_train_support_minutes": int(vsub["exceeds_train_temp_max"].sum()),
        })

    # Add In-Support vs Out-of-Support sub-rows for M1 Test
    for reg in ["test_in_support", "test_out_of_support"]:
        vsub = df26[(df26["raw_count"] >= 100) & (df26["test_support_regime"] == reg)]
        csub = vsub[vsub["is_clean_standby"]]
        split_stats_rows.append({
            "protocol": "Split_M1_2026_TestBreakdown",
            "subset": reg,
            "days": int(vsub["date"].nunique()),
            "date_range": f"{vsub['date'].min()} ~ {vsub['date'].max()}" if len(vsub) else "-",
            "calendar_minutes": int(len(vsub)),
            "valid_minutes": int(len(vsub)),
            "clean_standby_minutes": int(len(csub)),
            "motor_active_minutes": int(vsub["motor_active_flag"].sum()),
            "raw_records_sum": int(vsub["raw_count"].sum()),
            "temp_min_C": round(float(vsub["temp_mean"].min()), 3) if len(vsub) else np.nan,
            "temp_p50_C": round(float(vsub["temp_mean"].median()), 3) if len(vsub) else np.nan,
            "temp_p95_C": round(float(vsub["temp_mean"].quantile(0.95)), 3) if len(vsub) else np.nan,
            "temp_max_C": round(float(vsub["temp_mean"].max()), 3) if len(vsub) else np.nan,
            "out_of_dev_support_minutes": int(vsub["exceeds_dev_temp_max"].sum()),
            "out_of_train_support_minutes": int(vsub["exceeds_train_temp_max"].sum()),
        })

    # Split M2 rows
    for fname, *_ in m2_folds:
        col = f"split_M2_{fname}"
        for sp in ["train", "val", "test"]:
            sub = df26[df26[col] == sp]
            vsub = sub[sub["raw_count"] >= 100]
            dev_max_f = float(df26.loc[(df26["raw_count"] >= 100) & df26[col].isin(["train", "val"]), "temp_mean"].max())
            split_stats_rows.append({
                "protocol": f"Split_M2_{fname}",
                "subset": sp,
                "days": int(sub["date"].nunique()),
                "date_range": f"{sub['date'].min()} ~ {sub['date'].max()}",
                "calendar_minutes": int(len(sub)),
                "valid_minutes": int(len(vsub)),
                "clean_standby_minutes": int(sub["is_clean_standby"].sum()),
                "motor_active_minutes": int(vsub["motor_active_flag"].sum()),
                "raw_records_sum": int(sub["raw_count"].sum()),
                "temp_min_C": round(float(vsub["temp_mean"].min()), 3),
                "temp_p50_C": round(float(vsub["temp_mean"].median()), 3),
                "temp_p95_C": round(float(vsub["temp_mean"].quantile(0.95)), 3),
                "temp_max_C": round(float(vsub["temp_mean"].max()), 3),
                "out_of_dev_support_minutes": int((vsub["temp_mean"] > dev_max_f).sum()),
                "out_of_train_support_minutes": int((vsub["temp_mean"] > train_temp_max).sum()),
            })

    # Split D1 rows (2025)
    for sp in ["train", "val", "test"]:
        sub = df25[df25["split_D1"] == sp]
        vsub = sub[sub["raw_count"] >= 100]
        m34 = vsub[vsub["mode34_frac"] >= 0.9]
        split_stats_rows.append({
            "protocol": "Split_D1_2025",
            "subset": sp,
            "days": int(sub["date"].nunique()),
            "date_range": f"{sub['date'].min()} ~ {sub['date'].max()}",
            "calendar_minutes": int(len(sub)),
            "valid_minutes": int(len(vsub)),
            "clean_standby_minutes": int(sub["is_clean_standby"].sum()),
            "motor_active_minutes": int(len(m34)),  # here mode34 tracking minutes stored for quick check
            "raw_records_sum": int(sub["raw_count"].sum()),
            "temp_min_C": round(float(vsub["temp_mean"].min()), 3),
            "temp_p50_C": round(float(vsub["temp_mean"].median()), 3),
            "temp_p95_C": round(float(vsub["temp_mean"].quantile(0.95)), 3),
            "temp_max_C": round(float(vsub["temp_mean"].max()), 3),
            "out_of_dev_support_minutes": int(vsub["below_dev_temp_min"].sum()),
            "out_of_train_support_minutes": int(len(m34)),
        })

    split_df = pd.DataFrame(split_stats_rows)
    split_df.to_csv(META_DIR / "split_manifest_summary.csv", index=False, encoding="utf-8-sig")

    # Channel statistics on 2026 Valid Minutes (Train vs Test)
    target_channels = ["disp_49_01", "disp_49_02"] + [f"strain_49_{i:02d}" for i in range(3, 16)]
    ch_rows = []
    v26_tr = df26[(df26["raw_count"] >= 100) & (df26["split_M1"] == "train")]
    v26_te = df26[(df26["raw_count"] >= 100) & (df26["split_M1"] == "test")]
    for ch in target_channels:
        unit = "mm" if ch.startswith("disp") else "microstrain"
        r_tr = np.corrcoef(v26_tr["temp_mean"], v26_tr[ch])[0, 1]
        ch_rows.append({
            "channel": ch,
            "unit": unit,
            "train_mean": round(float(v26_tr[ch].mean()), 4),
            "train_std": round(float(v26_tr[ch].std()), 4),
            "train_p05": round(float(v26_tr[ch].quantile(0.05)), 4),
            "train_p95": round(float(v26_tr[ch].quantile(0.95)), 4),
            "test_mean": round(float(v26_te[ch].mean()), 4),
            "test_std": round(float(v26_te[ch].std()), 4),
            "test_p05": round(float(v26_te[ch].quantile(0.05)), 4),
            "test_p95": round(float(v26_te[ch].quantile(0.95)), 4),
            "corr_with_temp_train": round(float(r_tr), 4),
        })
    pd.DataFrame(ch_rows).to_csv(META_DIR / "channel_statistics_2026_M1.csv", index=False, encoding="utf-8-sig")

    meta_json = {
        "created_at": "2026-11-17",
        "total_parse_seconds": round(time.perf_counter() - t_start, 2),
        "2026_total_raw_records": int(df26["raw_count"].sum()),
        "2026_calendar_minutes": int(len(df26)),
        "2026_valid_minutes_ge100": int((df26["raw_count"] >= 100).sum()),
        "2026_clean_standby_minutes": int(df26["is_clean_standby"].sum()),
        "2026_motor_active_minutes": int(df26.loc[df26["raw_count"] >= 100, "motor_active_flag"].sum()),
        "M1_train_temp_range_C": [round(train_temp_min, 3), round(train_temp_max, 3)],
        "M1_dev_temp_range_C": [round(dev_temp_min, 3), round(dev_temp_max, 3)],
        "M1_test_temp_range_C": [
            round(float(v26_te["temp_mean"].min()), 3),
            round(float(v26_te["temp_mean"].max()), 3),
        ],
        "M1_test_max_single_station_temp_C": round(
            float(v26_te[["temp_06", "temp_11", "temp_13", "temp_20"]].max().max()), 3
        ),
        "M1_test_out_of_dev_support_minutes": int(v26_te["exceeds_dev_temp_max"].sum()),
        "M1_test_out_of_dev_support_pct": round(100.0 * float(v26_te["exceeds_dev_temp_max"].mean()), 2),
        "M1_test_out_of_train_support_minutes": int(v26_te["exceeds_train_temp_max"].sum()),
        "M1_test_out_of_train_support_pct": round(100.0 * float(v26_te["exceeds_train_temp_max"].mean()), 2),
        "wind_14_to_12_train_offset_2026_deg": round(offset26_deg, 3),
        "wind_14_to_12_train_offset_2025_deg": round(offset25_deg, 3),
        "2025_total_raw_records": int(df25["raw_count"].sum()),
        "2025_calendar_minutes": int(len(df25)),
        "2025_valid_minutes_ge100": int((df25["raw_count"] >= 100).sum()),
        "2025_mode34_minutes": int((df25.loc[df25["raw_count"] >= 100, "mode34_frac"] >= 0.9).sum()),
        "D1_dev_temp_min_C": round(dev25_temp_min, 3),
        "D1_test_temp_min_C": round(
            float(df25.loc[(df25["raw_count"] >= 100) & (df25["split_D1"] == "test"), "temp_mean"].min()), 3
        ),
    }
    with (META_DIR / "dataset_build_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta_json, f, ensure_ascii=False, indent=2)

    print("\n=== BUILD COMPLETED SUCCESSFULLY ===")
    print(json.dumps(meta_json, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
