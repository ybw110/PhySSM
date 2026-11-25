"""Multi-GPU Asynchronous Job Queue Runner for Paper 1 (Full Seed Parity, All-Seed Conformal/Ensemble Predictions)."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

EXP_ROOT = Path(__file__).resolve().parents[1]
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from src.models import NON_TORCH_MODELS


def build_job_manifest(stage: str, cfg: Dict, only_physsm: bool = False) -> List[Dict]:
    models = ["PhySSM"] if only_physsm else cfg["models_order"]
    seeds_5 = cfg["primary_seeds"]
    seeds_3 = cfg["rolling_and_dynamic_seeds"]
    jobs: List[Dict] = []

    if stage in ("all", "main_m1", "main_all"):
        for s_idx, s in enumerate(seeds_5):
            for m in models:
                if m in NON_TORCH_MODELS and s_idx > 0:
                    continue
                jobs.append({
                    "split": "M1",
                    "model": m,
                    "seed": s,
                    "horizon": 0,
                    "ablation": "none",
                    "mode_rank": 5,
                    # Save all 5 seeds on Split M1 for 5-Seed Deep Ensembles, Conformal Calibration & Pooled Bootstrap!
                    "save_predictions": True,
                })

    if stage in ("all", "rolling_and_d1", "main_all"):
        for sp in ["M2_fold1", "M2_fold2", "M2_fold3", "D1"]:
            r_def = 2 if sp == "D1" else 5
            for m in models:
                m_seeds = [42] if m in NON_TORCH_MODELS else seeds_3
                for s in m_seeds:
                    jobs.append({
                        "split": sp,
                        "model": m,
                        "seed": s,
                        "horizon": 0,
                        "ablation": "none",
                        "mode_rank": r_def,
                        "save_predictions": (s == 42 and sp in ("M2_fold1", "D1")),
                    })

    if stage in ("all", "horizons", "main_all"):
        rep_models = ["PhySSM"] if only_physsm else ["Composite-GPR", "N4SID", "Causal-Dilated TCN", "TimeXer", "TimePro", "PhySSM"]
        for hz in cfg["lookahead_horizons_min"]:
            for m in rep_models:
                m_seeds = [42] if m in NON_TORCH_MODELS else seeds_3
                for s in m_seeds:
                    jobs.append({
                        "split": "M1",
                        "model": m,
                        "seed": s,
                        "horizon": hz,
                        "ablation": "none",
                        "mode_rank": 5,
                        "save_predictions": False,
                    })

    if stage in ("all", "ablations"):
        ablations = [
            "wo_dissipative_ssm",
            "wo_wind_modulation",
            "wo_spatial_grad",
            "wo_low_rank_modes",
            "wo_homogeneous_dual_stream",
            "wo_actuation_quality_gate",
        ]
        for ab in ablations:
            for s in seeds_5:
                jobs.append({
                    "split": "M1",
                    "model": "PhySSM",
                    "seed": s,
                    "horizon": 0,
                    "ablation": ab,
                    "mode_rank": 5,
                    "save_predictions": (s == 42),
                })
        # Full 5-Seed Parity across ALL modal ranks r in {1, 3, 5, 8, 15} (eliminating 5-vs-3 seed asymmetry!)
        for rk in [1, 3, 8, 15]:
            for s in seeds_5:
                jobs.append({
                    "split": "M1",
                    "model": "PhySSM",
                    "seed": s,
                    "horizon": 0,
                    "ablation": "none",
                    "mode_rank": rk,
                    "save_predictions": False,
                })

    if stage in ("all", "ablations_ood"):
        ablations = [
            "wo_dissipative_ssm",
            "wo_wind_modulation",
            "wo_spatial_grad",
            "wo_low_rank_modes",
            "wo_homogeneous_dual_stream",
            "wo_actuation_quality_gate",
        ]
        for sp in ["M2_fold1", "M2_fold2", "M2_fold3", "D1"]:
            r_def = 2 if sp == "D1" else 5
            for ab in ablations:
                for s in seeds_3:
                    jobs.append({
                        "split": sp,
                        "model": "PhySSM",
                        "seed": s,
                        "horizon": 0,
                        "ablation": ab,
                        "mode_rank": r_def,
                        "save_predictions": False,
                    })

    if stage == "export_checkpoints":
        # Complete pretrained checkpoint portfolio (41 self-contained .pt files):
        # 1) PhySSM Full Suite (20 .pt): M1 all 5 seeds (single + 5-seed ensemble),
        #    M1 multi-horizon H=10/20/30 (seed 42), M2 Fold 1..3 all 3 seeds, D1 all 3 seeds
        for s in seeds_5:
            jobs.append({
                "split": "M1", "model": "PhySSM", "seed": s, "horizon": 0,
                "ablation": "none", "mode_rank": 5, "save_predictions": False, "save_checkpoint": True,
            })
        for hz in cfg["lookahead_horizons_min"]:
            jobs.append({
                "split": "M1", "model": "PhySSM", "seed": 42, "horizon": hz,
                "ablation": "none", "mode_rank": 5, "save_predictions": False, "save_checkpoint": True,
            })
        for sp in ["M2_fold1", "M2_fold2", "M2_fold3", "D1"]:
            r_def = 2 if sp == "D1" else 5
            for s in seeds_3:
                jobs.append({
                    "split": sp, "model": "PhySSM", "seed": s, "horizon": 0,
                    "ablation": "none", "mode_rank": r_def, "save_predictions": False, "save_checkpoint": True,
                })
        # 2) All 7 Baseline Models Representative Suite (21 .pt): M1, M2_fold1, D1 (seed 42)
        if not only_physsm:
            for sp in ["M1", "M2_fold1", "D1"]:
                r_def = 2 if sp == "D1" else 5
                for m in cfg["models_order"]:
                    if m == "PhySSM":
                        continue
                    jobs.append({
                        "split": sp, "model": m, "seed": 42, "horizon": 0,
                        "ablation": "none", "mode_rank": r_def, "save_predictions": False, "save_checkpoint": True,
                    })

    return jobs


def job_to_run_id(job: Dict) -> str:
    safe_m = job["model"].replace(" ", "_").replace("-", "_")
    return f"{job['split']}_H{job['horizon']}_{safe_m}_ab_{job['ablation']}_r{job['mode_rank']}_s{job['seed']}"


def job_is_up_to_date(j_path: Path, p_path: Path, ckpt_path: Path, need_preds: bool, need_ckpt: bool) -> bool:
    if not j_path.exists():
        return False
    try:
        meta = json.loads(j_path.read_text(encoding="utf-8"))
        # Require new Normalized Split-Conformal Calibration field `conformal_q95`
        if "conformal_q95" not in meta:
            return False
    except Exception:
        return False
    if need_preds:
        if not p_path.exists():
            return False
        try:
            z = list(p_path.exists() and __import__("numpy").load(p_path, allow_pickle=True).keys())
            if "val_mu_phys" not in z:
                return False
        except Exception:
            return False
    if need_ckpt and not ckpt_path.exists():
        return False
    return True


def execute_parallel_queue(
    jobs: List[Dict],
    num_gpus: int,
    workers_per_gpu: int,
    data_root: Path,
    out_root: Path,
    skip_existing: bool = True,
    force_save_checkpoints: bool = False,
) -> None:
    pending = []
    for j in jobs:
        need_ckpt = bool(j.get("save_checkpoint", False) or force_save_checkpoints)
        j = dict(j, save_checkpoint=need_ckpt)
        rid = job_to_run_id(j)
        j_path = out_root / "raw_runs" / f"{rid}.json"
        p_path = out_root / "predictions" / f"{rid}_preds.npz"
        c_path = out_root / "checkpoints" / f"{rid}.pt"
        if skip_existing and job_is_up_to_date(j_path, p_path, c_path, j["save_predictions"], need_ckpt):
            continue
        pending.append(j)

    total_slots = num_gpus * workers_per_gpu
    total = len(pending)
    print(
        f"[QUEUE] Total jobs requested: {len(jobs)} | Pending: {total} across {num_gpus} GPUs x {workers_per_gpu} slots = {total_slots} workers",
        flush=True,
    )
    if total == 0:
        return

    runner_script = Path(__file__).resolve().parent / "run_single_experiment.py"
    active: Dict[int, Tuple[subprocess.Popen, Dict]] = {}
    q_idx, done_cnt, fail_cnt = 0, 0, 0
    t0 = time.time()

    while q_idx < total or active:
        for slot_id in range(total_slots):
            if slot_id not in active and q_idx < total:
                gpu_id = slot_id % num_gpus
                job = pending[q_idx]
                q_idx += 1
                cmd = [
                    sys.executable,
                    str(runner_script),
                    "--split", job["split"],
                    "--model", job["model"],
                    "--seed", str(job["seed"]),
                    "--horizon", str(job["horizon"]),
                    "--ablation", job["ablation"],
                    "--mode_rank", str(job["mode_rank"]),
                    "--device", f"cuda:{gpu_id}",
                    "--data_root", str(data_root),
                    "--out_root", str(out_root),
                ]
                if job["save_predictions"]:
                    cmd.append("--save_predictions")
                if job.get("save_checkpoint", False):
                    cmd.append("--save_checkpoint")
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                active[slot_id] = (proc, job)

        finished_slots = []
        for slot_id, (proc, job) in active.items():
            ret = proc.poll()
            if ret is not None:
                out_txt, _ = proc.communicate()
                finished_slots.append(slot_id)
                gpu_id = slot_id % num_gpus
                if ret == 0:
                    done_cnt += 1
                    last_line = out_txt.strip().splitlines()[-1] if out_txt.strip() else "OK"
                    print(f"[{done_cnt}/{total} | GPU {gpu_id}] {last_line}", flush=True)
                else:
                    fail_cnt += 1
                    print(f"[ERROR GPU {gpu_id}] {job_to_run_id(job)} exited {ret}:\n{out_txt[-800:]}", flush=True)

        for sl in finished_slots:
            del active[sl]
        if active:
            time.sleep(0.6)

    print(f"[QUEUE COMPLETE] Finished {done_cnt}/{total} jobs (failures={fail_cnt}) in {time.time() - t0:.1f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-GPU Queue Runner")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "main_all", "main_m1", "rolling_and_d1", "ablations", "ablations_ood", "horizons", "export_checkpoints"],
    )
    parser.add_argument("--num_gpus", type=int, default=4)
    parser.add_argument("--workers_per_gpu", type=int, default=2)
    parser.add_argument("--only_physsm", action="store_true")
    parser.add_argument("--save_checkpoints", action="store_true")
    parser.add_argument("--data_root", type=str, default="")
    parser.add_argument("--out_root", type=str, default="")
    parser.add_argument("--no_skip", action="store_true")
    args = parser.parse_args()

    cfg = json.loads((EXP_ROOT / "configs" / "default_benchmark_config.json").read_text(encoding="utf-8"))
    data_root = Path(args.data_root) if args.data_root else (EXP_ROOT / "data")
    out_root = Path(args.out_root) if args.out_root else (EXP_ROOT / "outputs" / "raw_cache")
    out_root.mkdir(parents=True, exist_ok=True)

    jobs = build_job_manifest(args.stage, cfg, only_physsm=args.only_physsm)
    execute_parallel_queue(
        jobs,
        num_gpus=args.num_gpus,
        workers_per_gpu=args.workers_per_gpu,
        data_root=data_root,
        out_root=out_root,
        skip_existing=not args.no_skip,
        force_save_checkpoints=args.save_checkpoints,
    )


if __name__ == "__main__":
    main()
