from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


REPO = Path(__file__).resolve().parents[3]
OUT_ROOT = REPO / "ladd/results/converged_mainline_ladd_20260613"
FIG_DIR = REPO / "docs/experiments/figures/ladd_converged_mainline_ladd_20260613"
DOC_PATH = REPO / "docs/experiments/LADD_CONVERGED_MAINLINE_COMPARISON_20260613_CN.md"
SUMMARY_PATH = OUT_ROOT / "converged_mainline_ladd_summary_20260613.csv"
DOC_SUMMARY_PATH = REPO / "docs/experiments/ladd_converged_mainline_ladd_summary_20260613.csv"
PHASE_SUMMARY_PATH = OUT_ROOT / "converged_mainline_ladd_phase_summary_20260613.csv"
DOC_PHASE_SUMMARY_PATH = REPO / "docs/experiments/ladd_converged_mainline_ladd_phase_summary_20260613.csv"


BASELINES = {
    ("formal_nomosaic", "n", 0): {"best": 0.55654, "final": 0.55127},
    ("formal_nomosaic", "n", 42): {"best": 0.55794, "final": 0.55444},
    ("formal_nomosaic", "n", 123): {"best": 0.56128, "final": 0.56076},
    ("formal_nomosaic", "s", 0): {"best": 0.62897, "final": 0.62233},
    ("formal_nomosaic", "m", 0): {"best": 0.65580, "final": 0.64903},
    ("mosaic100", "n", 0): {"best": 0.54091, "final": 0.53836},
    ("mosaic100", "n", 42): {"best": 0.54091, "final": 0.53836},
    ("mosaic100", "n", 123): {"best": 0.54091, "final": 0.53836},
}


@dataclass(frozen=True)
class RunSpec:
    key: str
    label: str
    protocol: str
    server: str
    model: str
    seed: int
    method: str
    bn_stats: str
    status: str
    path: str
    notes: str


@dataclass(frozen=True)
class ChainSpec:
    key: str
    label: str
    protocol: str
    model: str
    seed: int
    method: str
    status: str
    a1_path: str
    a2_path: str
    b_path: str
    notes: str


RUNS = [
    RunSpec(
        "mosaic_legacy_s0",
        "mosaic100 legacy s0",
        "mosaic100",
        "90",
        "n",
        0,
        "legacy",
        "normal",
        "complete",
        "ladd/results/converged_mainline_ladd_20260613/source/mosaic90/ladd_b_runs/"
        "ladd_hbb_ogsod11n_ladd800r2_legacy_s0_b_e800_b64_s0_gpu2/results.csv",
        "old close@100 protocol; mosaic open for first 100 epochs; no collapse",
    ),
    RunSpec(
        "mosaic_legacy_s42",
        "mosaic100 legacy s42",
        "mosaic100",
        "90",
        "n",
        42,
        "legacy",
        "normal",
        "complete",
        "ladd/results/converged_mainline_ladd_20260613/source/mosaic90/ladd_b_runs/"
        "ladd_hbb_ogsod11n_ladd800r2_legacy_s42_b_e800_b64_s42_gpu5/results.csv",
        "old close@100 protocol; no collapse",
    ),
    RunSpec(
        "mosaic_legacy_s123",
        "mosaic100 legacy s123",
        "mosaic100",
        "90",
        "n",
        123,
        "legacy",
        "normal",
        "complete",
        "ladd/results/converged_mainline_ladd_20260613/source/mosaic90/ladd_b_runs/"
        "ladd_hbb_ogsod11n_ladd800r2_legacy_s123_b_e800_b64_s123_gpu3/results.csv",
        "old close@100 protocol; no collapse",
    ),
    RunSpec(
        "mosaic_cap2_s0",
        "mosaic100 cap2 s0",
        "mosaic100",
        "90",
        "n",
        0,
        "cap2",
        "normal",
        "complete",
        "ladd/results/converged_mainline_ladd_20260613/source/mosaic90/ladd_b_runs/"
        "ladd_hbb_ogsod11n_ladd800r2_cap2_s0_b_e800_b64_s0_gpu4/results.csv",
        "largest historical LADD gain under old close@100 protocol",
    ),
    RunSpec(
        "mosaic_cap2_s42",
        "mosaic100 cap2 s42",
        "mosaic100",
        "90",
        "n",
        42,
        "cap2",
        "normal",
        "complete",
        "ladd/results/converged_mainline_ladd_20260613/source/mosaic90/ladd_b_runs/"
        "ladd_hbb_ogsod11n_ladd800r2_cap2_s42_b_e800_b64_s42_gpu3/results.csv",
        "old close@100 protocol; no collapse",
    ),
    RunSpec(
        "mosaic_cap2_s123",
        "mosaic100 cap2 s123",
        "mosaic100",
        "90",
        "n",
        123,
        "cap2",
        "normal",
        "complete",
        "ladd/results/converged_mainline_ladd_20260613/source/mosaic90/ladd_b_runs/"
        "ladd_hbb_ogsod11n_ladd800r2_cap2_s123_b_e800_b64_s123_gpu5/results.csv",
        "old close@100 protocol; no collapse",
    ),
    RunSpec(
        "nomosaic_original_s0",
        "no-mosaic original/no-cap2 s0",
        "formal_nomosaic",
        "90",
        "n",
        0,
        "original",
        "normal",
        "complete",
        "ladd/results/converged_mainline_ladd_20260613/source/no_mosaic_90/n_original/"
        "ladd_hbb_ogsod11n_formal_nomosaic_yolo11n_original_s0_a2mu1e3_b_e800_b64_s0_gpu4/results.csv",
        "highest n no-mosaic LADD best AP; kept as ablation/diagnostic, not final mainline",
    ),
    RunSpec(
        "nomosaic_cap2_s0_a2mu",
        "no-mosaic cap2 s0 no-BN-freeze",
        "formal_nomosaic",
        "90",
        "n",
        0,
        "cap2_a2mu1e3",
        "normal",
        "complete",
        "ladd/results/converged_mainline_ladd_20260613/source/no_mosaic_90/n_cap2/"
        "ladd_hbb_ogsod11n_formal_nomosaic_yolo11n_cap2_s0_a2mu1e3_b_e800_b64_s0_gpu6/results.csv",
        "healthy no-BN-freeze seed0; strong best/final",
    ),
    RunSpec(
        "nomosaic_cap2_s42_a2mu",
        "no-mosaic cap2 s42 no-BN-freeze",
        "formal_nomosaic",
        "90",
        "n",
        42,
        "cap2_a2mu1e3",
        "normal",
        "complete",
        "ladd/results/converged_mainline_ladd_20260613/source/no_mosaic_90/n_cap2/"
        "ladd_hbb_ogsod11n_formal_nomosaic_yolo11n_cap2_s42_a2mu1e3_b_e800_b64_s42_gpu4/results.csv",
        "healthy no-BN-freeze seed42; positive but below seed0",
    ),
    RunSpec(
        "nomosaic_cap2_s123_old_crash",
        "no-mosaic cap2 s123 old-B crash",
        "formal_nomosaic",
        "90",
        "n",
        123,
        "cap2_old_b",
        "normal",
        "nan_crash",
        "ladd/results/converged_mainline_ladd_20260613/source/no_mosaic_90/n_cap2/"
        "ladd_hbb_ogsod11n_formal_nomosaic_yolo11n_cap2_s123_a2mu1e3_b_e800_b64_s123_gpu5/results.csv",
        "old B default/high-LR path; detection loss NaN around epoch 429 and final AP collapses to 0",
    ),
    RunSpec(
        "nomosaic_cap2_s123_bstable",
        "no-mosaic cap2 s123 B-lr1e-3 no-BN-freeze",
        "formal_nomosaic",
        "90",
        "n",
        123,
        "cap2_bstable1e3",
        "normal",
        "late_regression",
        "ladd/results/converged_mainline_ladd_20260613/source/no_mosaic_90/n_cap2/"
        "ladd_hbb_ogsod11n_formal_nomosaic_yolo11n_cap2_s123_a2mu1e3_bstable1e3_b_e800_b64_s123_gpu2/results.csv",
        "MuSGD lr1e-3 prevents NaN but final severely regresses",
    ),
    RunSpec(
        "nomosaic_cap2_s0_bnfreeze",
        "no-mosaic cap2 s0 BN-freeze",
        "formal_nomosaic",
        "90",
        "n",
        0,
        "cap2_bnfreeze1e3",
        "freeze",
        "complete",
        "ladd/results/converged_mainline_ladd_20260613/source/no_mosaic_90/n_cap2/"
        "ladd_hbb_ogsod11n_formal_nomosaic_yolo11n_cap2_s0_bnfreeze1e3_90_gpu7_b_e800_b64_s0_gpu7/results.csv",
        "stable BN-freeze candidate; slightly lower peak than healthy no-freeze seed0",
    ),
    RunSpec(
        "nomosaic_cap2_s42_bnfreeze",
        "no-mosaic cap2 s42 BN-freeze",
        "formal_nomosaic",
        "dual4090",
        "n",
        42,
        "cap2_bnfreeze1e3",
        "freeze",
        "complete",
        "ladd/results/converged_mainline_ladd_20260613/source/no_mosaic_4090dual/n_cap2/"
        "ladd_hbb_ogsod11n_formal_nomosaic_yolo11n_cap2_s42_bnfreeze1e3_public4090dual_final_v2_b_e800_b64_s42_gpu0/results.csv",
        "stable BN-freeze seed42; cross-machine evidence",
    ),
    RunSpec(
        "nomosaic_cap2_s123_bnfreeze",
        "no-mosaic cap2 s123 BN-freeze",
        "formal_nomosaic",
        "90",
        "n",
        123,
        "cap2_bnfreeze1e3",
        "freeze",
        "complete",
        "ladd/results/converged_mainline_ladd_20260613/source/no_mosaic_90/n_cap2/"
        "ladd_hbb_ogsod11n_formal_nomosaic_yolo11n_cap2_s123_bnfreeze1e3_90_gpu7_b_e800_b64_s123_gpu7/results.csv",
        "BN-freeze fixes seed123 collapse/late regression into positive stable run",
    ),
    RunSpec(
        "nomosaic_s_cap2_s0_a2mu",
        "s no-mosaic cap2 s0 no-BN-freeze partial",
        "formal_nomosaic",
        "90",
        "s",
        0,
        "cap2_a2mu1e3",
        "normal",
        "partial_positive",
        "ladd/results/converged_mainline_ladd_20260613/source/no_mosaic_90/s_cap2/"
        "ladd_hbb_ogsod11n_formal_nomosaic_yolo11s_cap2_s0_a2mu1e3_b_e800_b64_s0_gpu5/results.csv",
        "stopped at epoch 608; positive mid/late evidence, but not full closure",
    ),
    RunSpec(
        "nomosaic_s_cap2_s0_bnfreeze",
        "s no-mosaic cap2 s0 BN-freeze",
        "formal_nomosaic",
        "dual4090",
        "s",
        0,
        "cap2_bnfreeze1e3",
        "freeze",
        "late_regression",
        "ladd/results/converged_mainline_ladd_20260613/source/no_mosaic_4090dual/s_cap2/"
        "ladd_hbb_ogsod11n_formal_nomosaic_yolo11s_cap2_s0_bnfreeze1e3_public4090dual_final_v2_b_e800_b64_s0_gpu1/results.csv",
        "full 800; best positive but final below SAR final",
    ),
    RunSpec(
        "nomosaic_m_cap2_s0_a2mu",
        "m no-mosaic cap2 s0 partial abnormal",
        "formal_nomosaic",
        "90",
        "m",
        0,
        "cap2_a2mu1e3",
        "normal",
        "abnormal_partial",
        "ladd/results/converged_mainline_ladd_20260613/source/no_mosaic_90/m_cap2/"
        "ladd_hbb_ogsod11n_formal_nomosaic_yolo11m_cap2_s0_a2mu1e3_b_e800_b32_s0_gpu4/results.csv",
        "B entrance is already far below m SAR baseline; not valid mainline result",
    ),
]


CHAIN_SPECS = [
    ChainSpec(
        "chain_nomosaic_original_s0",
        "n original/no-cap2 s0",
        "formal_nomosaic",
        "n",
        0,
        "original",
        "complete",
        "ladd/results/converged_mainline_ladd_20260613/source/no_mosaic_90/n_original/"
        "ladd_hbb_ogsod11n_formal_nomosaic_yolo11n_original_s0_a2mu1e3_a1_e10_b64_s0_gpu4/results.csv",
        "ladd/results/converged_mainline_ladd_20260613/source/no_mosaic_90/n_original/"
        "ladd_hbb_ogsod11n_formal_nomosaic_yolo11n_original_s0_a2mu1e3_a2_e50_b64_s0_gpu4/results.csv",
        "ladd/results/converged_mainline_ladd_20260613/source/no_mosaic_90/n_original/"
        "ladd_hbb_ogsod11n_formal_nomosaic_yolo11n_original_s0_a2mu1e3_b_e800_b64_s0_gpu4/results.csv",
        "highest n no-mosaic best; diagnostic no-cap2 reference",
    ),
    ChainSpec(
        "chain_nomosaic_cap2_s0_a2mu",
        "n cap2 s0 no-BN-freeze",
        "formal_nomosaic",
        "n",
        0,
        "cap2_a2mu1e3",
        "complete",
        "ladd/results/converged_mainline_ladd_20260613/source/no_mosaic_90/n_cap2/"
        "ladd_hbb_ogsod11n_formal_nomosaic_yolo11n_cap2_s0_a2mu1e3_a1_e10_b64_s0_gpu6/results.csv",
        "ladd/results/converged_mainline_ladd_20260613/source/no_mosaic_90/n_cap2/"
        "ladd_hbb_ogsod11n_formal_nomosaic_yolo11n_cap2_s0_a2mu1e3_a2_e50_b64_s0_gpu6/results.csv",
        "ladd/results/converged_mainline_ladd_20260613/source/no_mosaic_90/n_cap2/"
        "ladd_hbb_ogsod11n_formal_nomosaic_yolo11n_cap2_s0_a2mu1e3_b_e800_b64_s0_gpu6/results.csv",
        "healthy no-BN-freeze seed0",
    ),
    ChainSpec(
        "chain_nomosaic_cap2_s42_a2mu",
        "n cap2 s42 no-BN-freeze",
        "formal_nomosaic",
        "n",
        42,
        "cap2_a2mu1e3",
        "complete",
        "ladd/results/converged_mainline_ladd_20260613/source/no_mosaic_90/n_cap2/"
        "ladd_hbb_ogsod11n_formal_nomosaic_yolo11n_cap2_s42_a2mu1e3_a1_e10_b64_s42_gpu4/results.csv",
        "ladd/results/converged_mainline_ladd_20260613/source/no_mosaic_90/n_cap2/"
        "ladd_hbb_ogsod11n_formal_nomosaic_yolo11n_cap2_s42_a2mu1e3_a2_e50_b64_s42_gpu4/results.csv",
        "ladd/results/converged_mainline_ladd_20260613/source/no_mosaic_90/n_cap2/"
        "ladd_hbb_ogsod11n_formal_nomosaic_yolo11n_cap2_s42_a2mu1e3_b_e800_b64_s42_gpu4/results.csv",
        "healthy no-BN-freeze seed42",
    ),
    ChainSpec(
        "chain_nomosaic_cap2_s123_old",
        "n cap2 s123 old-B crash",
        "formal_nomosaic",
        "n",
        123,
        "cap2_old_b",
        "nan_crash",
        "ladd/results/converged_mainline_ladd_20260613/source/no_mosaic_90/n_cap2/"
        "ladd_hbb_ogsod11n_formal_nomosaic_yolo11n_cap2_s123_a2mu1e3_a1_e10_b64_s123_gpu5/results.csv",
        "ladd/results/converged_mainline_ladd_20260613/source/no_mosaic_90/n_cap2/"
        "ladd_hbb_ogsod11n_formal_nomosaic_yolo11n_cap2_s123_a2mu1e3_a2_e50_b64_s123_gpu5/results.csv",
        "ladd/results/converged_mainline_ladd_20260613/source/no_mosaic_90/n_cap2/"
        "ladd_hbb_ogsod11n_formal_nomosaic_yolo11n_cap2_s123_a2mu1e3_b_e800_b64_s123_gpu5/results.csv",
        "A1/A2 are healthy; old B path crashes after entering B",
    ),
    ChainSpec(
        "chain_nomosaic_cap2_s123_bstable",
        "n cap2 s123 B-lr1e-3",
        "formal_nomosaic",
        "n",
        123,
        "cap2_bstable1e3",
        "late_regression",
        "ladd/results/converged_mainline_ladd_20260613/source/no_mosaic_90/n_cap2/"
        "ladd_hbb_ogsod11n_formal_nomosaic_yolo11n_cap2_s123_a2mu1e3_a1_e10_b64_s123_gpu5/results.csv",
        "ladd/results/converged_mainline_ladd_20260613/source/no_mosaic_90/n_cap2/"
        "ladd_hbb_ogsod11n_formal_nomosaic_yolo11n_cap2_s123_a2mu1e3_a2_e50_b64_s123_gpu5/results.csv",
        "ladd/results/converged_mainline_ladd_20260613/source/no_mosaic_90/n_cap2/"
        "ladd_hbb_ogsod11n_formal_nomosaic_yolo11n_cap2_s123_a2mu1e3_bstable1e3_b_e800_b64_s123_gpu2/results.csv",
        "lower B LR avoids NaN but cannot prevent late regression",
    ),
    ChainSpec(
        "chain_nomosaic_cap2_s123_bnfreeze",
        "n cap2 s123 BN-freeze",
        "formal_nomosaic",
        "n",
        123,
        "cap2_bnfreeze1e3",
        "complete",
        "ladd/results/converged_mainline_ladd_20260613/source/no_mosaic_90/n_cap2/"
        "ladd_hbb_ogsod11n_formal_nomosaic_yolo11n_cap2_s123_a2mu1e3_a1_e10_b64_s123_gpu5/results.csv",
        "ladd/results/converged_mainline_ladd_20260613/source/no_mosaic_90/n_cap2/"
        "ladd_hbb_ogsod11n_formal_nomosaic_yolo11n_cap2_s123_a2mu1e3_a2_e50_b64_s123_gpu5/results.csv",
        "ladd/results/converged_mainline_ladd_20260613/source/no_mosaic_90/n_cap2/"
        "ladd_hbb_ogsod11n_formal_nomosaic_yolo11n_cap2_s123_bnfreeze1e3_90_gpu7_b_e800_b64_s123_gpu7/results.csv",
        "BN-freeze stabilizes the same A1/A2 entrance",
    ),
    ChainSpec(
        "chain_nomosaic_cap2_s42_bnfreeze",
        "n cap2 s42 BN-freeze",
        "formal_nomosaic",
        "n",
        42,
        "cap2_bnfreeze1e3",
        "complete",
        "ladd/results/converged_mainline_ladd_20260613/source/no_mosaic_4090dual/n_cap2/"
        "ladd_hbb_ogsod11n_formal_nomosaic_yolo11n_cap2_s42_bnfreeze1e3_public4090dual_final_v2_a1_e10_b64_s42_gpu0/results.csv",
        "ladd/results/converged_mainline_ladd_20260613/source/no_mosaic_4090dual/n_cap2/"
        "ladd_hbb_ogsod11n_formal_nomosaic_yolo11n_cap2_s42_bnfreeze1e3_public4090dual_final_v2_a2_e50_b64_s42_gpu0/results.csv",
        "ladd/results/converged_mainline_ladd_20260613/source/no_mosaic_4090dual/n_cap2/"
        "ladd_hbb_ogsod11n_formal_nomosaic_yolo11n_cap2_s42_bnfreeze1e3_public4090dual_final_v2_b_e800_b64_s42_gpu0/results.csv",
        "cross-machine BN-freeze full chain",
    ),
    ChainSpec(
        "chain_mosaic_cap2_s0",
        "mosaic100 cap2 s0",
        "mosaic100",
        "n",
        0,
        "cap2",
        "complete",
        "ladd/results/converged_mainline_ladd_20260613/source/mosaic90/ladd_b_runs/"
        "ladd_hbb_ogsod11n_ladd800r2_cap2_s0_a1_e10_b64_s0_gpu4/results.csv",
        "ladd/results/converged_mainline_ladd_20260613/source/mosaic90/ladd_b_runs/"
        "ladd_hbb_ogsod11n_ladd800r2_cap2_s0_a2_e50_b64_s0_gpu4/results.csv",
        "ladd/results/converged_mainline_ladd_20260613/source/mosaic90/ladd_b_runs/"
        "ladd_hbb_ogsod11n_ladd800r2_cap2_s0_b_e800_b64_s0_gpu4/results.csv",
        "largest historical gain under mosaic100",
    ),
    ChainSpec(
        "chain_mosaic_cap2_s42",
        "mosaic100 cap2 s42",
        "mosaic100",
        "n",
        42,
        "cap2",
        "complete",
        "ladd/results/converged_mainline_ladd_20260613/source/mosaic90/ladd_b_runs/"
        "ladd_hbb_ogsod11n_ladd800r2_cap2_s42_a1_e10_b64_s42_gpu3/results.csv",
        "ladd/results/converged_mainline_ladd_20260613/source/mosaic90/ladd_b_runs/"
        "ladd_hbb_ogsod11n_ladd800r2_cap2_s42_a2_e50_b64_s42_gpu3/results.csv",
        "ladd/results/converged_mainline_ladd_20260613/source/mosaic90/ladd_b_runs/"
        "ladd_hbb_ogsod11n_ladd800r2_cap2_s42_b_e800_b64_s42_gpu3/results.csv",
        "mosaic100 stable chain",
    ),
    ChainSpec(
        "chain_mosaic_cap2_s123",
        "mosaic100 cap2 s123",
        "mosaic100",
        "n",
        123,
        "cap2",
        "complete",
        "ladd/results/converged_mainline_ladd_20260613/source/mosaic90/ladd_b_runs/"
        "ladd_hbb_ogsod11n_ladd800r2_cap2_s123_a1_e10_b64_s123_gpu5/results.csv",
        "ladd/results/converged_mainline_ladd_20260613/source/mosaic90/ladd_b_runs/"
        "ladd_hbb_ogsod11n_ladd800r2_cap2_s123_a2_e50_b64_s123_gpu5/results.csv",
        "ladd/results/converged_mainline_ladd_20260613/source/mosaic90/ladd_b_runs/"
        "ladd_hbb_ogsod11n_ladd800r2_cap2_s123_b_e800_b64_s123_gpu5/results.csv",
        "mosaic100 stable chain",
    ),
]


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def load_curve(path: str) -> pd.DataFrame:
    df = clean_columns(pd.read_csv(REPO / path))
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    df = df.dropna(subset=["epoch"]).drop_duplicates(subset=["epoch"], keep="last")
    df["epoch"] = df["epoch"].astype(int)
    df = df.sort_values("epoch")
    for col in df.columns:
        if col != "epoch":
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().any():
                df[col] = converted
    return df


def metric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([float("nan")] * len(df), index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def baseline_for(spec: RunSpec) -> dict[str, float]:
    return BASELINES[(spec.protocol, spec.model, spec.seed)]


def summarize(spec: RunSpec, df: pd.DataFrame) -> dict:
    y = metric(df, "metrics/mAP50-95(B)")
    best_idx = y.idxmax()
    last_idx = df.index[-1]
    base = baseline_for(spec)
    return {
        "key": spec.key,
        "label": spec.label,
        "protocol": spec.protocol,
        "server": spec.server,
        "model": spec.model,
        "seed": spec.seed,
        "method": spec.method,
        "bn_stats": spec.bn_stats,
        "status": spec.status,
        "epochs_recorded": int(df["epoch"].max()),
        "best_epoch": int(df.loc[best_idx, "epoch"]),
        "best_map": float(y.loc[best_idx]),
        "last_epoch": int(df.loc[last_idx, "epoch"]),
        "last_map": float(y.loc[last_idx]),
        "sar_baseline_best": base["best"],
        "sar_baseline_final": base["final"],
        "best_minus_sar_best": float(y.loc[best_idx] - base["best"]),
        "last_minus_sar_final": float(y.loc[last_idx] - base["final"]),
        "best_final_drop": float(y.loc[best_idx] - y.loc[last_idx]),
        "source_path": spec.path,
        "notes": spec.notes,
    }


def phase_path(chain: ChainSpec, phase: str) -> str:
    if phase == "A1":
        return chain.a1_path
    if phase == "A2":
        return chain.a2_path
    if phase == "B":
        return chain.b_path
    raise ValueError(phase)


def load_chain(chain: ChainSpec) -> pd.DataFrame:
    parts = []
    offsets = {"A1": 0, "A2": 10, "B": 60}
    for phase in ("A1", "A2", "B"):
        df = load_curve(phase_path(chain, phase)).copy()
        df["phase"] = phase
        df["chain_epoch"] = df["epoch"] + offsets[phase]
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def summarize_phase(chain: ChainSpec, phase: str) -> dict:
    df = load_curve(phase_path(chain, phase))
    y = metric(df, "metrics/mAP50-95(B)")
    best_idx = y.idxmax()
    last_idx = df.index[-1]
    base = baseline_for(
        RunSpec(
            key=chain.key,
            label=chain.label,
            protocol=chain.protocol,
            server="",
            model=chain.model,
            seed=chain.seed,
            method=chain.method,
            bn_stats="",
            status=chain.status,
            path=phase_path(chain, phase),
            notes=chain.notes,
        )
    )
    return {
        "chain_key": chain.key,
        "label": chain.label,
        "protocol": chain.protocol,
        "model": chain.model,
        "seed": chain.seed,
        "method": chain.method,
        "status": chain.status,
        "phase": phase,
        "epochs_recorded": int(df["epoch"].max()),
        "best_epoch_in_phase": int(df.loc[best_idx, "epoch"]),
        "best_map": float(y.loc[best_idx]),
        "last_epoch_in_phase": int(df.loc[last_idx, "epoch"]),
        "last_map": float(y.loc[last_idx]),
        "sar_baseline_best": base["best"],
        "sar_baseline_final": base["final"],
        "best_minus_sar_best": float(y.loc[best_idx] - base["best"]),
        "last_minus_sar_final": float(y.loc[last_idx] - base["final"]),
        "source_path": phase_path(chain, phase),
        "notes": chain.notes,
    }


def det_total(df: pd.DataFrame, prefix: str) -> pd.Series:
    return metric(df, f"{prefix}/box_loss") + metric(df, f"{prefix}/cls_loss") + metric(df, f"{prefix}/dfl_loss")


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "font.family": "DejaVu Sans",
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png")
    fig.savefig(FIG_DIR / f"{name}.pdf")
    plt.close(fig)


def add_stage_markers(ax: plt.Axes) -> None:
    ax.axvline(10, color="0.25", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axvline(60, color="0.25", linestyle="--", linewidth=0.8, alpha=0.7)
    ymax = ax.get_ylim()[1]
    ymin = ax.get_ylim()[0]
    y = ymax - 0.06 * (ymax - ymin)
    ax.text(5, y, "A1", ha="center", va="top", fontsize=8, color="0.25")
    ax.text(35, y, "A2", ha="center", va="top", fontsize=8, color="0.25")
    ax.text(460, y, "B", ha="center", va="top", fontsize=8, color="0.25")


def plot_chain_ap(
    ax: plt.Axes,
    chains: Iterable[ChainSpec],
    chain_data: dict[str, pd.DataFrame],
    linewidth: float = 1.4,
) -> None:
    for chain in chains:
        df = chain_data[chain.key]
        ax.plot(df["chain_epoch"], metric(df, "metrics/mAP50-95(B)"), linewidth=linewidth, label=chain.label)
    ax.set_xlabel("Chained epoch: A1(1-10) + A2(11-60) + B(61+)")
    ax.set_ylabel("AP50-95")
    add_stage_markers(ax)


def plot_ap(ax: plt.Axes, specs: Iterable[RunSpec], data: dict[str, pd.DataFrame], linewidth: float = 1.5) -> None:
    for spec in specs:
        df = data[spec.key]
        ax.plot(df["epoch"], metric(df, "metrics/mAP50-95(B)"), linewidth=linewidth, label=spec.label)
    ax.set_xlabel("B epoch")
    ax.set_ylabel("AP50-95")


def add_baseline(ax: plt.Axes, protocol: str, model: str, seed: int = 0, label_prefix: str = "SAR") -> None:
    base = BASELINES[(protocol, model, seed)]
    ax.axhline(base["best"], color="black", linestyle="--", linewidth=0.9, alpha=0.75, label=f"{label_prefix} best")
    ax.axhline(base["final"], color="0.35", linestyle=":", linewidth=0.9, alpha=0.85, label=f"{label_prefix} final")


def make_figures(data: dict[str, pd.DataFrame], summary: pd.DataFrame, chain_data: dict[str, pd.DataFrame]) -> None:
    configure_style()

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharex=True)
    n_nomosaic_keys = [
        "nomosaic_original_s0",
        "nomosaic_cap2_s0_a2mu",
        "nomosaic_cap2_s42_a2mu",
        "nomosaic_cap2_s123_old_crash",
        "nomosaic_cap2_s123_bstable",
        "nomosaic_cap2_s0_bnfreeze",
        "nomosaic_cap2_s42_bnfreeze",
        "nomosaic_cap2_s123_bnfreeze",
    ]
    n_nomosaic = [spec for spec in RUNS if spec.key in n_nomosaic_keys]
    plot_ap(axes[0], n_nomosaic, data, linewidth=1.35)
    add_baseline(axes[0], "formal_nomosaic", "n", 0)
    axes[0].set_title("Full scale: seed123 old-B collapse")
    axes[0].set_xlim(0, 820)
    axes[0].set_ylim(-0.02, 0.60)
    plot_ap(axes[1], n_nomosaic, data, linewidth=1.45)
    add_baseline(axes[1], "formal_nomosaic", "n", 0)
    axes[1].set_title("Zoom: healthy runs and late regression")
    axes[1].set_xlim(0, 820)
    axes[1].set_ylim(0.51, 0.585)
    axes[1].legend(ncol=1, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=6)
    save(fig, "fig1_nomosaic_n_no_bnfreeze_bnfreeze_curves")

    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    mosaic = [spec for spec in RUNS if spec.protocol == "mosaic100"]
    plot_ap(ax, mosaic, data, linewidth=1.5)
    add_baseline(ax, "mosaic100", "n", 0)
    ax.set_xlim(0, 820)
    ax.set_ylim(0.525, 0.573)
    ax.set_title("Old mosaic100/close@100 LADD curves")
    ax.legend(ncol=2, frameon=False, loc="lower right", fontsize=6)
    save(fig, "fig2_mosaic100_n_ladd_curves")

    complete = summary[summary["status"].isin(["complete", "partial_positive", "late_regression", "abnormal_partial"])].copy()
    complete = complete.sort_values(["protocol", "model", "seed", "method"])
    colors = complete["protocol"].map({"mosaic100": "#4C78A8", "formal_nomosaic": "#F58518"}).fillna("#777777")
    fig, ax = plt.subplots(figsize=(9.4, 5.6))
    labels = [
        f"{r.protocol.replace('formal_', '')} | {r.model} s{r.seed} | {r.method}"
        for r in complete.itertuples(index=False)
    ]
    y_pos = range(len(complete))
    ax.barh(y_pos, complete["best_minus_sar_best"], color=list(colors))
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("best AP - same-protocol SAR best")
    ax.set_title("Mainline LADD best gain overview")
    ax.invert_yaxis()
    save(fig, "fig3_all_mainline_best_gain")

    fig, ax = plt.subplots(figsize=(9.4, 5.6))
    colors = ["#4C78A8" if v >= 0 else "#E45756" for v in complete["last_minus_sar_final"]]
    ax.barh(y_pos, complete["last_minus_sar_final"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("final/latest AP - same-protocol SAR final")
    ax.set_title("Final/latest gain and late-regression risk")
    ax.invert_yaxis()
    save(fig, "fig4_all_mainline_final_gain")

    formal_keys = [
        "chain_nomosaic_original_s0",
        "chain_nomosaic_cap2_s0_a2mu",
        "chain_nomosaic_cap2_s42_a2mu",
        "chain_nomosaic_cap2_s123_old",
        "chain_nomosaic_cap2_s123_bstable",
        "chain_nomosaic_cap2_s123_bnfreeze",
        "chain_nomosaic_cap2_s42_bnfreeze",
    ]
    formal_chains = [chain for chain in CHAIN_SPECS if chain.key in formal_keys]
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 7.2), sharex=True)
    plot_chain_ap(axes[0], formal_chains, chain_data, linewidth=1.3)
    add_baseline(axes[0], "formal_nomosaic", "n", 0)
    axes[0].set_title("Formal no-mosaic YOLO11n A1->A2->B chain: full scale")
    axes[0].set_ylim(-0.02, 0.60)
    plot_chain_ap(axes[1], formal_chains, chain_data, linewidth=1.35)
    add_baseline(axes[1], "formal_nomosaic", "n", 0)
    axes[1].set_title("Formal no-mosaic YOLO11n A1->A2->B chain: zoom")
    axes[1].set_ylim(0.50, 0.585)
    axes[1].legend(ncol=1, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=6)
    save(fig, "fig5_nomosaic_n_a1a2b_stage_chain_ap")

    entrance_keys = [
        "chain_nomosaic_original_s0",
        "chain_nomosaic_cap2_s0_a2mu",
        "chain_nomosaic_cap2_s42_a2mu",
        "chain_nomosaic_cap2_s123_old",
        "chain_nomosaic_cap2_s42_bnfreeze",
    ]
    entrance_chains = [chain for chain in CHAIN_SPECS if chain.key in entrance_keys]
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 6.6), sharex=True)
    for chain in entrance_chains:
        df = chain_data[chain.key]
        df = df[df["phase"].isin(["A1", "A2"])]
        axes[0, 0].plot(df["chain_epoch"], metric(df, "metrics/mAP50-95(B)"), linewidth=1.4, label=chain.label)
        axes[0, 1].plot(df["chain_epoch"], det_total(df, "train"), linewidth=1.4, label=chain.label)
        axes[1, 0].plot(df["chain_epoch"], metric(df, "train/reach_match_loss"), linewidth=1.4, label=chain.label)
        axes[1, 1].plot(df["chain_epoch"], metric(df, "train/reach_rank_loss"), linewidth=1.4, label=chain.label)
    axes[0, 0].set_ylabel("AP50-95")
    axes[0, 1].set_ylabel("train det loss")
    axes[1, 0].set_ylabel("reach match")
    axes[1, 1].set_ylabel("reach rank")
    for ax in axes.ravel():
        ax.axvline(10, color="0.25", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_xlabel("Chained epoch: A1(1-10) + A2(11-60)")
    axes[0, 0].set_title("A1/A2 entrance AP")
    axes[0, 1].set_title("A1/A2 detector loss")
    axes[1, 0].set_title("A1/A2 reach match")
    axes[1, 1].set_title("A1/A2 reach rank")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.02), fontsize=6)
    save(fig, "fig6_nomosaic_n_a1a2_entrance_diagnostics")

    mosaic_keys = ["chain_mosaic_cap2_s0", "chain_mosaic_cap2_s42", "chain_mosaic_cap2_s123"]
    mosaic_chains = [chain for chain in CHAIN_SPECS if chain.key in mosaic_keys]
    fig, ax = plt.subplots(figsize=(9.4, 4.4))
    plot_chain_ap(ax, mosaic_chains, chain_data, linewidth=1.5)
    add_baseline(ax, "mosaic100", "n", 0)
    ax.set_title("Mosaic100/close@100 YOLO11n A1->A2->B chain")
    ax.set_xlim(0, 870)
    ax.set_ylim(0.50, 0.573)
    ax.legend(ncol=1, frameon=False, loc="lower right", fontsize=7)
    save(fig, "fig7_mosaic100_n_a1a2b_stage_chain_ap")


def format_table(df: pd.DataFrame) -> str:
    cols = [
        "protocol",
        "server",
        "model",
        "seed",
        "method",
        "bn_stats",
        "status",
        "epochs_recorded",
        "best_epoch",
        "best_map",
        "last_map",
        "sar_baseline_best",
        "best_minus_sar_best",
        "last_minus_sar_final",
        "best_final_drop",
        "notes",
    ]
    out = df[cols].copy()
    for col in ["best_map", "last_map", "sar_baseline_best", "best_minus_sar_best", "last_minus_sar_final", "best_final_drop"]:
        out[col] = out[col].map(lambda x: f"{x:.5f}")
    return out.to_markdown(index=False)


def write_report(summary: pd.DataFrame, phase_summary: pd.DataFrame) -> None:
    rel_fig = "https://cdn.jsdelivr.net/gh/yudongfang-thu/LADD_public@main/docs/experiments/figures/ladd_converged_mainline_ladd_20260613"
    local_fig = "./figures/ladd_converged_mainline_ladd_20260613"

    mosaic = summary[summary["protocol"] == "mosaic100"].sort_values(["method", "seed"])
    nomosaic_n = summary[(summary["protocol"] == "formal_nomosaic") & (summary["model"] == "n")].sort_values(["method", "seed"])
    nomosaic_sm = summary[(summary["protocol"] == "formal_nomosaic") & (summary["model"].isin(["s", "m"]))].sort_values(["model", "seed", "method"])
    text = f"""# LADD 收敛主线历史对比与曲线补充

日期：2026-06-13

本文件补充两类之前汇报中容易漏掉、但对解释 LADD 主线非常关键的证据：

1. 早期 formal no-mosaic 主线里，`B_FREEZE_BN_STATS=0` 的 LADD 确实有很强的健康 run：`YOLO11n original/no-cap2 seed0` best `0.57821@730`，是目前 n no-mosaic LADD 里最高的单点；`cap2 seed0/42 no-BN-freeze` 也分别达到 `0.57662/0.57420`。但同一设置在 seed123 old-B 上会检测 loss NaN 并 collapse 到 final `0.00000`。
2. 更早的 90 服务器 mosaic100/close@100 主线中，LADD 在六条 n seed/method run 上都没有 collapse，best 相对同协议 SAR baseline 大约提升 `+0.02072` 到 `+0.02750`，这是历史上 LADD 提升最大的主线证据。

因此汇报口径应该是：LADD 不是“从来学不到”；相反，早期 no-BN-freeze 和 mosaic100 都能冲到很高。但 formal no-mosaic 收敛协议下，稳定性从 seed123 开始暴露，BN-freeze 修复了 n 的三 seed 稳定性，同时也降低了一部分峰值；s/m 则仍有容量相关退化。

## 1. 数据与图件

汇总 CSV：

```text
ladd/results/converged_mainline_ladd_20260613/converged_mainline_ladd_summary_20260613.csv
docs/experiments/ladd_converged_mainline_ladd_summary_20260613.csv
ladd/results/converged_mainline_ladd_20260613/converged_mainline_ladd_phase_summary_20260613.csv
docs/experiments/ladd_converged_mainline_ladd_phase_summary_20260613.csv
```

绘图脚本：

```text
ladd/results/converged_mainline_ladd_20260613/gen_converged_mainline_ladd_overview.py
```

图件：

![formal no-mosaic n curves]({rel_fig}/fig1_nomosaic_n_no_bnfreeze_bnfreeze_curves.png)

![mosaic100 n curves]({rel_fig}/fig2_mosaic100_n_ladd_curves.png)

![all mainline best gain]({rel_fig}/fig3_all_mainline_best_gain.png)

![all mainline final gain]({rel_fig}/fig4_all_mainline_final_gain.png)

![formal no-mosaic A1/A2/B chain]({rel_fig}/fig5_nomosaic_n_a1a2b_stage_chain_ap.png)

![formal no-mosaic A1/A2 entrance diagnostics]({rel_fig}/fig6_nomosaic_n_a1a2_entrance_diagnostics.png)

![mosaic100 A1/A2/B chain]({rel_fig}/fig7_mosaic100_n_a1a2b_stage_chain_ap.png)

本地图片路径：

- [fig1_nomosaic_n_no_bnfreeze_bnfreeze_curves.png]({local_fig}/fig1_nomosaic_n_no_bnfreeze_bnfreeze_curves.png)
- [fig2_mosaic100_n_ladd_curves.png]({local_fig}/fig2_mosaic100_n_ladd_curves.png)
- [fig3_all_mainline_best_gain.png]({local_fig}/fig3_all_mainline_best_gain.png)
- [fig4_all_mainline_final_gain.png]({local_fig}/fig4_all_mainline_final_gain.png)
- [fig5_nomosaic_n_a1a2b_stage_chain_ap.png]({local_fig}/fig5_nomosaic_n_a1a2b_stage_chain_ap.png)
- [fig6_nomosaic_n_a1a2_entrance_diagnostics.png]({local_fig}/fig6_nomosaic_n_a1a2_entrance_diagnostics.png)
- [fig7_mosaic100_n_a1a2b_stage_chain_ap.png]({local_fig}/fig7_mosaic100_n_a1a2b_stage_chain_ap.png)

## 2. 早期 no-BN-freeze 主线：峰值强，但 seed123 暴露崩溃

`B_FREEZE_BN_STATS=0` 不是简单的坏设置。它在健康 seed 上峰值很强：

- `YOLO11n original/no-cap2 seed0`: best `0.57821@730`, last `0.57517`, 相对 n SAR seed0 best `+0.02167`。
- `YOLO11n cap2 seed0 no-BN-freeze`: best `0.57662@725`, last `0.57504`, 相对 n SAR seed0 best `+0.02008`。
- `YOLO11n cap2 seed42 no-BN-freeze`: best `0.57420@735`, last `0.57293`, 相对 n SAR seed42 best `+0.01626`。

问题出在稳定性边界：同样主线在 `seed123 old-B` 上记录到 epoch 483，best 只有 `0.52182@1`，final 为 `0.00000`，对应历史诊断中的 detection loss NaN / last.pt NaN-Inf 事件。随后 `bstable1e3` 能跑满 800，但 best/final 为 `0.56161/0.52875`，说明只降 B LR 能防 NaN，却不能防 late regression。

BN-freeze 的作用因此更像稳定性修复，而不是单纯涨点技巧：它让 n seed0/42/123 都回到正收益闭环，但 seed0 峰值低于 no-BN-freeze 的健康 run。

### A1/A2/B 连续链读法

补充的 A1->A2->B 连续曲线说明：`seed123 old-B` 的 A1/A2 入口并不坏。`seed123` 在 A1 last 为 `0.56128`，A2 best/last 为 `0.56574/0.56574`，已经高于同 seed SAR baseline best `0.56128`；崩溃发生在进入旧 B 后。`seed123 bstable1e3` 说明降低 B 学习率可以避免直接 NaN，但它的 B best/final 只有 `0.56161/0.52875`，仍然明显 late-regress。因此更准确的表述是：

- 降低 LR / 去 warmup：解决“旧 B 高 LR 路径直接 NaN”的一部分数值稳定问题。
- 只降低 LR：不能解决 seed123 的长期退化。
- BN-freeze：在 n seed123 上把同一个 A1/A2 入口稳定到 B best/final `0.57269/0.57219`，是目前 n 三 seed 闭环的关键稳定修复。
- 代价：BN-freeze seed0 峰值 `0.57276` 低于健康 no-BN-freeze seed0 `0.57662` 和 original/no-cap2 seed0 `0.57821`，所以它更像保守稳定化，而不是最高性能设置。

## 3. mosaic100/close@100：历史提升最大且没有 collapse

mosaic100/close@100 主线的同协议 SAR baseline best/final 为 `0.54091/0.53836`。六条 LADD B run 都稳定在 `0.56+`：

- legacy mean best `0.56631`，平均 gain `+0.02540`。
- cap2 mean best `0.56601`，平均 gain `+0.02510`。
- 单条最高为 `cap2 seed0` best `0.56841@798`，相对同协议 SAR best `+0.02750`。

这批结果很重要，因为它说明 LADD 在“带 mosaic 前 100 epoch、后 700 epoch 收敛”的旧协议下既能涨点，也没有当前 formal no-mosaic 中 seed123/s/m 暴露的崩溃或后期退化模式。后续汇报中应把它作为反证：问题不应被描述成 LADD 机制天然不可训练，而是训练协议、BN running stats、容量和阶段设置的交互。

## 4. 全部主线表

### mosaic100 / close@100

{format_table(mosaic)}

### formal no-mosaic: YOLO11n

{format_table(nomosaic_n)}

### formal no-mosaic: YOLO11s / YOLO11m

{format_table(nomosaic_sm)}

## 5. 汇报时建议放法

1. 先放 A1/A2/B 连续链：seed123 的 A1/A2 入口是健康的，分叉发生在 B；这比只展示 B 阶段更清楚。
2. 再放 no-mosaic n B-stage zoom 图：健康 no-BN-freeze run 很强，但 seed123 old-B 崩溃；这解释为什么“最开始性能最好”与“后来必须稳定修复”并不矛盾。
3. 再放 mosaic100 图：旧协议下 LADD 六条都正向，且提升最大；这说明方法潜力存在，当前问题是 formal no-mosaic 主线的新稳定性问题。
4. 最后放总表：BN-freeze 是 n 三 seed 闭环最稳的主线候选；s 的 BN-freeze 虽有 positive best，但 final 低于 SAR final；m 从 B 入口就异常，不能进入 full mainline。
"""
    DOC_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = {spec.key: load_curve(spec.path) for spec in RUNS}
    summary = pd.DataFrame([summarize(spec, data[spec.key]) for spec in RUNS])
    summary.to_csv(SUMMARY_PATH, index=False)
    summary.to_csv(DOC_SUMMARY_PATH, index=False)
    chain_data = {chain.key: load_chain(chain) for chain in CHAIN_SPECS}
    phase_summary = pd.DataFrame(
        [summarize_phase(chain, phase) for chain in CHAIN_SPECS for phase in ("A1", "A2", "B")]
    )
    phase_summary.to_csv(PHASE_SUMMARY_PATH, index=False)
    phase_summary.to_csv(DOC_PHASE_SUMMARY_PATH, index=False)
    make_figures(data, summary, chain_data)
    write_report(summary, phase_summary)
    print(f"wrote {SUMMARY_PATH.relative_to(REPO)}")
    print(f"wrote {DOC_SUMMARY_PATH.relative_to(REPO)}")
    print(f"wrote {PHASE_SUMMARY_PATH.relative_to(REPO)}")
    print(f"wrote {DOC_PHASE_SUMMARY_PATH.relative_to(REPO)}")
    print(f"wrote {DOC_PATH.relative_to(REPO)}")
    print(f"wrote {FIG_DIR.relative_to(REPO)}")


if __name__ == "__main__":
    main()
