#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import shutil
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd

from .evaluation import evaluate_prediction
from .plots import plot_40points
from .training import run_training_stage
from .utils import copy_code_snapshot, create_run_dir, ensure_dir, save_json, select_device, write_text
from .visualization import visualize_sr


PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PACKAGE_DIR)


def _write_log_line(log_path: str, message: str) -> None:
    print(message)
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def _logger_factory(log_path: str):
    def _log(message: str) -> None:
        _write_log_line(log_path, message)
    return _log


def _copy_final_checkpoint(src_checkpoint: str, result_dir: str) -> str:
    dst = os.path.join(result_dir, "X_hat_hrhsi_result.npy")
    shutil.copy2(src_checkpoint, dst)
    return dst


def _copy_checkpoint(src_checkpoint: str, dst_checkpoint: str) -> str:
    shutil.copy2(src_checkpoint, dst_checkpoint)
    return dst_checkpoint


def _spectral_segments(wavelengths_sorted_nm: np.ndarray, gap_threshold_nm: float = 30.0) -> list[tuple[int, int]]:
    breaks = np.where(np.diff(wavelengths_sorted_nm) > float(gap_threshold_nm))[0]
    start = 0
    segments: list[tuple[int, int]] = []
    for break_idx in breaks.tolist():
        end = int(break_idx) + 1
        segments.append((int(start), int(end)))
        start = end
    segments.append((int(start), int(wavelengths_sorted_nm.shape[0])))
    return segments


def _smooth_segment_last_axis(segment: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    radius = int(kernel.shape[0] // 2)
    padded = np.pad(segment, ((0, 0), (0, 0), (radius, radius)), mode="edge")
    smoothed = np.zeros_like(segment, dtype=np.float32)
    for kernel_idx in range(kernel.shape[0]):
        smoothed += float(kernel[kernel_idx]) * padded[..., kernel_idx:kernel_idx + segment.shape[-1]]
    return smoothed.astype(np.float32)


def _save_smoothed_checkpoint(
    *,
    src_checkpoint: str,
    dst_checkpoint: str,
    wavelengths_path: str,
    kernel: tuple[float, ...] = (0.25, 0.5, 0.25),
    gap_threshold_nm: float = 30.0,
) -> tuple[str, dict[str, Any]]:
    cube = np.load(src_checkpoint).astype(np.float32)
    wavelengths_raw = np.load(wavelengths_path).astype(np.float32).reshape(-1)
    if cube.ndim != 3:
        raise ValueError(f"Expected HWC cube for smoothing, got shape={cube.shape}")
    if cube.shape[-1] != wavelengths_raw.shape[0]:
        raise ValueError(
            f"Checkpoint channels do not match wavelengths for smoothing: {cube.shape[-1]} vs {wavelengths_raw.shape[0]}"
        )
    sort_idx = np.argsort(wavelengths_raw)
    inv_sort_idx = np.argsort(sort_idx)
    cube_sorted = cube[..., sort_idx]
    wavelengths_sorted = wavelengths_raw[sort_idx]
    kernel_np = np.asarray(kernel, dtype=np.float32)
    kernel_np = kernel_np / np.sum(kernel_np)
    smoothed_sorted = np.zeros_like(cube_sorted, dtype=np.float32)
    for start, end in _spectral_segments(wavelengths_sorted, gap_threshold_nm=gap_threshold_nm):
        smoothed_sorted[..., start:end] = _smooth_segment_last_axis(cube_sorted[..., start:end], kernel_np)
    smoothed = smoothed_sorted[..., inv_sort_idx]
    np.save(dst_checkpoint, smoothed.astype(np.float32))
    delta = smoothed.astype(np.float32) - cube.astype(np.float32)
    summary = {
        "src_checkpoint": os.path.abspath(src_checkpoint),
        "dst_checkpoint": os.path.abspath(dst_checkpoint),
        "kernel": [float(v) for v in kernel_np.tolist()],
        "gap_threshold_nm": float(gap_threshold_nm),
        "mean_abs_delta": float(np.mean(np.abs(delta))),
        "max_abs_delta": float(np.max(np.abs(delta))),
    }
    return dst_checkpoint, summary


def _combine_scores(output_path: str, stage1_csv: str, stage2_csv: str) -> None:
    frames = []
    for csv_path in [stage1_csv, stage2_csv]:
        if os.path.exists(csv_path):
            frames.append(pd.read_csv(csv_path))
    if len(frames) == 0:
        pd.DataFrame().to_csv(output_path, index=False)
        return
    pd.concat(frames, ignore_index=True, sort=False).to_csv(output_path, index=False)


def _teacher_run_label(run_tag: str) -> str:
    mapping = {
        "hs_only": "HS+RTM",
        "both": "HS+MS+RTM",
        "ms_only": "MS+RTM",
    }
    return mapping.get(run_tag, run_tag)


def _run_teacher_rtm(
    *,
    prisma_path: str,
    s2_path: str,
    wl_path: str,
    inv_path: str,
    fwd_path: str,
    scalers_path: str,
    srf_xlsx: str,
    result_dir: str,
    log_path: str,
    run_tag: str,
) -> tuple[str, str]:
    ensure_dir(result_dir)
    out_x_npy = os.path.join(result_dir, f"X_hat_hrhsi_{run_tag}.npy")
    out_wl_sort_idx = os.path.join(result_dir, "prisma_wl_sort_idx_used.npy")
    cmd = [
        sys.executable,
        "-m",
        "rtm_stagewise_sr.rtm.teacher",
        "--prisma_path", prisma_path,
        "--s2_path", s2_path,
        "--wl_path", wl_path,
        "--srf_xlsx", srf_xlsx,
        "--inv_torchscript_path", inv_path,
        "--fwd_torchscript_path", fwd_path,
        "--scalers_pkl", scalers_path,
        "--run_tag", run_tag,
        "--out_x_npy", out_x_npy,
        "--out_wl_sort_idx", out_wl_sort_idx,
        "--use_rtm_prior", "1",
        "--iters", "2000",
        "--lr", "0.01",
    ]
    _write_log_line(log_path, f"[teacher_{run_tag}] running: {' '.join(cmd)}")
    with open(log_path, "a", encoding="utf-8") as handle:
        subprocess.run(cmd, check=True, cwd=REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT)
    return out_x_npy, os.path.join(result_dir, "train_RTM_prior_only_summary.json")


def _teacher_config_label(
    *,
    base_run_tag: str,
    protected_run_tag: str | None,
    protect_band_min_nm: float,
    protect_band_max_nm: float,
) -> str:
    base_desc = _teacher_run_label(base_run_tag)
    if not protected_run_tag or protected_run_tag == base_run_tag:
        return f"{base_run_tag} ({base_desc})"
    protected_desc = _teacher_run_label(protected_run_tag)
    return (
        f"{base_run_tag} ({base_desc}) + "
        f"{protected_run_tag} ({protected_desc}) on {protect_band_min_nm:.0f}-{protect_band_max_nm:.0f} nm"
    )


def _build_protected_band_teacher(
    *,
    base_checkpoint: str,
    protected_checkpoint: str,
    wavelengths_path: str,
    protect_band_min_nm: float,
    protect_band_max_nm: float,
    result_dir: str,
    base_run_tag: str,
    protected_run_tag: str,
) -> tuple[str, str]:
    ensure_dir(result_dir)
    base_cube = np.load(base_checkpoint).astype(np.float32)
    protected_cube = np.load(protected_checkpoint).astype(np.float32)
    if base_cube.shape != protected_cube.shape:
        raise ValueError(
            "Teacher checkpoints must share the same shape for protected-band blending: "
            f"{base_cube.shape} vs {protected_cube.shape}"
        )
    if base_cube.ndim != 3:
        raise ValueError(f"Expected HWC teacher cube for blending, got shape={base_cube.shape}")

    wavelengths_raw = np.load(wavelengths_path).astype(np.float32).reshape(-1)
    wavelengths_sorted = wavelengths_raw[np.argsort(wavelengths_raw)]
    if int(base_cube.shape[-1]) != int(wavelengths_sorted.shape[0]):
        raise ValueError(
            "Teacher checkpoint channels do not match wavelength count for protected-band blending: "
            f"{base_cube.shape[-1]} vs {wavelengths_sorted.shape[0]}"
        )

    protected_mask = (
        (wavelengths_sorted >= float(protect_band_min_nm))
        & (wavelengths_sorted <= float(protect_band_max_nm))
    )
    hybrid_cube = base_cube.copy()
    hybrid_cube[..., protected_mask] = protected_cube[..., protected_mask]

    out_x_npy = os.path.join(
        result_dir,
        f"X_hat_hrhsi_{base_run_tag}_protect_{protected_run_tag}_{int(protect_band_min_nm)}_{int(protect_band_max_nm)}.npy",
    )
    np.save(out_x_npy, hybrid_cube.astype(np.float32))

    summary = {
        "base_checkpoint": os.path.abspath(base_checkpoint),
        "protected_checkpoint": os.path.abspath(protected_checkpoint),
        "hybrid_checkpoint": os.path.abspath(out_x_npy),
        "base_run_tag": base_run_tag,
        "protected_run_tag": protected_run_tag,
        "protect_band_min_nm": float(protect_band_min_nm),
        "protect_band_max_nm": float(protect_band_max_nm),
        "protected_band_count": int(np.sum(protected_mask)),
        "mean_abs_delta_allbands": float(np.mean(np.abs(hybrid_cube - base_cube))),
        "mean_abs_delta_protected": float(np.mean(np.abs(hybrid_cube[..., protected_mask] - base_cube[..., protected_mask]))),
        "mean_abs_delta_non_protected": float(
            np.mean(np.abs(hybrid_cube[..., ~protected_mask] - base_cube[..., ~protected_mask]))
        ),
    }
    summary_path = os.path.join(result_dir, "teacher_protected_band_blend_summary.json")
    save_json(summary_path, summary)
    return out_x_npy, summary_path


def _write_readme(
    *,
    readme_path: str,
    run_label: str,
    dataset_root: str,
    meta_json: str,
    field_csv: str,
    wavelengths_path: str,
    valid_point_count: int,
    stage1_info: dict[str, Any],
    baseline_info: dict[str, Any],
    final_info: dict[str, Any],
    smoothing_summary: dict[str, Any],
    metrics_raw_std: dict[str, Any],
    metrics_raw_700: dict[str, Any],
    metrics_std: dict[str, Any],
    metrics_700: dict[str, Any],
    teacher_checkpoint: str,
    baseline_checkpoint: str,
    raw_final_checkpoint: str,
    final_checkpoint: str,
    teacher_mode_desc: str,
    teacher_summary_path: str,
) -> None:
    lines = [
        f"# {run_label}",
        "",
        "## Dataset",
        f"- dataset_root: {dataset_root}",
        f"- meta_json: {meta_json}",
        f"- field_csv: {field_csv}",
        f"- wavelengths_path: {wavelengths_path}",
        f"- valid_field_points: {valid_point_count}",
        "",
        "## Final Scheme",
        "- stage2_update_mode: teacher_veg_band_alpha",
        "- lambda_ms_stage2: 0.20",
        "- lambda_alpha_smooth: 0.01",
        "- lambda_detail_lock: 1.0",
        "- protect_band_nm: 700-1300",
        "- blend_nm: 680-1320",
        "- post_spectral_smoothing: enabled",
        "",
        "## Prelude Inputs",
        f"- teacher_mode: {teacher_mode_desc}",
        f"- teacher_checkpoint: {teacher_checkpoint}",
        f"- teacher_summary: {teacher_summary_path}",
        f"- baseline_veg_lowfreq_checkpoint: {baseline_checkpoint}",
        "",
        "## Stage 1",
        f"- stage1_end_iter: {stage1_info.get('stage_end_iter')}",
        f"- stage1_stable_detected: {stage1_info.get('stage_stable_detected')}",
        "",
        "## Baseline Veg-Lowfreq Prelude",
        f"- best_iter: {baseline_info.get('best_iter')}",
        f"- selected_checkpoint: {baseline_info.get('selected_checkpoint')}",
        "",
        "## Final Teacher-Veg Alpha",
        f"- best_iter: {final_info.get('best_iter')}",
        f"- selected_checkpoint: {final_info.get('selected_checkpoint')}",
        f"- raw_final_checkpoint: {raw_final_checkpoint}",
        f"- smoothed_final_checkpoint: {final_checkpoint}",
        "",
        "## Post Smoothing",
        f"- kernel: {smoothing_summary.get('kernel')}",
        f"- gap_threshold_nm: {smoothing_summary.get('gap_threshold_nm')}",
        f"- mean_abs_delta: {smoothing_summary.get('mean_abs_delta')}",
        f"- max_abs_delta: {smoothing_summary.get('max_abs_delta')}",
        "",
        "## Raw Final Metrics",
        f"- field_mae_allbands: {metrics_raw_std.get('field_mae_allbands')}",
        f"- field_mae_nir_default: {metrics_raw_std.get('field_mae_nir')}",
        f"- field_mae_nir_700_1300: {metrics_raw_700.get('field_mae_nir')}",
        f"- field_mae_non_nir_700_1300: {metrics_raw_700.get('field_mae_non_nir')}",
        f"- n_points_used: {metrics_raw_std.get('n_points')}",
        "",
        "## Smoothed Final Metrics",
        f"- field_mae_allbands: {metrics_std.get('field_mae_allbands')}",
        f"- field_mae_nir_default: {metrics_std.get('field_mae_nir')}",
        f"- field_mae_nir_700_1300: {metrics_700.get('field_mae_nir')}",
        f"- field_mae_non_nir_700_1300: {metrics_700.get('field_mae_non_nir')}",
        f"- n_points_used: {metrics_std.get('n_points')}",
    ]
    write_text(readme_path, "\n".join(lines) + "\n")


def run_single_dataset(
    *,
    dataset_name: str,
    dataset_root: str,
    prisma_path: str,
    s2_path: str,
    meta_json: str,
    field_csv: str,
    wavelengths_path: str,
    rtm_inv_path: str,
    rtm_fwd_path: str,
    rtm_scalers_path: str,
    srf_xlsx: str,
    output_base: str,
    run_suffix: str = "",
    teacher_run_tag: str = "hs_only",
    teacher_protected_run_tag: str | None = None,
    stage1_checkpoint_override: str | None = None,
    device: str | None = None,
) -> str:
    run_label = f"{dataset_name}_centered_plot15aw_teacher_veg_alpha_ms020_as010"
    if run_suffix:
        run_label = f"{run_label}_{run_suffix}"
    run_root = create_run_dir(output_base, run_label)
    log_path = os.path.join(run_root, "train.log")
    logger = _logger_factory(log_path)
    training_device = select_device(device)

    with open(meta_json, "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    valid_point_count = int(len(meta.get("points", [])))
    protect_band_min_nm = 700.0
    protect_band_max_nm = 1300.0
    blend_in_start_nm = 680.0
    blend_out_end_nm = 1320.0

    save_json(
        os.path.join(run_root, "run_inputs.json"),
        {
            "dataset_name": dataset_name,
            "dataset_root": os.path.abspath(dataset_root),
            "prisma_path": os.path.abspath(prisma_path),
            "s2_path": os.path.abspath(s2_path),
            "meta_json": os.path.abspath(meta_json),
            "field_csv": os.path.abspath(field_csv),
            "wavelengths_path": os.path.abspath(wavelengths_path),
            "rtm_inv_path": os.path.abspath(rtm_inv_path),
            "rtm_fwd_path": os.path.abspath(rtm_fwd_path),
            "rtm_scalers_path": os.path.abspath(rtm_scalers_path),
            "srf_xlsx": os.path.abspath(srf_xlsx),
            "teacher_run_tag": teacher_run_tag,
            "teacher_protected_run_tag": teacher_protected_run_tag,
            "stage1_checkpoint_override": (
                os.path.abspath(stage1_checkpoint_override) if stage1_checkpoint_override else None
            ),
            "device": training_device,
            "protect_band_min_nm": protect_band_min_nm,
            "protect_band_max_nm": protect_band_max_nm,
            "valid_point_count": valid_point_count,
        },
    )

    stage1_dir = os.path.join(run_root, "stage1_train")
    if stage1_checkpoint_override:
        logger(f"[step] Reuse external Stage 1 checkpoint: {stage1_checkpoint_override}")
        ensure_dir(stage1_dir)
        stage1_checkpoint = os.path.abspath(stage1_checkpoint_override)
        if not os.path.exists(stage1_checkpoint):
            raise FileNotFoundError(f"External Stage 1 checkpoint not found: {stage1_checkpoint}")
        stage1_info = {
            "selected_checkpoint": stage1_checkpoint,
            "stage_end_iter": 0,
            "stage_stable_detected": None,
            "stage_stability_rule": "external_stage1_checkpoint",
            "checkpoint_scores_path": os.path.join(stage1_dir, "checkpoint_scores.csv"),
        }
        save_json(os.path.join(stage1_dir, "external_stage1_checkpoint.json"), stage1_info)
    else:
        logger("[step] Stage 1 no-RTM")
        stage1_info = run_training_stage(
            output_dir=stage1_dir,
            stage_name="stage1",
            prisma_path=prisma_path,
            s2_path=s2_path,
            wavelengths_path=wavelengths_path,
            init_checkpoint_path=None,
            iters=1200,
            lr=1e-2,
            lambda_hs=1.0,
            lambda_ms=1.0,
            lambda_spec=1e-2,
            lambda_tv=1e-3,
            use_rtm=False,
            keep_ms_loss=True,
            anchor_strategy="constant",
            anchor_mu_start=0.0,
            anchor_mu_end=0.0,
            selection_interval=50,
            min_select_iter=500,
            global_iter_offset=0,
            enable_stability_stop=True,
            stage1_min_iters=500,
            stage1_stability_window=50,
            stage1_stability_patience=3,
            stage1_stability_rel_tol=1e-3,
            stage2_score_patience=0,
            stage2_score_min_delta=0.0,
            device=training_device,
            logger=logger,
        )
        stage1_checkpoint = stage1_info["selected_checkpoint"]

    visualize_sr(
        pred_path=stage1_checkpoint,
        output_dir=os.path.join(run_root, "stage1_visual_compare"),
        wavelengths_path=wavelengths_path,
        ms_path=s2_path,
        prisma_path=prisma_path,
        legacy_no_rtm_path="",
        meta_json_path=meta_json,
        experiment_name=run_label,
        stage_name="stage1",
        iter_value=stage1_info["stage_end_iter"],
        use_rtm=False,
        keep_ms_spectral=True,
        roi_mode="field_points",
    )

    logger("[step] Prelude baseline veg_lowfreq_ms020_spec000")
    baseline_stage2_dir = os.path.join(run_root, "prelude_veg_lowfreq_ms020_spec000", "stage2_train")
    ensure_dir(os.path.dirname(baseline_stage2_dir))
    baseline_stage2 = run_training_stage(
        output_dir=baseline_stage2_dir,
        stage_name="stage2",
        prisma_path=prisma_path,
        s2_path=s2_path,
        wavelengths_path=wavelengths_path,
        init_checkpoint_path=stage1_checkpoint,
        iters=200,
        lr=3e-3,
        lambda_hs=1.0,
        lambda_ms=0.2,
        lambda_spec=5e-2,
        lambda_tv=1e-3,
        use_rtm=True,
        keep_ms_loss=True,
        anchor_strategy="constant",
        anchor_mu_start=0.5,
        anchor_mu_end=0.5,
        selection_interval=25,
        min_select_iter=50,
        global_iter_offset=int(stage1_info["stage_end_iter"]) + 1,
        enable_stability_stop=False,
        stage1_min_iters=int(stage1_info["stage_end_iter"]),
        stage1_stability_window=50,
        stage1_stability_patience=3,
        stage1_stability_rel_tol=1e-3,
        stage2_score_patience=4,
        stage2_score_min_delta=1e-5,
        stage2_update_mode="veg_lowfreq_residual",
        stage2_lowpass_kernel_size=9,
        stage2_lowpass_sigma=1.5,
        lambda_delta=0.003,
        lambda_delta_spec=0.0,
        lambda_detail_lock=1.0,
        stage2_score_ms_weight=0.25,
        stage2_score_osc_weight=0.3,
        stage2_spatial_gate_detail_max=8e-5,
        stage2_spatial_gate_hf_max=5e-5,
        stage2_veg_mask_ndvi_thr=0.2,
        rtm_inv_path=rtm_inv_path,
        rtm_fwd_path=rtm_fwd_path,
        rtm_scalers_path=rtm_scalers_path,
        device=training_device,
        logger=logger,
    )
    baseline_checkpoint = baseline_stage2["selected_checkpoint"]
    baseline_result_dir = os.path.join(run_root, "prelude_veg_lowfreq_ms020_spec000")
    save_json(
        os.path.join(baseline_result_dir, "best_checkpoint_info.json"),
        json.load(open(baseline_stage2["best_checkpoint_info_path"], "r", encoding="utf-8")),
    )
    shutil.copy2(baseline_checkpoint, os.path.join(baseline_result_dir, "X_hat_hrhsi_result.npy"))

    teacher_desc = _teacher_run_label(teacher_run_tag)
    logger(f"[step] Prelude teacher {teacher_run_tag} ({teacher_desc})")
    teacher_result_dir = os.path.join(run_root, f"prelude_teacher_{teacher_run_tag}_rtm")
    teacher_checkpoint_base, teacher_summary_path_base = _run_teacher_rtm(
        prisma_path=prisma_path,
        s2_path=s2_path,
        wl_path=wavelengths_path,
        inv_path=rtm_inv_path,
        fwd_path=rtm_fwd_path,
        scalers_path=rtm_scalers_path,
        srf_xlsx=srf_xlsx,
        result_dir=teacher_result_dir,
        log_path=log_path,
        run_tag=teacher_run_tag,
    )
    teacher_checkpoint = teacher_checkpoint_base
    teacher_summary_path = teacher_summary_path_base
    teacher_mode_desc = _teacher_config_label(
        base_run_tag=teacher_run_tag,
        protected_run_tag=None,
        protect_band_min_nm=protect_band_min_nm,
        protect_band_max_nm=protect_band_max_nm,
    )
    protected_teacher_checkpoint = None
    protected_teacher_summary_path = None

    if teacher_protected_run_tag and teacher_protected_run_tag != teacher_run_tag:
        protected_teacher_desc = _teacher_run_label(teacher_protected_run_tag)
        logger(
            f"[step] Prelude protected-band teacher {teacher_protected_run_tag} "
            f"({protected_teacher_desc})"
        )
        protected_teacher_result_dir = os.path.join(run_root, f"prelude_teacher_{teacher_protected_run_tag}_rtm")
        protected_teacher_checkpoint, protected_teacher_summary_path = _run_teacher_rtm(
            prisma_path=prisma_path,
            s2_path=s2_path,
            wl_path=wavelengths_path,
            inv_path=rtm_inv_path,
            fwd_path=rtm_fwd_path,
            scalers_path=rtm_scalers_path,
            srf_xlsx=srf_xlsx,
            result_dir=protected_teacher_result_dir,
            log_path=log_path,
            run_tag=teacher_protected_run_tag,
        )
        logger(
            f"[step] Blend teacher protected bands {protect_band_min_nm:.0f}-{protect_band_max_nm:.0f} nm "
            f"with {teacher_protected_run_tag}"
        )
        teacher_hybrid_dir = os.path.join(
            run_root,
            f"prelude_teacher_hybrid_{teacher_run_tag}_protect_{teacher_protected_run_tag}",
        )
        teacher_checkpoint, teacher_summary_path = _build_protected_band_teacher(
            base_checkpoint=teacher_checkpoint_base,
            protected_checkpoint=protected_teacher_checkpoint,
            wavelengths_path=wavelengths_path,
            protect_band_min_nm=protect_band_min_nm,
            protect_band_max_nm=protect_band_max_nm,
            result_dir=teacher_hybrid_dir,
            base_run_tag=teacher_run_tag,
            protected_run_tag=teacher_protected_run_tag,
        )
        teacher_mode_desc = _teacher_config_label(
            base_run_tag=teacher_run_tag,
            protected_run_tag=teacher_protected_run_tag,
            protect_band_min_nm=protect_band_min_nm,
            protect_band_max_nm=protect_band_max_nm,
        )

    logger("[step] Final scheme-3 winner teacher_veg_alpha_ms020_as010")
    stage2_dir = os.path.join(run_root, "stage2_train")
    final_stage2 = run_training_stage(
        output_dir=stage2_dir,
        stage_name="stage2",
        prisma_path=prisma_path,
        s2_path=s2_path,
        wavelengths_path=wavelengths_path,
        init_checkpoint_path=stage1_checkpoint,
        iters=200,
        lr=3e-3,
        lambda_hs=1.0,
        lambda_ms=0.2,
        lambda_spec=5e-2,
        lambda_tv=1e-3,
        use_rtm=True,
        keep_ms_loss=True,
        anchor_strategy="constant",
        anchor_mu_start=0.5,
        anchor_mu_end=0.5,
        selection_interval=25,
        min_select_iter=50,
        global_iter_offset=int(stage1_info["stage_end_iter"]) + 1,
        enable_stability_stop=False,
        stage1_min_iters=int(stage1_info["stage_end_iter"]),
        stage1_stability_window=50,
        stage1_stability_patience=3,
        stage1_stability_rel_tol=1e-3,
        stage2_score_patience=4,
        stage2_score_min_delta=1e-5,
        stage2_update_mode="teacher_veg_band_alpha",
        stage2_lowpass_kernel_size=9,
        stage2_lowpass_sigma=1.5,
        lambda_delta=0.0,
        lambda_delta_spec=0.0,
        lambda_detail_lock=1.0,
        stage2_score_ms_weight=0.25,
        stage2_score_osc_weight=0.3,
        stage2_spatial_gate_detail_max=2e-3,
        stage2_spatial_gate_hf_max=6e-4,
        stage2_veg_mask_ndvi_thr=0.2,
        stage2_teacher_checkpoint_path=teacher_checkpoint,
        stage2_baseline_checkpoint_path=baseline_checkpoint,
        stage2_protect_band_min_nm=protect_band_min_nm,
        stage2_protect_band_max_nm=protect_band_max_nm,
        stage2_blend_in_start_nm=blend_in_start_nm,
        stage2_blend_out_end_nm=blend_out_end_nm,
        lambda_alpha_smooth=1e-2,
        rtm_inv_path=rtm_inv_path,
        rtm_fwd_path=rtm_fwd_path,
        rtm_scalers_path=rtm_scalers_path,
        device=training_device,
        logger=logger,
    )
    raw_final_checkpoint = _copy_checkpoint(
        final_stage2["selected_checkpoint"],
        os.path.join(run_root, "X_hat_hrhsi_result_raw.npy"),
    )
    final_checkpoint, smoothing_summary = _save_smoothed_checkpoint(
        src_checkpoint=raw_final_checkpoint,
        dst_checkpoint=os.path.join(run_root, "X_hat_hrhsi_result.npy"),
        wavelengths_path=wavelengths_path,
    )
    save_json(os.path.join(run_root, "post_smoothing_summary.json"), smoothing_summary)

    _combine_scores(
        os.path.join(run_root, "checkpoint_scores.csv"),
        stage1_csv=stage1_info["checkpoint_scores_path"],
        stage2_csv=final_stage2["checkpoint_scores_path"],
    )
    final_best_info = json.load(open(final_stage2["best_checkpoint_info_path"], "r", encoding="utf-8"))
    final_best_info.update(
        {
            "stage1_end_iter": stage1_info.get("stage_end_iter"),
            "stage1_stable_detected": stage1_info.get("stage_stable_detected"),
            "stage1_stability_rule": stage1_info.get("stage_stability_rule"),
            "selected_checkpoint": os.path.abspath(final_checkpoint),
            "checkpoint_scores_csv": os.path.abspath(os.path.join(run_root, "checkpoint_scores.csv")),
        }
    )
    save_json(os.path.join(run_root, "best_checkpoint_info.json"), final_best_info)

    consistency_summary = json.load(open(final_stage2["train_no_gt_consistency_path"], "r", encoding="utf-8"))
    consistency_summary.update(
        {
            "stage1_end_iter": stage1_info.get("stage_end_iter"),
            "stage1_stable_detected": stage1_info.get("stage_stable_detected"),
            "stage1_stability_rule": stage1_info.get("stage_stability_rule"),
            "best_iter": final_best_info.get("best_iter"),
            "last_iter": final_best_info.get("last_iter"),
            "total_best_global_iter": final_best_info.get("total_best_global_iter"),
        }
    )
    save_json(os.path.join(run_root, "train_no_gt_consistency.json"), consistency_summary)

    visualize_sr(
        pred_path=raw_final_checkpoint,
        output_dir=os.path.join(run_root, "stage2_visual_compare_raw"),
        wavelengths_path=wavelengths_path,
        ms_path=s2_path,
        prisma_path=prisma_path,
        legacy_no_rtm_path="",
        comparison_path=stage1_checkpoint,
        comparison_label="Stage1 no-RTM",
        meta_json_path=meta_json,
        experiment_name=run_label,
        stage_name="stage2_blind_selected_raw",
        iter_value=final_best_info.get("best_iter"),
        use_rtm=True,
        keep_ms_spectral=True,
        roi_mode="field_points",
    )
    visualize_sr(
        pred_path=final_checkpoint,
        output_dir=os.path.join(run_root, "stage2_visual_compare"),
        wavelengths_path=wavelengths_path,
        ms_path=s2_path,
        prisma_path=prisma_path,
        legacy_no_rtm_path="",
        comparison_path=stage1_checkpoint,
        comparison_label="Stage1 no-RTM",
        meta_json_path=meta_json,
        experiment_name=run_label,
        stage_name="stage2_blind_selected_smoothed",
        iter_value=final_best_info.get("best_iter"),
        use_rtm=True,
        keep_ms_spectral=True,
        roi_mode="field_points",
    )

    metrics_raw_std = evaluate_prediction(
        checkpoint_path=raw_final_checkpoint,
        consistency_summary=consistency_summary,
        save_path=os.path.join(run_root, "final_field_metrics_raw.json"),
        wavelengths_path=wavelengths_path,
        meta_json_path=meta_json,
        field_csv_path=field_csv,
        legacy_no_rtm_path=stage1_checkpoint,
    )
    metrics_raw_700 = evaluate_prediction(
        checkpoint_path=raw_final_checkpoint,
        consistency_summary=consistency_summary,
        save_path=os.path.join(run_root, "final_field_metrics_700_1300_raw.json"),
        wavelengths_path=wavelengths_path,
        meta_json_path=meta_json,
        field_csv_path=field_csv,
        legacy_no_rtm_path=stage1_checkpoint,
        nir_min_nm=700.0,
        nir_max_nm=1300.0,
    )
    metrics_std = evaluate_prediction(
        checkpoint_path=final_checkpoint,
        consistency_summary=consistency_summary,
        save_path=os.path.join(run_root, "final_field_metrics.json"),
        wavelengths_path=wavelengths_path,
        meta_json_path=meta_json,
        field_csv_path=field_csv,
        legacy_no_rtm_path=stage1_checkpoint,
    )
    metrics_700 = evaluate_prediction(
        checkpoint_path=final_checkpoint,
        consistency_summary=consistency_summary,
        save_path=os.path.join(run_root, "final_field_metrics_700_1300.json"),
        wavelengths_path=wavelengths_path,
        meta_json_path=meta_json,
        field_csv_path=field_csv,
        legacy_no_rtm_path=stage1_checkpoint,
        nir_min_nm=700.0,
        nir_max_nm=1300.0,
    )

    spectra_raw_path = plot_40points(
        checkpoint_path=raw_final_checkpoint,
        run_dir=run_root,
        title=f"{run_label} | raw_final | n_points={metrics_raw_std.get('n_points')}",
        output_basename="spectra_40points_raw.png",
        wavelengths_path=wavelengths_path,
        meta_json_path=meta_json,
        field_csv_path=field_csv,
        legacy_no_rtm_path=stage1_checkpoint,
        legacy_label="Stage1 no-RTM",
    )
    spectra_path = plot_40points(
        checkpoint_path=final_checkpoint,
        run_dir=run_root,
        title=f"{run_label} | smoothed_final | n_points={metrics_std.get('n_points')}",
        wavelengths_path=wavelengths_path,
        meta_json_path=meta_json,
        field_csv_path=field_csv,
        legacy_no_rtm_path=stage1_checkpoint,
        legacy_label="Stage1 no-RTM",
    )

    final_selection = {
        "scheme": run_label,
        "dataset_name": dataset_name,
        "dataset_root": os.path.abspath(dataset_root),
        "meta_json_path": os.path.abspath(meta_json),
        "teacher_checkpoint": teacher_checkpoint,
        "teacher_summary_path": teacher_summary_path,
        "teacher_run_tag": teacher_run_tag,
        "teacher_mode_desc": teacher_mode_desc,
        "teacher_base_checkpoint": teacher_checkpoint_base,
        "teacher_base_summary_path": teacher_summary_path_base,
        "teacher_protected_run_tag": teacher_protected_run_tag,
        "teacher_protected_checkpoint": protected_teacher_checkpoint,
        "teacher_protected_summary_path": protected_teacher_summary_path,
        "baseline_checkpoint": baseline_checkpoint,
        "stage1_checkpoint": stage1_checkpoint,
        "raw_final_checkpoint": os.path.abspath(raw_final_checkpoint),
        "final_checkpoint": os.path.abspath(final_checkpoint),
        "best_iter": final_best_info.get("best_iter"),
        "n_points_used": metrics_std.get("n_points"),
        "spectra_40points_raw_path": spectra_raw_path,
        "spectra_40points_path": spectra_path,
        "post_smoothing_summary_path": os.path.join(run_root, "post_smoothing_summary.json"),
        "final_field_metrics_raw_path": os.path.join(run_root, "final_field_metrics_raw.json"),
        "final_field_metrics_700_1300_raw_path": os.path.join(run_root, "final_field_metrics_700_1300_raw.json"),
        "final_field_metrics_path": os.path.join(run_root, "final_field_metrics.json"),
        "final_field_metrics_700_1300_path": os.path.join(run_root, "final_field_metrics_700_1300.json"),
    }
    save_json(os.path.join(run_root, "final_selection.json"), final_selection)

    copy_code_snapshot(
        run_root,
        [
            os.path.join(PACKAGE_DIR, "pipeline.py"),
            os.path.join(PACKAGE_DIR, "training.py"),
            os.path.join(PACKAGE_DIR, "evaluation.py"),
            os.path.join(PACKAGE_DIR, "plots.py"),
            os.path.join(PACKAGE_DIR, "visualization.py"),
            os.path.join(PACKAGE_DIR, "data.py"),
            os.path.join(PACKAGE_DIR, "operators.py"),
            os.path.join(PACKAGE_DIR, "losses.py"),
            os.path.join(PACKAGE_DIR, "utils.py"),
            os.path.join(PACKAGE_DIR, "anchor.py"),
            os.path.join(PACKAGE_DIR, "preprocessing", "prepare.py"),
            os.path.join(PACKAGE_DIR, "rtm", "teacher.py"),
        ],
    )

    _write_readme(
        readme_path=os.path.join(run_root, "README.md"),
        run_label=run_label,
        dataset_root=dataset_root,
        meta_json=meta_json,
        field_csv=field_csv,
        wavelengths_path=wavelengths_path,
        valid_point_count=valid_point_count,
        stage1_info=stage1_info,
        baseline_info=baseline_stage2,
        final_info=final_stage2,
        smoothing_summary=smoothing_summary,
        metrics_raw_std=metrics_raw_std,
        metrics_raw_700=metrics_raw_700,
        metrics_std=metrics_std,
        metrics_700=metrics_700,
        teacher_checkpoint=teacher_checkpoint,
        baseline_checkpoint=baseline_checkpoint,
        raw_final_checkpoint=raw_final_checkpoint,
        final_checkpoint=final_checkpoint,
        teacher_mode_desc=teacher_mode_desc,
        teacher_summary_path=teacher_summary_path,
    )
    logger(f"[done] result_dir={run_root}")
    return run_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the centered_plot15aw teacher_veg_alpha winner on one dataset.")
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--prisma_path", required=True)
    parser.add_argument("--s2_path", required=True)
    parser.add_argument("--meta_json", required=True)
    parser.add_argument("--field_csv", required=True)
    parser.add_argument("--wavelengths_path", required=True)
    parser.add_argument("--rtm_inv_path", required=True)
    parser.add_argument("--rtm_fwd_path", required=True)
    parser.add_argument("--rtm_scalers_path", required=True)
    parser.add_argument("--srf_xlsx", required=True, help="Sentinel-2 spectral response workbook")
    parser.add_argument("--run_suffix", default="")
    parser.add_argument("--output_base", required=True)
    parser.add_argument("--teacher_run_tag", default="hs_only", choices=["hs_only", "both", "ms_only"])
    parser.add_argument("--teacher_protected_run_tag", default=None, choices=["hs_only", "both", "ms_only"])
    parser.add_argument("--stage1_checkpoint_override", default=None)
    parser.add_argument("--device", default=None, help="Training device, e.g. cuda, cuda:0, or cpu")
    args = parser.parse_args()
    result_dir = run_single_dataset(
        dataset_name=args.dataset_name,
        dataset_root=args.dataset_root,
        prisma_path=args.prisma_path,
        s2_path=args.s2_path,
        meta_json=args.meta_json,
        field_csv=args.field_csv,
        wavelengths_path=args.wavelengths_path,
        rtm_inv_path=args.rtm_inv_path,
        rtm_fwd_path=args.rtm_fwd_path,
        rtm_scalers_path=args.rtm_scalers_path,
        srf_xlsx=args.srf_xlsx,
        output_base=args.output_base,
        run_suffix=args.run_suffix,
        teacher_run_tag=args.teacher_run_tag,
        teacher_protected_run_tag=args.teacher_protected_run_tag,
        stage1_checkpoint_override=args.stage1_checkpoint_override,
        device=args.device,
    )
    print(result_dir)


if __name__ == "__main__":
    main()
