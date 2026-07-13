#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn

from .data import DEFAULT_PRISMA_PATH, DEFAULT_S2_PATH, DEFAULT_WL_PATH, load_core_data
from .losses import (
    LossWeights,
    build_total_loss,
    psnr_from_mse,
    sam_mean_rad,
    second_order_energy,
    spectral_smoothness_loss,
    spatial_gradient_consistency_loss,
)
from .operators import (
    FixedGaussianBlur,
    FixedGaussianDownsample,
    bicubic_initialize_from_hs,
    build_gaussian_s2_response,
    mix_to_multispectral,
)
from .anchor import (
    DEFAULT_FWD_PATH,
    DEFAULT_INV_PATH,
    DEFAULT_SCALERS_PATH,
    RTMAnchorProjector,
    amplitude_anchor_loss,
)
from .utils import append_history_csv, save_json, save_sort_index, select_device


def _compute_anchor_weight(strategy: str, iter_idx: int, total_iters: int, mu_start: float, mu_end: float) -> float:
    if strategy == "constant":
        return float(mu_start)
    progress = 0.0 if total_iters <= 0 else min(max(float(iter_idx) / float(total_iters), 0.0), 1.0)
    if strategy == "linear_decay":
        return float(mu_start + progress * (mu_end - mu_start))
    if strategy == "cosine_decay":
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        return float(mu_end + (mu_start - mu_end) * cosine)
    raise ValueError(f"Unsupported anchor strategy: {strategy}")


def _load_init_hsi(stage_init_path: str | None, hs_target_bchw: torch.Tensor) -> torch.Tensor:
    if stage_init_path is None:
        return bicubic_initialize_from_hs(hs_target_bchw, scale_factor=3)
    init_np = np.load(stage_init_path).astype(np.float32)
    if init_np.ndim != 3:
        raise ValueError(f"Stage init checkpoint must be 3D, got {init_np.shape}")
    if init_np.shape[0] == int(hs_target_bchw.shape[1]) and init_np.shape[-1] != int(hs_target_bchw.shape[1]):
        init_np = np.transpose(init_np, (1, 2, 0))
    return torch.from_numpy(init_np).permute(2, 0, 1).unsqueeze(0).to(hs_target_bchw.device)


def _is_lowfreq_residual_mode(stage_name: str, stage2_update_mode: str) -> bool:
    return stage_name == "stage2" and stage2_update_mode in {
        "lowfreq_residual",
        "veg_lowfreq_residual",
        "teacher_nir_non_nir_lowfreq",
        "teacher_nir_veg_lowfreq",
        "teacher_veg_band_alpha_delta",
    }


def _build_band_weight_from_wavelengths(
    wavelengths_nm: np.ndarray,
    blend_in_start_nm: float,
    protect_start_nm: float,
    protect_end_nm: float,
    blend_out_end_nm: float,
) -> np.ndarray:
    weights = np.zeros_like(np.asarray(wavelengths_nm, dtype=np.float32), dtype=np.float32)
    for idx, wl in enumerate(np.asarray(wavelengths_nm, dtype=np.float32).reshape(-1)):
        x = float(wl)
        if x < float(blend_in_start_nm):
            weights[idx] = 0.0
        elif x < float(protect_start_nm):
            denom = max(float(protect_start_nm) - float(blend_in_start_nm), 1e-6)
            weights[idx] = (x - float(blend_in_start_nm)) / denom
        elif x <= float(protect_end_nm):
            weights[idx] = 1.0
        elif x < float(blend_out_end_nm):
            denom = max(float(blend_out_end_nm) - float(protect_end_nm), 1e-6)
            weights[idx] = (float(blend_out_end_nm) - x) / denom
        else:
            weights[idx] = 0.0
    return np.clip(weights, 0.0, 1.0).astype(np.float32)


def _build_vegetation_mask_from_ms(ms_bchw: torch.Tensor, ndvi_thr: float) -> torch.Tensor:
    if ms_bchw.ndim != 4 or ms_bchw.shape[1] < 7:
        raise ValueError(f"Expected Sentinel-2 BCHW with at least 7 bands, got {tuple(ms_bchw.shape)}")
    b4 = ms_bchw[:, 2:3, :, :]
    b8 = ms_bchw[:, 6:7, :, :]
    ndvi = (b8 - b4) / (b8 + b4 + 1e-6)
    return (ndvi > float(ndvi_thr)).to(ms_bchw.dtype)


def _highfreq_drift_energy(
    hsi_bchw: torch.Tensor,
    ref_bchw: torch.Tensor,
    lowpass: FixedGaussianBlur,
) -> torch.Tensor:
    hsi_high = hsi_bchw - lowpass(hsi_bchw)
    ref_high = ref_bchw - lowpass(ref_bchw)
    return torch.mean((hsi_high - ref_high) ** 2)


def _write_csv(path: str, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def _stage_consistency_summary(
    stage_name: str,
    hsi_param: torch.Tensor,
    hs_target: torch.Tensor,
    ms_target: torch.Tensor,
    downsample: FixedGaussianDownsample,
    response: torch.Tensor,
    stage_end_iter: int,
    selection_interval: int,
    min_select_iter: int,
    best_iter: int | None,
    result_iter_used: int | None,
    total_best_global_iter: int | None,
    stage1_end_iter: int | None,
    stage1_stable_detected: bool | None,
    stage1_stability_rule: str | None,
    stage1_stability_last_rel_range: float | None,
    detail_lock_loss: float | None = None,
    highfreq_drift: float | None = None,
) -> dict[str, Any]:
    with torch.no_grad():
        hs_pred = downsample(hsi_param)
        ms_pred = mix_to_multispectral(hsi_param, response)
        hs_mse = torch.mean((hs_pred - hs_target) ** 2)
        ms_mse = torch.mean((ms_pred - ms_target) ** 2)
        osc_energy = second_order_energy(hsi_param)
    return {
        "stage_name": stage_name,
        "hs_mse": float(hs_mse.item()),
        "ms_mse": float(ms_mse.item()),
        "hs_psnr_db": float(psnr_from_mse(hs_mse)),
        "ms_psnr_db": float(psnr_from_mse(ms_mse)),
        "hs_sam_rad": float(sam_mean_rad(hs_pred, hs_target)),
        "ms_sam_rad": float(sam_mean_rad(ms_pred, ms_target)),
        "oscillation_energy": float(osc_energy.item()),
        "spectral_smoothness": float(osc_energy.item()),
        "selection_interval": int(selection_interval),
        "min_select_iter": int(min_select_iter),
        "last_iter": int(stage_end_iter),
        "best_iter": best_iter,
        "result_iter_used_for_metrics": result_iter_used,
        "result_iter_used_for_plot": result_iter_used,
        "result_iter_used_for_array_output": result_iter_used,
        "total_best_global_iter": total_best_global_iter,
        "stage1_end_iter": stage1_end_iter,
        "stage1_stable_detected": stage1_stable_detected,
        "stage1_stability_rule": stage1_stability_rule,
        "stage1_stability_last_rel_range": stage1_stability_last_rel_range,
        "loss_detail_lock": detail_lock_loss,
        "highfreq_drift": highfreq_drift,
    }


def run_training_stage(
    *,
    output_dir: str,
    stage_name: str,
    prisma_path: str = DEFAULT_PRISMA_PATH,
    s2_path: str = DEFAULT_S2_PATH,
    wavelengths_path: str = DEFAULT_WL_PATH,
    init_checkpoint_path: str | None = None,
    iters: int,
    lr: float,
    lambda_hs: float,
    lambda_ms: float,
    lambda_spec: float,
    lambda_tv: float,
    use_rtm: bool,
    keep_ms_loss: bool,
    anchor_strategy: str,
    anchor_mu_start: float,
    anchor_mu_end: float,
    selection_interval: int,
    min_select_iter: int,
    global_iter_offset: int,
    enable_stability_stop: bool,
    stage1_min_iters: int,
    stage1_stability_window: int,
    stage1_stability_patience: int,
    stage1_stability_rel_tol: float,
    stage2_score_patience: int,
    stage2_score_min_delta: float,
    stage2_update_mode: str = "full",
    stage2_lowpass_kernel_size: int = 9,
    stage2_lowpass_sigma: float = 2.0,
    lambda_delta: float = 0.0,
    lambda_delta_spec: float = 0.0,
    lambda_detail_lock: float = 0.0,
    stage2_score_ms_weight: float = 1.0,
    stage2_score_osc_weight: float = 0.3,
    stage2_spatial_gate_detail_max: float = float("inf"),
    stage2_spatial_gate_hf_max: float = float("inf"),
    stage2_veg_mask_ndvi_thr: float = 0.2,
    stage2_disable_veg_mask: bool = False,
    stage2_teacher_checkpoint_path: str | None = None,
    stage2_baseline_checkpoint_path: str | None = None,
    stage2_protect_band_min_nm: float = 700.0,
    stage2_protect_band_max_nm: float = 1300.0,
    stage2_blend_in_start_nm: float = 680.0,
    stage2_blend_out_end_nm: float = 1320.0,
    lambda_alpha_smooth: float = 0.0,
    rtm_chunk_size: int = 512,
    rtm_inv_path: str | None = None,
    rtm_fwd_path: str | None = None,
    rtm_scalers_path: str | None = None,
    device: str | None = None,
    logger: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    device_name = select_device(device)

    def log(message: str) -> None:
        if logger is not None:
            logger(message)

    data = load_core_data(
        prisma_path=prisma_path,
        s2_path=s2_path,
        wavelength_path=wavelengths_path,
        normalize=True,
    )
    save_sort_index(os.path.join(output_dir, "sort_idx.npy"), data.sort_idx)

    hs_target = torch.from_numpy(data.hs_hwc_sorted).permute(2, 0, 1).unsqueeze(0).to(device_name)
    ms_target = torch.from_numpy(data.ms_hwc).permute(2, 0, 1).unsqueeze(0).to(device_name)
    hsi_init = _load_init_hsi(init_checkpoint_path, hs_target)
    use_lowfreq_residual = _is_lowfreq_residual_mode(stage_name, stage2_update_mode)
    base_hsi = hsi_init.detach().clone() if use_lowfreq_residual else None
    use_veg_masked_lowfreq = stage_name == "stage2" and stage2_update_mode in {
        "veg_lowfreq_residual",
        "teacher_nir_veg_lowfreq",
        "teacher_veg_band_alpha",
        "teacher_veg_band_alpha_delta",
    }
    use_teacher_nir_protect = stage_name == "stage2" and stage2_update_mode in {
        "teacher_nir_non_nir_lowfreq",
        "teacher_nir_veg_lowfreq",
    }
    use_teacher_veg_alpha = stage_name == "stage2" and stage2_update_mode in {
        "teacher_veg_band_alpha",
        "teacher_veg_band_alpha_delta",
    }
    exclude_protected_bands_from_detail_lock = stage_name == "stage2" and stage2_update_mode in {
        "teacher_nir_non_nir_lowfreq",
        "teacher_nir_veg_lowfreq",
    }

    response = build_gaussian_s2_response(data.wavelengths_sorted_nm, device=device_name)
    downsample = FixedGaussianDownsample(
        channels=int(hs_target.shape[1]),
        scale=3,
        kernel_size=11,
        sigma=1.2,
    ).to(device_name)
    spatial_lowpass = FixedGaussianBlur(
        channels=int(hs_target.shape[1]),
        kernel_size=int(stage2_lowpass_kernel_size),
        sigma=float(stage2_lowpass_sigma),
    ).to(device_name)
    veg_mask = None
    veg_ratio = None
    if use_veg_masked_lowfreq and not stage2_disable_veg_mask:
        veg_mask = _build_vegetation_mask_from_ms(ms_target, ndvi_thr=float(stage2_veg_mask_ndvi_thr))
        veg_ratio = float(veg_mask.mean().item())
    protected_band_mask = None
    non_protected_band_mask = None
    teacher_hsi = None
    baseline_hsi = None
    alpha_init_tensor = None
    if use_teacher_nir_protect:
        if not stage2_teacher_checkpoint_path:
            raise ValueError(
                "stage2_teacher_checkpoint_path is required for teacher_nir_non_nir_lowfreq/teacher_nir_veg_lowfreq mode"
            )
        teacher_hsi = _load_init_hsi(stage2_teacher_checkpoint_path, hs_target).detach()
        wl = torch.as_tensor(data.wavelengths_sorted_nm, dtype=torch.float32, device=device_name)
        protected_band_mask = (
            (wl >= float(stage2_protect_band_min_nm)) & (wl <= float(stage2_protect_band_max_nm))
        ).to(torch.float32).view(1, -1, 1, 1)
        non_protected_band_mask = 1.0 - protected_band_mask
    if use_teacher_veg_alpha:
        if not stage2_teacher_checkpoint_path:
            raise ValueError("stage2_teacher_checkpoint_path is required for teacher_veg_band_alpha modes")
        if not stage2_baseline_checkpoint_path:
            raise ValueError("stage2_baseline_checkpoint_path is required for teacher_veg_band_alpha modes")
        teacher_hsi = _load_init_hsi(stage2_teacher_checkpoint_path, hs_target).detach()
        baseline_hsi = _load_init_hsi(stage2_baseline_checkpoint_path, hs_target).detach()
        wl = np.asarray(data.wavelengths_sorted_nm, dtype=np.float32).reshape(-1)
        alpha_init_np = _build_band_weight_from_wavelengths(
            wavelengths_nm=wl,
            blend_in_start_nm=float(stage2_blend_in_start_nm),
            protect_start_nm=float(stage2_protect_band_min_nm),
            protect_end_nm=float(stage2_protect_band_max_nm),
            blend_out_end_nm=float(stage2_blend_out_end_nm),
        )
        alpha_init_tensor = torch.from_numpy(alpha_init_np).to(device_name).view(1, -1, 1, 1)
        protected_band_mask = (
            (torch.as_tensor(wl, dtype=torch.float32, device=device_name) >= float(stage2_protect_band_min_nm))
            & (torch.as_tensor(wl, dtype=torch.float32, device=device_name) <= float(stage2_protect_band_max_nm))
        ).to(torch.float32).view(1, -1, 1, 1)
        non_protected_band_mask = 1.0 - protected_band_mask
    if use_lowfreq_residual:
        if stage2_update_mode == "teacher_veg_band_alpha_delta":
            base_hsi = baseline_hsi.detach().clone()
            alpha_init = torch.clamp(alpha_init_tensor, 1e-4, 1.0 - 1e-4)
            alpha_logits = nn.Parameter(torch.log(alpha_init / (1.0 - alpha_init)))
            delta_param = nn.Parameter(torch.zeros_like(base_hsi))

            def compose_alpha_band() -> torch.Tensor:
                alpha_band = torch.sigmoid(alpha_logits)
                if protected_band_mask is not None:
                    alpha_band = alpha_band * non_protected_band_mask + protected_band_mask
                return alpha_band

            def compose_hsi() -> tuple[torch.Tensor, torch.Tensor]:
                alpha_band = compose_alpha_band()
                current = base_hsi + alpha_band * (teacher_hsi - base_hsi)
                if veg_mask is not None:
                    current = base_hsi + veg_mask * (current - base_hsi)
                delta_low = spatial_lowpass(delta_param)
                delta_effective = delta_low
                if veg_mask is not None:
                    delta_effective = delta_effective * veg_mask
                if non_protected_band_mask is not None:
                    delta_effective = delta_effective * non_protected_band_mask
                current = torch.clamp(current + delta_effective, 0.0, 1.0)
                return current, delta_effective

            optimizer = torch.optim.Adam([alpha_logits, delta_param], lr=lr)
        else:
            base_low = spatial_lowpass(base_hsi).detach()
            base_high = (base_hsi - base_low).detach()
            delta_param = nn.Parameter(torch.zeros_like(base_hsi))

            def compose_alpha_band() -> torch.Tensor:
                return torch.zeros((1, int(hs_target.shape[1]), 1, 1), device=device_name)

            def compose_hsi() -> tuple[torch.Tensor, torch.Tensor]:
                delta_low = spatial_lowpass(delta_param)
                delta_effective = delta_low
                if veg_mask is not None:
                    delta_effective = delta_effective * veg_mask
                if non_protected_band_mask is not None:
                    delta_effective = delta_effective * non_protected_band_mask
                current = torch.clamp(base_high + base_low + delta_effective, 0.0, 1.0)
                if protected_band_mask is not None and teacher_hsi is not None:
                    current = current * non_protected_band_mask + teacher_hsi * protected_band_mask
                return current, delta_effective

            optimizer = torch.optim.Adam([delta_param], lr=lr)
    else:
        if stage2_update_mode == "teacher_veg_band_alpha":
            if baseline_hsi is None or teacher_hsi is None or alpha_init_tensor is None:
                raise ValueError("teacher_veg_band_alpha mode requires teacher, baseline, and alpha init")
            base_hsi = baseline_hsi.detach().clone()
            alpha_init = torch.clamp(alpha_init_tensor, 1e-4, 1.0 - 1e-4)
            alpha_logits = nn.Parameter(torch.log(alpha_init / (1.0 - alpha_init)))

            def compose_alpha_band() -> torch.Tensor:
                alpha_band = torch.sigmoid(alpha_logits)
                if protected_band_mask is not None:
                    alpha_band = alpha_band * non_protected_band_mask + protected_band_mask
                return alpha_band

            def compose_hsi() -> tuple[torch.Tensor, torch.Tensor]:
                alpha_band = compose_alpha_band()
                current = base_hsi + alpha_band * (teacher_hsi - base_hsi)
                if veg_mask is not None:
                    current = base_hsi + veg_mask * (current - base_hsi)
                current = torch.clamp(current, 0.0, 1.0)
                delta_effective = torch.zeros_like(current)
                return current, delta_effective

            optimizer = torch.optim.Adam([alpha_logits], lr=lr)
        else:
            hsi_param = nn.Parameter(hsi_init.clone())

            def compose_alpha_band() -> torch.Tensor:
                return torch.zeros((1, int(hs_target.shape[1]), 1, 1), device=device_name)

            def compose_hsi() -> tuple[torch.Tensor, torch.Tensor]:
                current = hsi_param
                delta_effective = torch.zeros_like(current)
                return current, delta_effective

            optimizer = torch.optim.Adam([hsi_param], lr=lr)
    weights = LossWeights(
        lambda_hs=lambda_hs,
        lambda_ms=(lambda_ms if keep_ms_loss else 0.0),
        lambda_spec=lambda_spec,
        lambda_tv=lambda_tv,
    )
    projector = (
        RTMAnchorProjector(
            device=device_name,
            chunk_size=rtm_chunk_size,
            inv_path=(rtm_inv_path or DEFAULT_INV_PATH),
            fwd_path=(rtm_fwd_path or DEFAULT_FWD_PATH),
            scalers_path=(rtm_scalers_path or DEFAULT_SCALERS_PATH),
        )
        if use_rtm
        else None
    )
    initial_hsi, _ = compose_hsi()
    anchor_tensor = projector.project(initial_hsi.detach()) if projector is not None else None

    train_history: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    best_score = float("inf")
    best_iter = None
    best_checkpoint_path = None
    best_metrics = None
    best_fallback_key = None
    fallback_iter = None
    fallback_checkpoint_path = None
    fallback_metrics = None
    selection_used_spatial_fallback = False

    stability_hits = 0
    stability_last_rel_range = None
    stage_stable_detected = False
    stage_end_iter = int(iters)
    stage_stability_rule = (
        f"window={int(stage1_stability_window)}, patience={int(stage1_stability_patience)}, "
        f"rel_tol={float(stage1_stability_rel_tol):.6e}, min_iters={int(stage1_min_iters)}"
    )
    stage2_no_improve_count = 0
    stage2_early_stop_triggered = False
    stage2_early_stop_reason = None
    candidate_dir = os.path.join(output_dir, "candidate_checkpoints")
    os.makedirs(candidate_dir, exist_ok=True)

    for iter_idx in range(int(iters) + 1):
        optimizer.zero_grad()
        current_hsi, delta_effective = compose_hsi()
        alpha_band = compose_alpha_band()
        hs_pred = downsample(current_hsi)
        ms_pred = mix_to_multispectral(current_hsi, response)
        total_loss, pieces = build_total_loss(
            hsi_hr_bchw=current_hsi,
            hs_pred_bchw=hs_pred,
            hs_target_bchw=hs_target,
            ms_pred_bchw=ms_pred,
            ms_target_bchw=ms_target,
            weights=weights,
        )
        delta_penalty = torch.tensor(0.0, device=device_name)
        delta_spec_loss = torch.tensor(0.0, device=device_name)
        alpha_smooth_loss = torch.tensor(0.0, device=device_name)
        detail_lock_loss = torch.tensor(0.0, device=device_name)
        if use_lowfreq_residual:
            delta_penalty = torch.mean(delta_effective ** 2)
            delta_spec_loss = spectral_smoothness_loss(delta_effective)
            total_loss = (
                total_loss
                + float(lambda_delta) * delta_penalty
                + float(lambda_delta_spec) * delta_spec_loss
            )
        if use_teacher_veg_alpha:
            alpha_smooth_loss = spectral_smoothness_loss(alpha_band)
            total_loss = total_loss + float(lambda_alpha_smooth) * alpha_smooth_loss
        if stage_name == "stage2" and base_hsi is not None and stage2_update_mode != "full":
            detail_ref_current = current_hsi
            detail_ref_base = base_hsi
            if exclude_protected_bands_from_detail_lock and non_protected_band_mask is not None:
                detail_ref_current = detail_ref_current * non_protected_band_mask
                detail_ref_base = detail_ref_base * non_protected_band_mask
            detail_lock_loss = spatial_gradient_consistency_loss(detail_ref_current, detail_ref_base)
            total_loss = total_loss + float(lambda_detail_lock) * detail_lock_loss
        anchor_weight = 0.0
        anchor_loss_value = torch.tensor(0.0, device=device_name)
        if use_rtm and anchor_tensor is not None:
            anchor_weight = _compute_anchor_weight(
                anchor_strategy,
                iter_idx=iter_idx,
                total_iters=max(1, int(iters)),
                mu_start=anchor_mu_start,
                mu_end=anchor_mu_end,
            )
            anchor_loss_value = float(anchor_weight) * amplitude_anchor_loss(current_hsi, anchor_tensor)
            total_loss = total_loss + anchor_loss_value

        total_loss.backward()
        optimizer.step()
        if projector is not None:
            current_hsi_after_step, _ = compose_hsi()
            anchor_tensor = projector.project(current_hsi_after_step.detach())

        row = {
            "iter": int(iter_idx),
            "global_iter": int(global_iter_offset + iter_idx),
            "stage_id": stage_name,
            "loss_total": float(total_loss.item()),
            "loss_hs": float(pieces["loss_hs"].item()),
            "loss_ms": float(pieces["loss_ms"].item()),
            "loss_spec": float(pieces["loss_spec"].item()),
            "loss_tv": float(pieces["loss_tv"].item()),
            "anchor_weight": float(anchor_weight),
            "anchor_loss": float(anchor_loss_value.item()),
            "loss_delta": float(delta_penalty.item()),
            "loss_delta_spec": float(delta_spec_loss.item()),
            "loss_alpha_smooth": float(alpha_smooth_loss.item()),
            "loss_detail_lock": float(detail_lock_loss.item()),
            "stage2_update_mode": stage2_update_mode,
        }
        train_history.append(row)

        if iter_idx % int(selection_interval) == 0 or iter_idx == int(iters):
            with torch.no_grad():
                current_hsi_eval, _ = compose_hsi()
                hs_pred_eval = downsample(current_hsi_eval)
                ms_pred_eval = mix_to_multispectral(current_hsi_eval, response)
                hs_reproj = torch.mean((hs_pred_eval - hs_target) ** 2)
                ms_reproj = torch.mean((ms_pred_eval - ms_target) ** 2)
                osc_energy = second_order_energy(current_hsi_eval)
                detail_lock_eval = torch.tensor(0.0, device=device_name)
                highfreq_drift_eval = torch.tensor(0.0, device=device_name)
                if stage_name == "stage2" and base_hsi is not None and stage2_update_mode != "full":
                    detail_eval_current = current_hsi_eval
                    detail_eval_base = base_hsi
                    if exclude_protected_bands_from_detail_lock and non_protected_band_mask is not None:
                        detail_eval_current = detail_eval_current * non_protected_band_mask
                        detail_eval_base = detail_eval_base * non_protected_band_mask
                    detail_lock_eval = spatial_gradient_consistency_loss(detail_eval_current, detail_eval_base)
                    highfreq_drift_eval = _highfreq_drift_energy(detail_eval_current, detail_eval_base, spatial_lowpass)
            score = float(
                hs_reproj.item()
                + float(stage2_score_ms_weight) * ms_reproj.item()
                + float(stage2_score_osc_weight) * osc_energy.item()
            )
            candidate_path = os.path.join(candidate_dir, f"hsi_iter{iter_idx:04d}.npy")
            np.save(
                candidate_path,
                current_hsi_eval.detach().cpu().squeeze(0).permute(1, 2, 0).numpy().astype(np.float32),
            )
            spatial_gate_ok = True
            if stage_name == "stage2" and base_hsi is not None and stage2_update_mode != "full":
                if float(detail_lock_eval.item()) > float(stage2_spatial_gate_detail_max):
                    spatial_gate_ok = False
                if float(highfreq_drift_eval.item()) > float(stage2_spatial_gate_hf_max):
                    spatial_gate_ok = False
            eligible = bool(stage_name == "stage2" and int(iter_idx) >= int(min_select_iter) and spatial_gate_ok)
            is_best = False
            improved = False
            if eligible and score < (best_score - float(stage2_score_min_delta)):
                best_score = score
                best_iter = int(iter_idx)
                best_checkpoint_path = candidate_path
                best_metrics = {
                    "hs_reproj": float(hs_reproj.item()),
                    "ms_reproj": float(ms_reproj.item()),
                    "osc_2diff_energy": float(osc_energy.item()),
                    "loss_detail_lock": float(detail_lock_eval.item()),
                    "highfreq_drift": float(highfreq_drift_eval.item()),
                    "score": float(score),
                }
                is_best = True
                improved = True
                stage2_no_improve_count = 0
            if stage_name == "stage2" and int(iter_idx) >= int(min_select_iter):
                detail_excess = max(0.0, float(detail_lock_eval.item()) - float(stage2_spatial_gate_detail_max))
                hf_excess = max(0.0, float(highfreq_drift_eval.item()) - float(stage2_spatial_gate_hf_max))
                fallback_key = (detail_excess + hf_excess, float(score))
                if best_fallback_key is None or fallback_key < best_fallback_key:
                    best_fallback_key = fallback_key
                    fallback_iter = int(iter_idx)
                    fallback_checkpoint_path = candidate_path
                    fallback_metrics = {
                        "hs_reproj": float(hs_reproj.item()),
                        "ms_reproj": float(ms_reproj.item()),
                        "osc_2diff_energy": float(osc_energy.item()),
                        "loss_detail_lock": float(detail_lock_eval.item()),
                        "highfreq_drift": float(highfreq_drift_eval.item()),
                        "score": float(score),
                        "spatial_gate_passed": bool(spatial_gate_ok),
                        "spatial_gate_violation": float(detail_excess + hf_excess),
                    }
            checkpoint_rows.append(
                {
                    "global_iter": int(global_iter_offset + iter_idx),
                    "stage_id": stage_name,
                    "stage2_iter": (int(iter_idx) if stage_name == "stage2" else ""),
                    "iter": int(iter_idx),
                    "hs_reproj": float(hs_reproj.item()),
                    "ms_reproj": float(ms_reproj.item()),
                    "osc_2diff_energy": float(osc_energy.item()),
                    "loss_detail_lock": float(detail_lock_eval.item()),
                    "highfreq_drift": float(highfreq_drift_eval.item()),
                    "score": float(score),
                    "eligible_for_best": bool(eligible),
                    "spatial_gate_passed": bool(spatial_gate_ok),
                    "is_best": bool(is_best),
                    "candidate_checkpoint": candidate_path,
                }
            )
            log(
                f"[{stage_name}][{iter_idx:04d}] loss={row['loss_total']:.6f} "
                f"L_hs={row['loss_hs']:.6f} L_ms={row['loss_ms']:.6f} "
                f"L_delta={row['loss_delta']:.6f} L_dspec={row['loss_delta_spec']:.6f} "
                f"L_asmooth={row['loss_alpha_smooth']:.6f} "
                f"L_detail={row['loss_detail_lock']:.6f} "
                f"anchor_w={row['anchor_weight']:.6f} anchor_loss={row['anchor_loss']:.6f} "
                f"hs_reproj={float(hs_reproj.item()):.6e} ms_reproj={float(ms_reproj.item()):.6e} "
                f"osc={float(osc_energy.item()):.6e} hf_drift={float(highfreq_drift_eval.item()):.6e} "
                f"score={score:.6f} gate={spatial_gate_ok} eligible={eligible}"
            )
            if stage_name == "stage2" and eligible and best_iter is not None and not improved:
                stage2_no_improve_count += 1
                if int(stage2_score_patience) > 0 and stage2_no_improve_count >= int(stage2_score_patience):
                    stage2_early_stop_triggered = True
                    stage2_early_stop_reason = (
                        f"no blind-score improvement for {int(stage2_no_improve_count)} eligible checkpoints "
                        f"(patience={int(stage2_score_patience)}, min_delta={float(stage2_score_min_delta):.6e})"
                    )
                    stage_end_iter = int(iter_idx)
                    log(
                        f"[{stage_name}][early_stop] iter={stage_end_iter} reason={stage2_early_stop_reason}"
                    )
                    break

        if enable_stability_stop and int(iter_idx) >= int(stage1_min_iters):
            if len(train_history) >= int(stage1_stability_window):
                recent = np.array(
                    [float(item["loss_total"]) for item in train_history[-int(stage1_stability_window):]],
                    dtype=np.float64,
                )
                rel_range = float((recent.max() - recent.min()) / (abs(recent.mean()) + 1e-12))
                stability_last_rel_range = rel_range
                if rel_range <= float(stage1_stability_rel_tol):
                    stability_hits += 1
                else:
                    stability_hits = 0
                if stability_hits >= int(stage1_stability_patience):
                    stage_stable_detected = True
                    stage_end_iter = int(iter_idx)
                    log(
                        f"[{stage_name}][stable] iter={stage_end_iter} rel_range={rel_range:.6e} "
                        f"rule={stage_stability_rule}"
                    )
                    break

    final_path = os.path.join(output_dir, f"{stage_name}_prediction.npy")
    final_hsi, _ = compose_hsi()
    np.save(final_path, final_hsi.detach().cpu().squeeze(0).permute(1, 2, 0).numpy().astype(np.float32))

    selected_checkpoint_path = final_path
    result_iter_used = int(stage_end_iter)
    if stage_name == "stage2" and best_checkpoint_path is None and fallback_checkpoint_path is not None:
        best_checkpoint_path = fallback_checkpoint_path
        best_iter = fallback_iter
        best_metrics = fallback_metrics
        best_score = float(fallback_metrics["score"])
        selection_used_spatial_fallback = True
    if stage_name == "stage2" and best_checkpoint_path is not None:
        selected_checkpoint_path = best_checkpoint_path
        result_iter_used = int(best_iter)
    selected_hsi = torch.from_numpy(np.load(selected_checkpoint_path).astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(device_name)
    selected_detail_lock = None
    selected_highfreq_drift = None
    if stage_name == "stage2" and base_hsi is not None and stage2_update_mode != "full":
        with torch.no_grad():
            detail_selected_current = selected_hsi
            detail_selected_base = base_hsi
            if exclude_protected_bands_from_detail_lock and non_protected_band_mask is not None:
                detail_selected_current = detail_selected_current * non_protected_band_mask
                detail_selected_base = detail_selected_base * non_protected_band_mask
            selected_detail_lock = float(spatial_gradient_consistency_loss(detail_selected_current, detail_selected_base).item())
            selected_highfreq_drift = float(_highfreq_drift_energy(detail_selected_current, detail_selected_base, spatial_lowpass).item())
    consistency_summary = _stage_consistency_summary(
        stage_name=stage_name,
        hsi_param=selected_hsi,
        hs_target=hs_target,
        ms_target=ms_target,
        downsample=downsample,
        response=response,
        stage_end_iter=int(stage_end_iter),
        selection_interval=int(selection_interval),
        min_select_iter=int(min_select_iter),
        best_iter=best_iter,
        result_iter_used=result_iter_used,
        total_best_global_iter=(int(global_iter_offset + result_iter_used) if stage_name == "stage2" else None),
        stage1_end_iter=(int(stage_end_iter) if stage_name == "stage1" else None),
        stage1_stable_detected=(bool(stage_stable_detected) if stage_name == "stage1" else None),
        stage1_stability_rule=(stage_stability_rule if stage_name == "stage1" else None),
        stage1_stability_last_rel_range=stability_last_rel_range,
        detail_lock_loss=selected_detail_lock,
        highfreq_drift=selected_highfreq_drift,
    )
    save_json(os.path.join(output_dir, "train_no_gt_consistency.json"), consistency_summary)
    append_history_csv(os.path.join(output_dir, "train_history.csv"), train_history)
    _write_csv(
        os.path.join(output_dir, "checkpoint_scores.csv"),
        checkpoint_rows,
        [
            "global_iter",
            "stage_id",
            "stage2_iter",
            "iter",
            "hs_reproj",
            "ms_reproj",
            "osc_2diff_energy",
            "loss_detail_lock",
            "highfreq_drift",
            "score",
            "eligible_for_best",
            "spatial_gate_passed",
            "is_best",
            "candidate_checkpoint",
        ],
    )

    best_info = {
        "stage_id": stage_name,
        "last_iter": int(stage_end_iter),
        "best_iter": best_iter,
        "best_score": (None if best_iter is None else float(best_score)),
        "best_metrics": best_metrics,
        "selection_rule": (
            f"spatial_gate(detail<={float(stage2_spatial_gate_detail_max):.6e}, "
            f"hf<={float(stage2_spatial_gate_hf_max):.6e}) then "
            f"score = hs_reproj + {float(stage2_score_ms_weight):.6f} * ms_reproj + "
            f"{float(stage2_score_osc_weight):.6f} * osc_2diff_energy"
        ),
        "selection_interval": int(selection_interval),
        "min_select_iter": int(min_select_iter),
        "result_iter_used_for_metrics": int(result_iter_used),
        "result_iter_used_for_plot": int(result_iter_used),
        "result_iter_used_for_array_output": int(result_iter_used),
        "stage1_end_iter": (int(stage_end_iter) if stage_name == "stage1" else None),
        "stage1_stable_detected": (bool(stage_stable_detected) if stage_name == "stage1" else None),
        "stage1_stability_rule": stage_stability_rule,
        "stage1_stability_last_rel_range": stability_last_rel_range,
        "selected_checkpoint": os.path.abspath(selected_checkpoint_path),
        "final_checkpoint": os.path.abspath(final_path),
        "checkpoint_scores_csv": os.path.abspath(os.path.join(output_dir, "checkpoint_scores.csv")),
        "stage2_early_stop_triggered": bool(stage2_early_stop_triggered),
        "stage2_early_stop_reason": stage2_early_stop_reason,
        "stage2_score_ms_weight": float(stage2_score_ms_weight),
        "stage2_score_osc_weight": float(stage2_score_osc_weight),
        "stage2_spatial_gate_detail_max": float(stage2_spatial_gate_detail_max),
        "stage2_spatial_gate_hf_max": float(stage2_spatial_gate_hf_max),
        "stage2_selection_used_spatial_fallback": bool(selection_used_spatial_fallback),
    }
    save_json(os.path.join(output_dir, "best_checkpoint_info.json"), best_info)
    save_json(
        os.path.join(output_dir, "train_config.json"),
        {
            "stage_name": stage_name,
            "prisma_path": os.path.abspath(prisma_path),
            "s2_path": os.path.abspath(s2_path),
            "wavelengths_path": os.path.abspath(wavelengths_path),
            "init_checkpoint_path": init_checkpoint_path,
            "iters": int(iters),
            "lr": float(lr),
            "lambda_hs": float(lambda_hs),
            "lambda_ms": float(lambda_ms),
            "lambda_spec": float(lambda_spec),
            "lambda_tv": float(lambda_tv),
            "use_rtm": bool(use_rtm),
            "keep_ms_loss": bool(keep_ms_loss),
            "anchor_strategy": anchor_strategy,
            "anchor_mu_start": float(anchor_mu_start),
            "anchor_mu_end": float(anchor_mu_end),
            "selection_interval": int(selection_interval),
            "min_select_iter": int(min_select_iter),
            "global_iter_offset": int(global_iter_offset),
            "enable_stability_stop": bool(enable_stability_stop),
            "stage1_min_iters": int(stage1_min_iters),
            "stage1_stability_window": int(stage1_stability_window),
            "stage1_stability_patience": int(stage1_stability_patience),
            "stage1_stability_rel_tol": float(stage1_stability_rel_tol),
            "stage2_score_patience": int(stage2_score_patience),
            "stage2_score_min_delta": float(stage2_score_min_delta),
            "stage2_update_mode": stage2_update_mode,
            "stage2_lowpass_kernel_size": int(stage2_lowpass_kernel_size),
            "stage2_lowpass_sigma": float(stage2_lowpass_sigma),
            "lambda_delta": float(lambda_delta),
            "lambda_delta_spec": float(lambda_delta_spec),
            "lambda_detail_lock": float(lambda_detail_lock),
            "stage2_score_ms_weight": float(stage2_score_ms_weight),
            "stage2_score_osc_weight": float(stage2_score_osc_weight),
            "stage2_spatial_gate_detail_max": float(stage2_spatial_gate_detail_max),
            "stage2_spatial_gate_hf_max": float(stage2_spatial_gate_hf_max),
            "stage2_veg_mask_ndvi_thr": float(stage2_veg_mask_ndvi_thr),
            "stage2_disable_veg_mask": bool(stage2_disable_veg_mask),
            "stage2_teacher_checkpoint_path": stage2_teacher_checkpoint_path,
            "stage2_baseline_checkpoint_path": stage2_baseline_checkpoint_path,
            "stage2_protect_band_min_nm": float(stage2_protect_band_min_nm),
            "stage2_protect_band_max_nm": float(stage2_protect_band_max_nm),
            "stage2_blend_in_start_nm": float(stage2_blend_in_start_nm),
            "stage2_blend_out_end_nm": float(stage2_blend_out_end_nm),
            "lambda_alpha_smooth": float(lambda_alpha_smooth),
            "rtm_inv_path": os.path.abspath(rtm_inv_path or DEFAULT_INV_PATH),
            "rtm_fwd_path": os.path.abspath(rtm_fwd_path or DEFAULT_FWD_PATH),
            "rtm_scalers_path": os.path.abspath(rtm_scalers_path or DEFAULT_SCALERS_PATH),
            "veg_ratio": veg_ratio,
            "device": device_name,
            "sort_idx_path": os.path.join(output_dir, "sort_idx.npy"),
        },
    )
    return {
        "output_dir": output_dir,
        "selected_checkpoint": selected_checkpoint_path,
        "final_checkpoint": final_path,
        "train_history_path": os.path.join(output_dir, "train_history.csv"),
        "checkpoint_scores_path": os.path.join(output_dir, "checkpoint_scores.csv"),
        "best_checkpoint_info_path": os.path.join(output_dir, "best_checkpoint_info.json"),
        "train_no_gt_consistency_path": os.path.join(output_dir, "train_no_gt_consistency.json"),
        "stage_end_iter": int(stage_end_iter),
        "stage_stable_detected": bool(stage_stable_detected),
        "stage_stability_rule": stage_stability_rule,
        "stage_stability_last_rel_range": stability_last_rel_range,
        "best_iter": best_iter,
        "total_best_global_iter": (int(global_iter_offset + best_iter) if best_iter is not None else None),
        "stage2_early_stop_triggered": bool(stage2_early_stop_triggered),
        "stage2_early_stop_reason": stage2_early_stop_reason,
        "stage2_update_mode": stage2_update_mode,
        "stage2_selection_used_spatial_fallback": bool(selection_used_spatial_fallback),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean stagewise trainer with Stage1 stability and Stage2 RTM anchor.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--stage_name", choices=["stage1", "stage2"], required=True)
    parser.add_argument("--prisma_path", default=DEFAULT_PRISMA_PATH)
    parser.add_argument("--s2_path", default=DEFAULT_S2_PATH)
    parser.add_argument("--wavelengths_path", default=DEFAULT_WL_PATH)
    parser.add_argument("--init_checkpoint_path", default=None)
    parser.add_argument("--iters", type=int, required=True)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--lambda_hs", type=float, default=1.0)
    parser.add_argument("--lambda_ms", type=float, default=1.0)
    parser.add_argument("--lambda_spec", type=float, default=1e-2)
    parser.add_argument("--lambda_tv", type=float, default=1e-3)
    parser.add_argument("--use_rtm", action="store_true")
    parser.add_argument("--disable_ms_loss", action="store_true")
    parser.add_argument("--anchor_strategy", default="constant", choices=["constant", "linear_decay", "cosine_decay"])
    parser.add_argument("--anchor_mu_start", type=float, default=0.5)
    parser.add_argument("--anchor_mu_end", type=float, default=0.1)
    parser.add_argument("--selection_interval", type=int, default=50)
    parser.add_argument("--min_select_iter", type=int, default=100)
    parser.add_argument("--global_iter_offset", type=int, default=0)
    parser.add_argument("--enable_stability_stop", action="store_true")
    parser.add_argument("--stage1_min_iters", type=int, default=500)
    parser.add_argument("--stage1_stability_window", type=int, default=50)
    parser.add_argument("--stage1_stability_patience", type=int, default=3)
    parser.add_argument("--stage1_stability_rel_tol", type=float, default=1e-3)
    parser.add_argument("--stage2_score_patience", type=int, default=0)
    parser.add_argument("--stage2_score_min_delta", type=float, default=0.0)
    parser.add_argument(
        "--stage2_update_mode",
        choices=[
            "full",
            "lowfreq_residual",
            "veg_lowfreq_residual",
            "teacher_nir_non_nir_lowfreq",
            "teacher_nir_veg_lowfreq",
            "teacher_veg_band_alpha",
            "teacher_veg_band_alpha_delta",
        ],
        default="full",
    )
    parser.add_argument("--stage2_lowpass_kernel_size", type=int, default=9)
    parser.add_argument("--stage2_lowpass_sigma", type=float, default=2.0)
    parser.add_argument("--lambda_delta", type=float, default=0.0)
    parser.add_argument("--lambda_delta_spec", type=float, default=0.0)
    parser.add_argument("--lambda_detail_lock", type=float, default=0.0)
    parser.add_argument("--stage2_score_ms_weight", type=float, default=1.0)
    parser.add_argument("--stage2_score_osc_weight", type=float, default=0.3)
    parser.add_argument("--stage2_spatial_gate_detail_max", type=float, default=float("inf"))
    parser.add_argument("--stage2_spatial_gate_hf_max", type=float, default=float("inf"))
    parser.add_argument("--stage2_veg_mask_ndvi_thr", type=float, default=0.2)
    parser.add_argument("--stage2_disable_veg_mask", action="store_true")
    parser.add_argument("--stage2_teacher_checkpoint_path", default=None)
    parser.add_argument("--stage2_baseline_checkpoint_path", default=None)
    parser.add_argument("--stage2_protect_band_min_nm", type=float, default=700.0)
    parser.add_argument("--stage2_protect_band_max_nm", type=float, default=1300.0)
    parser.add_argument("--stage2_blend_in_start_nm", type=float, default=680.0)
    parser.add_argument("--stage2_blend_out_end_nm", type=float, default=1320.0)
    parser.add_argument("--lambda_alpha_smooth", type=float, default=0.0)
    parser.add_argument("--rtm_chunk_size", type=int, default=512)
    parser.add_argument("--rtm_inv_path", default=None)
    parser.add_argument("--rtm_fwd_path", default=None)
    parser.add_argument("--rtm_scalers_path", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    result = run_training_stage(
        output_dir=args.output_dir,
        stage_name=args.stage_name,
        prisma_path=args.prisma_path,
        s2_path=args.s2_path,
        wavelengths_path=args.wavelengths_path,
        init_checkpoint_path=args.init_checkpoint_path,
        iters=args.iters,
        lr=args.lr,
        lambda_hs=args.lambda_hs,
        lambda_ms=args.lambda_ms,
        lambda_spec=args.lambda_spec,
        lambda_tv=args.lambda_tv,
        use_rtm=bool(args.use_rtm),
        keep_ms_loss=(not bool(args.disable_ms_loss)),
        anchor_strategy=args.anchor_strategy,
        anchor_mu_start=args.anchor_mu_start,
        anchor_mu_end=args.anchor_mu_end,
        selection_interval=args.selection_interval,
        min_select_iter=args.min_select_iter,
        global_iter_offset=args.global_iter_offset,
        enable_stability_stop=bool(args.enable_stability_stop),
        stage1_min_iters=args.stage1_min_iters,
        stage1_stability_window=args.stage1_stability_window,
        stage1_stability_patience=args.stage1_stability_patience,
        stage1_stability_rel_tol=args.stage1_stability_rel_tol,
        stage2_score_patience=args.stage2_score_patience,
        stage2_score_min_delta=args.stage2_score_min_delta,
        stage2_update_mode=args.stage2_update_mode,
        stage2_lowpass_kernel_size=args.stage2_lowpass_kernel_size,
        stage2_lowpass_sigma=args.stage2_lowpass_sigma,
        lambda_delta=args.lambda_delta,
        lambda_delta_spec=args.lambda_delta_spec,
        lambda_detail_lock=args.lambda_detail_lock,
        stage2_score_ms_weight=args.stage2_score_ms_weight,
        stage2_score_osc_weight=args.stage2_score_osc_weight,
        stage2_spatial_gate_detail_max=args.stage2_spatial_gate_detail_max,
        stage2_spatial_gate_hf_max=args.stage2_spatial_gate_hf_max,
        stage2_veg_mask_ndvi_thr=args.stage2_veg_mask_ndvi_thr,
        stage2_disable_veg_mask=args.stage2_disable_veg_mask,
        stage2_teacher_checkpoint_path=args.stage2_teacher_checkpoint_path,
        stage2_baseline_checkpoint_path=args.stage2_baseline_checkpoint_path,
        stage2_protect_band_min_nm=args.stage2_protect_band_min_nm,
        stage2_protect_band_max_nm=args.stage2_protect_band_max_nm,
        stage2_blend_in_start_nm=args.stage2_blend_in_start_nm,
        stage2_blend_out_end_nm=args.stage2_blend_out_end_nm,
        lambda_alpha_smooth=args.lambda_alpha_smooth,
        rtm_chunk_size=args.rtm_chunk_size,
        rtm_inv_path=args.rtm_inv_path,
        rtm_fwd_path=args.rtm_fwd_path,
        rtm_scalers_path=args.rtm_scalers_path,
        device=args.device,
        logger=print,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
