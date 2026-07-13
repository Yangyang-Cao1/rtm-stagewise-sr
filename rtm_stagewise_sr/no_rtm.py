#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .data import DEFAULT_PRISMA_PATH, DEFAULT_S2_PATH, DEFAULT_WL_PATH, load_core_data
from .losses import LossWeights, build_total_loss, psnr_from_mse, sam_mean_rad, second_order_energy
from .operators import FixedGaussianDownsample, bicubic_initialize_from_hs, build_gaussian_s2_response, mix_to_multispectral
from .utils import append_history_csv, create_run_dir, save_json, save_sort_index, select_device


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RUNS_DIR = os.path.join(os.path.dirname(BASE_DIR), "outputs/no_rtm")


def run_training(
    *,
    prisma_path: str = DEFAULT_PRISMA_PATH,
    s2_path: str = DEFAULT_S2_PATH,
    wavelengths_path: str = DEFAULT_WL_PATH,
    output_dir: str | None = None,
    run_label: str = "no_rtm_clean",
    iters: int = 2000,
    lr: float = 1e-2,
    lambda_hs: float = 1.0,
    lambda_ms: float = 1.0,
    lambda_spec: float = 1e-2,
    lambda_tv: float = 1e-3,
    scale_factor: int = 3,
    psf_kernel_size: int = 11,
    psf_sigma: float = 1.2,
    log_every: int = 100,
    device: str | None = None,
) -> dict[str, Any]:
    run_dir = output_dir or create_run_dir(DEFAULT_RUNS_DIR, run_label)
    os.makedirs(run_dir, exist_ok=True)
    device_name = select_device(device)

    data = load_core_data(
        prisma_path=prisma_path,
        s2_path=s2_path,
        wavelength_path=wavelengths_path,
        normalize=True,
    )
    save_sort_index(os.path.join(run_dir, "sort_idx.npy"), data.sort_idx)

    hs_target = torch.from_numpy(data.hs_hwc_sorted).permute(2, 0, 1).unsqueeze(0).to(device_name)
    ms_target = torch.from_numpy(data.ms_hwc).permute(2, 0, 1).unsqueeze(0).to(device_name)

    hsi_init = bicubic_initialize_from_hs(hs_target, scale_factor=scale_factor)
    hsi_param = nn.Parameter(hsi_init.clone())

    response = build_gaussian_s2_response(data.wavelengths_sorted_nm, device=device_name)
    downsample = FixedGaussianDownsample(
        channels=int(hs_target.shape[1]),
        scale=scale_factor,
        kernel_size=psf_kernel_size,
        sigma=psf_sigma,
    ).to(device_name)
    optimizer = torch.optim.Adam([hsi_param], lr=lr)
    weights = LossWeights(
        lambda_hs=lambda_hs,
        lambda_ms=lambda_ms,
        lambda_spec=lambda_spec,
        lambda_tv=lambda_tv,
    )

    history: list[dict[str, Any]] = []
    for step in range(int(iters) + 1):
        optimizer.zero_grad()
        hs_pred = downsample(hsi_param)
        ms_pred = mix_to_multispectral(hsi_param, response)
        total_loss, pieces = build_total_loss(
            hsi_hr_bchw=hsi_param,
            hs_pred_bchw=hs_pred,
            hs_target_bchw=hs_target,
            ms_pred_bchw=ms_pred,
            ms_target_bchw=ms_target,
            weights=weights,
        )
        total_loss.backward()
        optimizer.step()
        with torch.no_grad():
            hsi_param.clamp_(0.0, 1.0)

        row = {
            "iter": int(step),
            "loss_total": float(total_loss.item()),
            "loss_hs": float(pieces["loss_hs"].item()),
            "loss_ms": float(pieces["loss_ms"].item()),
            "loss_spec": float(pieces["loss_spec"].item()),
            "loss_tv": float(pieces["loss_tv"].item()),
        }
        history.append(row)
        if step % int(log_every) == 0 or step == int(iters):
            print(
                f"[{step:04d}] loss={row['loss_total']:.6f} "
                f"L_hs={row['loss_hs']:.6f} L_ms={row['loss_ms']:.6f} "
                f"L_spec={row['loss_spec']:.6f} L_tv={row['loss_tv']:.6f}"
            )

    pred_hwc = hsi_param.detach().cpu().squeeze(0).permute(1, 2, 0).numpy().astype(np.float32)
    pred_path = os.path.join(run_dir, "X_hat_hrhsi_no_rtm.npy")
    np.save(pred_path, pred_hwc)

    with torch.no_grad():
        hs_pred = downsample(hsi_param)
        ms_pred = mix_to_multispectral(hsi_param, response)
        hs_mse = torch.mean((hs_pred - hs_target) ** 2)
        ms_mse = torch.mean((ms_pred - ms_target) ** 2)
        metrics = {
            "hs_mse": float(hs_mse.item()),
            "ms_mse": float(ms_mse.item()),
            "hs_psnr_db": float(psnr_from_mse(hs_mse)),
            "ms_psnr_db": float(psnr_from_mse(ms_mse)),
            "hs_sam_rad": float(sam_mean_rad(hs_pred, hs_target)),
            "ms_sam_rad": float(sam_mean_rad(ms_pred, ms_target)),
            "spectral_2nd_diff_energy": float(second_order_energy(hsi_param).item()),
            "output_path": pred_path,
            "sort_idx_path": os.path.join(run_dir, "sort_idx.npy"),
        }

    config = {
        "prisma_path": os.path.abspath(prisma_path),
        "s2_path": os.path.abspath(s2_path),
        "wavelengths_path": os.path.abspath(wavelengths_path),
        "output_dir": os.path.abspath(run_dir),
        "device": device_name,
        "iters": int(iters),
        "lr": float(lr),
        "scale_factor": int(scale_factor),
        "psf_kernel_size": int(psf_kernel_size),
        "psf_sigma": float(psf_sigma),
        "loss_weights": {
            "lambda_hs": float(lambda_hs),
            "lambda_ms": float(lambda_ms),
            "lambda_spec": float(lambda_spec),
            "lambda_tv": float(lambda_tv),
        },
        "band_sorting": {
            "enabled": True,
            "sort_idx_path": os.path.join(run_dir, "sort_idx.npy"),
        },
        "metadata": data.metadata,
    }
    save_json(os.path.join(run_dir, "config.json"), config)
    save_json(os.path.join(run_dir, "metrics.json"), metrics)
    append_history_csv(os.path.join(run_dir, "train_history.csv"), history)

    return {
        "run_dir": run_dir,
        "prediction_path": pred_path,
        "metrics_path": os.path.join(run_dir, "metrics.json"),
        "config_path": os.path.join(run_dir, "config.json"),
        "sort_idx_path": os.path.join(run_dir, "sort_idx.npy"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean no-RTM HSI super-resolution trainer.")
    parser.add_argument("--prisma_path", default=DEFAULT_PRISMA_PATH)
    parser.add_argument("--s2_path", default=DEFAULT_S2_PATH)
    parser.add_argument("--wavelengths_path", default=DEFAULT_WL_PATH)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--run_label", default="no_rtm_clean")
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--lambda_hs", type=float, default=1.0)
    parser.add_argument("--lambda_ms", type=float, default=1.0)
    parser.add_argument("--lambda_spec", type=float, default=1e-2)
    parser.add_argument("--lambda_tv", type=float, default=1e-3)
    parser.add_argument("--scale_factor", type=int, default=3)
    parser.add_argument("--psf_kernel_size", type=int, default=11)
    parser.add_argument("--psf_sigma", type=float, default=1.2)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    result = run_training(
        prisma_path=args.prisma_path,
        s2_path=args.s2_path,
        wavelengths_path=args.wavelengths_path,
        output_dir=args.output_dir,
        run_label=args.run_label,
        iters=args.iters,
        lr=args.lr,
        lambda_hs=args.lambda_hs,
        lambda_ms=args.lambda_ms,
        lambda_spec=args.lambda_spec,
        lambda_tv=args.lambda_tv,
        scale_factor=args.scale_factor,
        psf_kernel_size=args.psf_kernel_size,
        psf_sigma=args.psf_sigma,
        log_every=args.log_every,
        device=args.device,
    )
    print(result)


if __name__ == "__main__":
    main()
