#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def hs_consistency_loss(pred_hs: torch.Tensor, target_hs: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred_hs, target_hs)


def ms_consistency_loss(pred_ms: torch.Tensor, target_ms: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred_ms, target_ms)


def spectral_smoothness_loss(hsi_bchw: torch.Tensor) -> torch.Tensor:
    first_diff = hsi_bchw[:, 1:] - hsi_bchw[:, :-1]
    second_diff = first_diff[:, 1:] - first_diff[:, :-1]
    return (second_diff ** 2).mean()


def spatial_tv_loss(hsi_bchw: torch.Tensor) -> torch.Tensor:
    loss_x = ((hsi_bchw[:, :, :, 1:] - hsi_bchw[:, :, :, :-1]) ** 2).mean()
    loss_y = ((hsi_bchw[:, :, 1:, :] - hsi_bchw[:, :, :-1, :]) ** 2).mean()
    return loss_x + loss_y


def spatial_gradient_consistency_loss(hsi_bchw: torch.Tensor, ref_bchw: torch.Tensor) -> torch.Tensor:
    grad_x = hsi_bchw[:, :, :, 1:] - hsi_bchw[:, :, :, :-1]
    ref_grad_x = ref_bchw[:, :, :, 1:] - ref_bchw[:, :, :, :-1]
    grad_y = hsi_bchw[:, :, 1:, :] - hsi_bchw[:, :, :-1, :]
    ref_grad_y = ref_bchw[:, :, 1:, :] - ref_bchw[:, :, :-1, :]
    return ((grad_x - ref_grad_x) ** 2).mean() + ((grad_y - ref_grad_y) ** 2).mean()


def second_order_energy(hsi_bchw: torch.Tensor) -> torch.Tensor:
    return spectral_smoothness_loss(hsi_bchw)


def psnr_from_mse(mse_value: torch.Tensor) -> float:
    return float(10.0 * torch.log10(1.0 / (mse_value + 1e-12)))


def sam_mean_rad(a_bchw: torch.Tensor, b_bchw: torch.Tensor) -> float:
    a_flat = a_bchw[0].permute(1, 2, 0).reshape(-1, a_bchw.shape[1])
    b_flat = b_bchw[0].permute(1, 2, 0).reshape(-1, b_bchw.shape[1])
    dot = (a_flat * b_flat).sum(dim=1)
    norm_a = torch.sqrt((a_flat * a_flat).sum(dim=1) + 1e-12)
    norm_b = torch.sqrt((b_flat * b_flat).sum(dim=1) + 1e-12)
    cos = torch.clamp(dot / (norm_a * norm_b + 1e-12), -1.0, 1.0)
    return float(torch.acos(cos).mean().item())


@dataclass
class LossWeights:
    lambda_hs: float = 1.0
    lambda_ms: float = 1.0
    lambda_spec: float = 1e-2
    lambda_tv: float = 1e-3


def build_total_loss(
    hsi_hr_bchw: torch.Tensor,
    hs_pred_bchw: torch.Tensor,
    hs_target_bchw: torch.Tensor,
    ms_pred_bchw: torch.Tensor,
    ms_target_bchw: torch.Tensor,
    weights: LossWeights,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss_hs = hs_consistency_loss(hs_pred_bchw, hs_target_bchw)
    loss_ms = ms_consistency_loss(ms_pred_bchw, ms_target_bchw)
    loss_spec = spectral_smoothness_loss(hsi_hr_bchw)
    loss_tv = spatial_tv_loss(hsi_hr_bchw)
    total = (
        float(weights.lambda_hs) * loss_hs
        + float(weights.lambda_ms) * loss_ms
        + float(weights.lambda_spec) * loss_spec
        + float(weights.lambda_tv) * loss_tv
    )
    return total, {
        "loss_hs": loss_hs,
        "loss_ms": loss_ms,
        "loss_spec": loss_spec,
        "loss_tv": loss_tv,
    }
