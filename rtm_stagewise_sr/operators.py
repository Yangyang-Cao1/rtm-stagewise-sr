#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


S2_CENTERS_NM = [496.6, 560.0, 664.5, 703.9, 740.2, 782.5, 835.1, 1613.7, 2202.4]
S2_SIGMAS_NM = [65.0, 35.0, 30.0, 15.0, 15.0, 20.0, 115.0, 90.0, 180.0]


def build_gaussian_s2_response(wavelengths_nm, device: torch.device | str) -> torch.Tensor:
    wl = torch.as_tensor(wavelengths_nm, dtype=torch.float32, device=device)
    centers = torch.tensor(S2_CENTERS_NM, dtype=torch.float32, device=device)
    sigmas = torch.tensor(S2_SIGMAS_NM, dtype=torch.float32, device=device)
    diff = wl[None, :] - centers[:, None]
    response = torch.exp(-(diff ** 2) / (2.0 * (sigmas[:, None] ** 2)))
    response = response / (response.sum(dim=1, keepdim=True) + 1e-12)
    return response


def mix_to_multispectral(hsi_bchw: torch.Tensor, response_lc: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bchw,lc->blhw", hsi_bchw, response_lc)


def bicubic_initialize_from_hs(hs_bchw: torch.Tensor, scale_factor: int = 3) -> torch.Tensor:
    return F.interpolate(hs_bchw, scale_factor=scale_factor, mode="bicubic", align_corners=False)


def gaussian_kernel2d(kernel_size: int, sigma: float, device: torch.device | str) -> torch.Tensor:
    axis = torch.arange(kernel_size, device=device, dtype=torch.float32) - (kernel_size - 1) / 2.0
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2.0 * float(sigma) ** 2))
    kernel = kernel / (kernel.sum() + 1e-12)
    return kernel


class FixedGaussianDownsample(nn.Module):
    def __init__(self, channels: int, scale: int = 3, kernel_size: int = 11, sigma: float = 1.2):
        super().__init__()
        self.channels = int(channels)
        self.scale = int(scale)
        self.kernel_size = int(kernel_size)
        self.sigma = float(sigma)

    def forward(self, hsi_bchw: torch.Tensor) -> torch.Tensor:
        if hsi_bchw.shape[1] != self.channels:
            raise ValueError(
                f"Expected channels={self.channels}, got shape={tuple(hsi_bchw.shape)}"
            )
        kernel = gaussian_kernel2d(self.kernel_size, self.sigma, hsi_bchw.device)
        kernel = kernel.view(1, 1, self.kernel_size, self.kernel_size).repeat(self.channels, 1, 1, 1)
        blurred = F.conv2d(hsi_bchw, kernel, padding=self.kernel_size // 2, groups=self.channels)
        return F.avg_pool2d(blurred, kernel_size=self.scale, stride=self.scale)


class FixedGaussianBlur(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 9, sigma: float = 2.0):
        super().__init__()
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.sigma = float(sigma)

    def forward(self, hsi_bchw: torch.Tensor) -> torch.Tensor:
        if hsi_bchw.shape[1] != self.channels:
            raise ValueError(
                f"Expected channels={self.channels}, got shape={tuple(hsi_bchw.shape)}"
            )
        kernel = gaussian_kernel2d(self.kernel_size, self.sigma, hsi_bchw.device)
        kernel = kernel.view(1, 1, self.kernel_size, self.kernel_size).repeat(self.channels, 1, 1, 1)
        return F.conv2d(hsi_bchw, kernel, padding=self.kernel_size // 2, groups=self.channels)
