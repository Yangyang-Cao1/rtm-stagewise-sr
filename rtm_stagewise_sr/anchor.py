#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle

import joblib
import torch


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)
DEFAULT_INV_PATH = os.path.join(REPO_ROOT, "artifacts/surrogate_out_v3/inv_net_168.pt")
DEFAULT_FWD_PATH = os.path.join(REPO_ROOT, "artifacts/surrogate_out_v3/fwd_net_168.pt")
DEFAULT_SCALERS_PATH = os.path.join(REPO_ROOT, "artifacts/surrogate_out_v3/scalers.pkl")


def load_minmax_scalers_to_torch(pkl_path: str, device: str):
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"scalers file not found: {pkl_path}")

    obj = None
    loaders = [
        lambda: torch.load(pkl_path, map_location="cpu"),
        lambda: joblib.load(pkl_path),
        lambda: pickle.load(open(pkl_path, "rb")),
    ]
    last_error = None
    for loader in loaders:
        try:
            obj = loader()
            break
        except Exception as exc:
            last_error = exc
    if obj is None:
        raise RuntimeError(f"Failed to load scalers from {pkl_path}: {last_error}")

    if isinstance(obj, dict):
        spec_scaler = obj.get("spec_scaler") or obj.get("spec") or obj.get("x_scaler")
        param_scaler = obj.get("param_scaler") or obj.get("param") or obj.get("y_scaler")
    elif isinstance(obj, (list, tuple)) and len(obj) == 2:
        spec_scaler, param_scaler = obj
    else:
        raise ValueError(f"Unknown scalers object structure: {type(obj)}")

    spec_min = torch.tensor(spec_scaler.min_, dtype=torch.float32, device=device)
    spec_scale = torch.tensor(spec_scaler.scale_, dtype=torch.float32, device=device)
    param_min = torch.tensor(param_scaler.min_, dtype=torch.float32, device=device)
    param_scale = torch.tensor(param_scaler.scale_, dtype=torch.float32, device=device)
    return spec_min, spec_scale, param_min, param_scale


def minmax_transform(x: torch.Tensor, x_min: torch.Tensor, x_scale: torch.Tensor) -> torch.Tensor:
    return (x - x_min) * x_scale


def minmax_inverse_transform(x_scaled: torch.Tensor, x_min: torch.Tensor, x_scale: torch.Tensor) -> torch.Tensor:
    return x_scaled / (x_scale + 1e-12) + x_min


class RTMAnchorProjector:
    def __init__(
        self,
        device: str,
        inv_path: str = DEFAULT_INV_PATH,
        fwd_path: str = DEFAULT_FWD_PATH,
        scalers_path: str = DEFAULT_SCALERS_PATH,
        chunk_size: int = 512,
    ):
        if not os.path.exists(inv_path):
            raise FileNotFoundError(f"Missing inverse RTM surrogate: {inv_path}")
        if not os.path.exists(fwd_path):
            raise FileNotFoundError(f"Missing forward RTM surrogate: {fwd_path}")
        self.device = device
        self.chunk_size = int(chunk_size)
        self.inv_net = torch.jit.load(inv_path, map_location=device).eval()
        self.fwd_net = torch.jit.load(fwd_path, map_location=device).eval()
        for parameter in self.inv_net.parameters():
            parameter.requires_grad_(False)
        for parameter in self.fwd_net.parameters():
            parameter.requires_grad_(False)
        self.spec_min, self.spec_scale, self.param_min, self.param_scale = load_minmax_scalers_to_torch(
            scalers_path, device=device
        )

    def project(self, hsi_bchw: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = hsi_bchw.shape
        hsi_flat = hsi_bchw.permute(0, 2, 3, 1).reshape(-1, channels)
        output_flat = torch.empty_like(hsi_flat)
        for start in range(0, hsi_flat.shape[0], self.chunk_size):
            end = min(start + self.chunk_size, hsi_flat.shape[0])
            chunk = hsi_flat[start:end]
            chunk_scaled = minmax_transform(chunk, self.spec_min, self.spec_scale)
            params = self.inv_net(chunk_scaled)
            pred_scaled = self.fwd_net(params)
            output_flat[start:end] = minmax_inverse_transform(pred_scaled, self.spec_min, self.spec_scale)
        return output_flat.reshape(batch, height, width, channels).permute(0, 3, 1, 2).detach()


def amplitude_anchor_loss(hsi_bchw: torch.Tensor, anchor_bchw: torch.Tensor) -> torch.Tensor:
    return torch.mean((hsi_bchw - anchor_bchw) ** 2)
