#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Final runnable fusion code:
PRISMA (LR 128x128x168) + Sentinel-2 (HR 384x384x9) -> HR HSI X (384x384x168)
with official S2A SRF + learnable PSF + RTM (PROSAIL surrogate) manifold prior.

You already exported:
  ./fwd_net_168.pt
  ./inv_net_168.pt
and scalers:
  ./surrogate_prisma168/scalers.pkl

Run:
  python train_physics_hsms3_rtm_final.py

"""

import os
import pickle
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib


# =========================
# Paths (EDIT IF NEEDED)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
ARTIFACTS_DIR = os.path.join(REPO_ROOT, "artifacts")
PRISMA_PATH = os.path.join(REPO_ROOT, "datasets_crop/dataset_onepatch/prisma_patches/ALL_POINTS_prisma.npy")
S2_PATH = os.path.join(REPO_ROOT, "datasets_crop/dataset_onepatch/s2_patches/ALL_POINTS_s2.npy")
WL_PATH = os.path.join(ARTIFACTS_DIR, "filtered_wavelengths.npy")
SRF_XLSX = os.path.join(ARTIFACTS_DIR, "Sentinel2SRF2024-4.0.xlsx")

# RTM surrogate artifacts
USE_RTM_PRIOR = True
INV_TORCHSCRIPT_PATH = os.path.join(ARTIFACTS_DIR, "surrogate_out_v3/inv_net_168.pt")
FWD_TORCHSCRIPT_PATH = os.path.join(ARTIFACTS_DIR, "surrogate_out_v3/fwd_net_168.pt")
SCALERS_PKL = os.path.join(ARTIFACTS_DIR, "surrogate_out_v3/scalers.pkl")

# --- ablation switches ---
RUN_TAG = "both"       # "hs_only" or "ms_only" or "both"
USE_HS_TERM = True     # PRISMA LR consistency
USE_MS_TERM = True     # Sentinel-2 HR consistency

# Outputs
OUT_X_NPY = os.path.join(REPO_ROOT, "outputs", f"X_hat_hrhsi_{RUN_TAG}.npy")
OUT_WL_SORT_IDX = os.path.join(REPO_ROOT, "outputs", "prisma_wl_sort_idx_used.npy")


# =========================
# Settings
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision("high") if hasattr(torch, "set_float32_matmul_precision") else None

ITERS = 2000
LR = 1e-2


# Loss weights
LAMBDA_SPEC = 5e-2
LAMBDA_TV   = 1e-3

# RTM prior schedule (for 2000 iters)
LAMBDA_RTM_MAX = 0.01
RTM_WARMUP = 400
RTM_RAMP   = 400
RTM_SAMPLES = 16384
NDVI_THR = 0.2

# PSF params
PSF_INIT_SIGMA = 1.2
PSF_MIN_SIGMA = 0.5
PSF_MAX_SIGMA = 3.0
PSF_KS = 11

# Sentinel-2 bands used (fixed 9-band set)
S2_BANDS = ("B2", "B3", "B4", "B5", "B6", "B7", "B8", "B11", "B12")


# =========================
# Helpers
# =========================

def spectral_lowpass(x: torch.Tensor, k: int = 7) -> torch.Tensor:
    """
    x: (N,B) or (B,)  -> returns same shape after 1D moving-average smoothing
    """
    if x.dim() == 1:
        x2 = x[None, None, :]            # (1,1,B)
    else:
        x2 = x[:, None, :]               # (N,1,B)

    pad = k // 2
    w = torch.ones(1, 1, k, device=x.device, dtype=x.dtype) / float(k)
    x2 = F.pad(x2, (pad, pad), mode="reflect")
    y = F.conv1d(x2, w)

    if x.dim() == 1:
        return y[0, 0, :]
    return y[:, 0, :]

def clip01_np(x: np.ndarray) -> np.ndarray:
    return np.clip(x.astype(np.float32), 0.0, 1.0)

def spectral_smoothness(X: torch.Tensor) -> torch.Tensor:
    # 2nd-order spectral difference
    d1 = X[:, 1:] - X[:, :-1]
    d2 = d1[:, 1:] - d1[:, :-1]
    return (d2 ** 2).mean()

def tv_loss(X: torch.Tensor) -> torch.Tensor:
    return ((X[:, :, :, 1:] - X[:, :, :, :-1]) ** 2).mean() + \
           ((X[:, :, 1:, :] - X[:, :, :-1, :]) ** 2).mean()

def psnr_from_mse(m: torch.Tensor) -> float:
    return float(10.0 * torch.log10(1.0 / (m + 1e-12)))

def sam_mean_rad(A: torch.Tensor, B: torch.Tensor) -> float:
    # A,B: (1,C,H,W)
    a = A[0].permute(1, 2, 0).reshape(-1, A.shape[1])
    b = B[0].permute(1, 2, 0).reshape(-1, B.shape[1])
    dot = (a * b).sum(dim=1)
    na = torch.sqrt((a * a).sum(dim=1) + 1e-12)
    nb = torch.sqrt((b * b).sum(dim=1) + 1e-12)
    cos = torch.clamp(dot / (na * nb), -1.0, 1.0)
    ang = torch.acos(cos)
    return float(ang.mean().item())

def lambda_rtm(it: int, warm: int, ramp: int, lam_max: float) -> float:
    if it < warm:
        return 0.0
    if it < warm + ramp:
        return lam_max * (it - warm) / float(ramp)
    return lam_max


# =========================
# SRF: Official Sentinel-2A
# =========================
def build_s2a_R_from_official_xlsx(
    wl_prisma_nm_sorted: np.ndarray,
    xlsx_path: str,
    device: str,
    sheet_name: str = "Spectral Responses (S2A)",
    bands=S2_BANDS
) -> torch.Tensor:
    """
    wl_prisma_nm_sorted: (168,) ascending
    returns R: (9,168), rows sum to 1 (after SRF*dλ integration normalization)
    """
    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(f"SRF XLSX not found: {xlsx_path}")

    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    df.columns = [c.strip() for c in df.columns]

    wl_srf = df["SR_WL"].to_numpy(np.float32)
    wl = wl_prisma_nm_sorted.astype(np.float32)

    # dλ on PRISMA grid
    dlam = np.empty_like(wl)
    dlam[1:-1] = 0.5 * (wl[2:] - wl[:-2])
    dlam[0] = wl[1] - wl[0]
    dlam[-1] = wl[-1] - wl[-2]
    dlam = np.clip(dlam, 1e-6, None)

    rows = []
    for b in bands:
        col = f"S2A_SR_AV_{b}"
        if col not in df.columns:
            raise KeyError(f"Missing SRF column in xlsx: {col}")
        srf = df[col].to_numpy(np.float32)
        srf_i = np.interp(wl, wl_srf, srf, left=0.0, right=0.0)
        r = srf_i * dlam
        s = r.sum()
        if s <= 0:
            raise RuntimeError(f"SRF integration sum <= 0 for band {b}. Check wavelength range.")
        r /= s
        rows.append(r)

    R = np.stack(rows, axis=0)  # (9,168)
    return torch.from_numpy(R).float().to(device)

def srf_mix(X: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    # X: (1,168,H,W), R: (9,168) -> (1,9,H,W)
    return torch.einsum("bchw,lc->blhw", X, R)


# =========================
# Learnable PSF
# =========================
class LearnableGaussianPSF(nn.Module):
    def __init__(self, init_sigma=1.2, min_sigma=0.5, max_sigma=3.0, ks=11):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(float(init_sigma))))
        self.min_sigma = float(min_sigma)
        self.max_sigma = float(max_sigma)
        self.ks = int(ks)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        sigma = torch.clamp(torch.exp(self.log_sigma), self.min_sigma, self.max_sigma)
        k = self._kernel(sigma, X.device)  # (1,1,ks,ks)
        C = X.shape[1]
        k = k.repeat(C, 1, 1, 1)           # (C,1,ks,ks)
        return F.conv2d(X, k, padding=self.ks // 2, groups=C)

    def _kernel(self, sigma: torch.Tensor, device: torch.device) -> torch.Tensor:
        ax = torch.arange(self.ks, device=device) - self.ks // 2
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        k = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
        k = k / k.sum()
        return k[None, None]


# =========================
# MinMaxScaler (torch)
# =========================
def load_minmax_scalers_to_torch(pkl_path: str, device: str):
    """
    Robust loader for scalers.pkl saved by different methods:
      - pickle.dump
      - joblib.dump (possibly compressed)
      - torch.save
      - gzip-compressed pickle

    Returns torch tensors:
      spec_min (168,), spec_scale (168,), par_min (19,), par_scale (19,)
    """
    import os
    import pickle

    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"scalers file not found: {pkl_path}")

    # Read a few bytes to detect format
    with open(pkl_path, "rb") as f:
        head = f.read(4)

    obj = None
    last_err = None

    # 1) Try torch.load (works if saved by torch.save)
    try:
        obj = torch.load(pkl_path, map_location="cpu")
    except Exception as e:
        last_err = e

    # 2) Try joblib.load (works if saved by joblib.dump, incl. compressed)
    if obj is None:
        try:

            obj = joblib.load(pkl_path)
        except Exception as e:
            last_err = e

    # 3) Try pickle.load directly (standard pickle)
    if obj is None:
        try:
            with open(pkl_path, "rb") as f:
                obj = pickle.load(f)
        except Exception as e:
            last_err = e

    # 4) Try gzip+pickle (if gzipped)
    if obj is None:
        try:
            import gzip
            with gzip.open(pkl_path, "rb") as f:
                obj = pickle.load(f)
        except Exception as e:
            last_err = e

    if obj is None:
        raise RuntimeError(
            f"Failed to load scalers from {pkl_path}. "
            f"Header bytes={head!r}. Last error={last_err}"
        )

    # Expect dict with keys
    # {"spec_scaler": MinMaxScaler, "param_scaler": MinMaxScaler}
    if isinstance(obj, dict) and ("spec_scaler" in obj) and ("param_scaler" in obj):
        spec = obj["spec_scaler"]
        param = obj["param_scaler"]
    else:
        # Sometimes people save tuple/list
        # Try to infer
        if isinstance(obj, (list, tuple)) and len(obj) == 2:
            spec, param = obj
        else:
            raise ValueError(f"Unknown scalers object structure: type={type(obj)} keys={getattr(obj,'keys',lambda:[])()}")

    # Pull out min_ and scale_ from sklearn MinMaxScaler
    # (these attributes exist after fitting)
    spec_min = torch.tensor(spec.min_, dtype=torch.float32, device=device)
    spec_scale = torch.tensor(spec.scale_, dtype=torch.float32, device=device)
    par_min = torch.tensor(param.min_, dtype=torch.float32, device=device)
    par_scale = torch.tensor(param.scale_, dtype=torch.float32, device=device)

    return spec_min, spec_scale, par_min, par_scale

def minmax_transform(x: torch.Tensor, x_min: torch.Tensor, x_scale: torch.Tensor) -> torch.Tensor:
    # x: (...,D), x_min/x_scale: (D,)
    return (x - x_min) * x_scale

def minmax_inverse_transform(x_scaled: torch.Tensor, x_min: torch.Tensor, x_scale: torch.Tensor) -> torch.Tensor:
    return x_scaled / (x_scale + 1e-12) + x_min


# =========================
# Main
# =========================
def main(
    *,
    prisma_path: str = PRISMA_PATH,
    s2_path: str = S2_PATH,
    wl_path: str = WL_PATH,
    srf_xlsx: str = SRF_XLSX,
    inv_torchscript_path: str = INV_TORCHSCRIPT_PATH,
    fwd_torchscript_path: str = FWD_TORCHSCRIPT_PATH,
    scalers_pkl: str = SCALERS_PKL,
    run_tag: str = RUN_TAG,
    out_x_npy: str | None = None,
    out_wl_sort_idx: str | None = None,
    use_rtm_prior: bool = USE_RTM_PRIOR,
    use_hs_term: bool | None = None,
    use_ms_term: bool | None = None,
    iters: int = ITERS,
    lr: float = LR,
):
    print("DEVICE:", DEVICE)
    torch.manual_seed(0)
    np.random.seed(0)
    if run_tag not in {"hs_only", "ms_only", "both"}:
        raise ValueError(f"Unsupported run_tag: {run_tag}")
    if use_hs_term is None:
        use_hs_term = run_tag in {"hs_only", "both"}
    if use_ms_term is None:
        use_ms_term = run_tag in {"ms_only", "both"}
    out_x_npy = out_x_npy or f"./X_hat_hrhsi_{run_tag}.npy"
    out_wl_sort_idx = out_wl_sort_idx or "./prisma_wl_sort_idx_used.npy"
    os.makedirs(os.path.dirname(os.path.abspath(out_x_npy)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(out_wl_sort_idx)), exist_ok=True)

    # ---- Load reflectance ----
    Y_hs = clip01_np(np.load(prisma_path))
    Y_ms = clip01_np(np.load(s2_path))
    wl   = np.load(wl_path).astype(np.float32)

    # handle channel-first
    if Y_hs.ndim != 3:
        raise ValueError(f"PRISMA array must be 3D, got {Y_hs.shape}")
    if Y_ms.ndim != 3:
        raise ValueError(f"S2 array must be 3D, got {Y_ms.shape}")

    hs_channels = int(wl.shape[0])

    if Y_hs.shape[0] == hs_channels:
        Y_hs = np.transpose(Y_hs, (1, 2, 0))
    if Y_ms.shape[0] == 9:    # (9,384,384)
        Y_ms = np.transpose(Y_ms, (1, 2, 0))

    if Y_hs.shape[-1] != hs_channels:
        raise ValueError(f"HS band count {Y_hs.shape[-1]} does not match wavelength count {hs_channels}")

    # ---- Sort wl & reorder HS bands accordingly ----
    idx = np.argsort(wl)
    wl_sorted = wl[idx]
    Y_hs_sorted = Y_hs[:, :, idx]
    np.save(out_wl_sort_idx, idx)
    print("Saved wl sort idx:", out_wl_sort_idx)

    # ---- Torch tensors ----
    Y_hs_t = torch.from_numpy(Y_hs_sorted).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    Y_ms_t = torch.from_numpy(Y_ms).permute(2, 0, 1).unsqueeze(0).to(DEVICE)         # (1,9,384,384)

    # ---- Init X (HR HSI) ----
    X_init = F.interpolate(Y_hs_t, scale_factor=3, mode="bicubic", align_corners=False)
    X = nn.Parameter(X_init.clamp(0.0, 1.0))

    # ---- Operators ----
    R0 = build_s2a_R_from_official_xlsx(wl_sorted, srf_xlsx, DEVICE)  # (9,168)
    psf = LearnableGaussianPSF(
        init_sigma=PSF_INIT_SIGMA,
        min_sigma=PSF_MIN_SIGMA,
        max_sigma=PSF_MAX_SIGMA,
        ks=PSF_KS
    ).to(DEVICE)

    # ---- Vegetation mask (NDVI) from S2 HR ----
    # S2 band order: B2,B3,B4,B5,B6,B7,B8,B11,B12
    eps = 1e-6
    B4 = Y_ms_t[:, 2:3, :, :]
    B8 = Y_ms_t[:, 6:7, :, :]
    ndvi = (B8 - B4) / (B8 + B4 + eps)
    veg_mask = (ndvi > NDVI_THR).float()  # (1,1,384,384)
    veg_ratio = float((veg_mask > 0.5).float().mean().item())
    print(f"Vegetation ratio NDVI>{NDVI_THR}: {veg_ratio:.3f}")

    # ---- RTM surrogate + scalers ----
    inv_net = fwd_net = None
    spec_min = spec_scale = par_min = par_scale = None
    if use_rtm_prior:
        if not (os.path.exists(inv_torchscript_path) and os.path.exists(fwd_torchscript_path)):
            raise FileNotFoundError("TorchScript model not found. Check paths:\n"
                                    f"  {inv_torchscript_path}\n  {fwd_torchscript_path}")
        inv_net = torch.jit.load(inv_torchscript_path, map_location=DEVICE).eval()
        fwd_net = torch.jit.load(fwd_torchscript_path, map_location=DEVICE).eval()
        for p in inv_net.parameters():
            p.requires_grad_(False)
        for p in fwd_net.parameters():
            p.requires_grad_(False)

        spec_min, spec_scale, par_min, par_scale = load_minmax_scalers_to_torch(scalers_pkl, DEVICE)
        if spec_min.numel() != hs_channels or spec_scale.numel() != hs_channels:
            raise ValueError(f"Spec scaler dims mismatch: {spec_min.numel()}, {spec_scale.numel()}")
        print("Loaded RTM surrogate + scalers.")

    # ---- Optimizer ----
    opt = torch.optim.Adam([X] + list(psf.parameters()), lr=lr)

    # ---- Train loop ----
    for it in range(int(iters) + 1):
        opt.zero_grad(set_to_none=True)

        # degradation consistency
        X_blur = psf(X)
        Y_hs_hat = F.avg_pool2d(X_blur, kernel_size=3, stride=3)   # (1,168,128,128)
        Y_ms_hat = srf_mix(X, R0)                                  # (1,9,384,384)

        # data terms
        L_hs = F.mse_loss(Y_hs_hat, Y_hs_t) if use_hs_term else torch.tensor(0.0, device=DEVICE)
        L_ms = F.mse_loss(Y_ms_hat, Y_ms_t) if use_ms_term else torch.tensor(0.0, device=DEVICE)


        # regularizers
        L_reg = LAMBDA_SPEC * spectral_smoothness(X) + LAMBDA_TV * tv_loss(X)

        # RTM manifold prior (sampled)
        L_rtm = torch.tensor(0.0, device=DEVICE)
        lam_rtm = 0.0
        if use_rtm_prior:
            lam_rtm = lambda_rtm(it, RTM_WARMUP, RTM_RAMP, LAMBDA_RTM_MAX)
            if lam_rtm > 0:
                # sample vegetation pixels
                mask_flat = veg_mask.view(-1)  # (H*W,)
                veg_idx = torch.nonzero(mask_flat > 0.5, as_tuple=False).view(-1)
                if veg_idx.numel() > 0:
                    n = min(RTM_SAMPLES, int(veg_idx.numel()))
                    pick = veg_idx[torch.randperm(veg_idx.numel(), device=DEVICE)[:n]]

                    # gather sampled spectra (reflectance domain)
                    X_flat = X[0].permute(1, 2, 0).reshape(-1, hs_channels)
                    X_s = X_flat[pick]

                    # scaler-domain projection: X -> scaled -> inv -> fwd -> scaled -> inverse -> reflectance
                    X_s_scaled = minmax_transform(X_s, spec_min, spec_scale)
                    P_scaled = inv_net(X_s_scaled)
                    X_proj_scaled = fwd_net(P_scaled)
                    X_proj = minmax_inverse_transform(X_proj_scaled, spec_min, spec_scale)

                    X_s_lp    = spectral_lowpass(X_s, k=7)
                    X_proj_lp = spectral_lowpass(X_proj, k=7)
                    L_rtm = F.mse_loss(X_s_lp, X_proj_lp)

        loss = L_hs + L_ms + L_reg + (lam_rtm * L_rtm)
        loss.backward()
        opt.step()

        with torch.no_grad():
            X.clamp_(0.0, 1.0)

        # logging
        if it % 100 == 0:
            sigma_val = float(torch.exp(psf.log_sigma).detach().cpu())
            msg = (
                f"[{it:04d}] loss={loss.item():.6f} "
                f"L_hs={L_hs.item():.6f} L_ms={L_ms.item():.6f} L_reg={L_reg.item():.6f} "
                f"psf_sigma={sigma_val:.3f}"
            )
            if use_rtm_prior:
                msg += f" lam_rtm={lam_rtm:.4f} L_rtm={float(L_rtm.item()):.6f}"
            print(msg)

    # ---- Save X ----
    X_out = X.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    np.save(out_x_npy, X_out)
    print("Saved X:", out_x_npy, X_out.shape)

    # ---- No-GT evaluation: consistency ----
    with torch.no_grad():
        X_hat = X
        X_blur = psf(X_hat)
        Y_hs_hat = F.avg_pool2d(X_blur, kernel_size=3, stride=3)
        Y_ms_hat = srf_mix(X_hat, R0)

        hs_mse = F.mse_loss(Y_hs_hat, Y_hs_t)
        ms_mse = F.mse_loss(Y_ms_hat, Y_ms_t)

        print("\n===== No-GT consistency =====")
        print(f"HS: MSE={hs_mse.item():.6e}, PSNR={psnr_from_mse(hs_mse):.2f} dB, SAM={sam_mean_rad(Y_hs_hat, Y_hs_t):.4f} rad")
        print(f"MS: MSE={ms_mse.item():.6e}, PSNR={psnr_from_mse(ms_mse):.2f} dB, SAM={sam_mean_rad(Y_ms_hat, Y_ms_t):.4f} rad")
        print(f"Spectral smoothness (2nd diff): {float(spectral_smoothness(X_hat).item()):.6e}")
        print(f"Vegetation ratio NDVI>{NDVI_THR}: {veg_ratio:.3f}")

    summary = {
        "run_tag": run_tag,
        "prisma_path": os.path.abspath(prisma_path),
        "s2_path": os.path.abspath(s2_path),
        "wl_path": os.path.abspath(wl_path),
        "srf_xlsx": os.path.abspath(srf_xlsx),
        "inv_torchscript_path": os.path.abspath(inv_torchscript_path),
        "fwd_torchscript_path": os.path.abspath(fwd_torchscript_path),
        "scalers_pkl": os.path.abspath(scalers_pkl),
        "out_x_npy": os.path.abspath(out_x_npy),
        "out_wl_sort_idx": os.path.abspath(out_wl_sort_idx),
        "use_rtm_prior": bool(use_rtm_prior),
        "use_hs_term": bool(use_hs_term),
        "use_ms_term": bool(use_ms_term),
        "iters": int(iters),
        "lr": float(lr),
        "hs_mse": float(hs_mse.item()),
        "ms_mse": float(ms_mse.item()),
        "hs_psnr_db": float(psnr_from_mse(hs_mse)),
        "ms_psnr_db": float(psnr_from_mse(ms_mse)),
        "hs_sam_rad": float(sam_mean_rad(Y_hs_hat, Y_hs_t)),
        "ms_sam_rad": float(sam_mean_rad(Y_ms_hat, Y_ms_t)),
        "spectral_smoothness": float(spectral_smoothness(X_hat).item()),
        "veg_ratio": float(veg_ratio),
        "device": DEVICE,
        "n_hs_bands": int(hs_channels),
    }
    summary_path = os.path.join(os.path.dirname(os.path.abspath(out_x_npy)), "train_RTM_prior_only_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print("Saved summary:", summary_path)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RTM-prior-only HR-HSI reconstruction with configurable inputs.")
    parser.add_argument("--prisma_path", default=PRISMA_PATH)
    parser.add_argument("--s2_path", default=S2_PATH)
    parser.add_argument("--wl_path", default=WL_PATH)
    parser.add_argument("--srf_xlsx", default=SRF_XLSX)
    parser.add_argument("--inv_torchscript_path", default=INV_TORCHSCRIPT_PATH)
    parser.add_argument("--fwd_torchscript_path", default=FWD_TORCHSCRIPT_PATH)
    parser.add_argument("--scalers_pkl", default=SCALERS_PKL)
    parser.add_argument("--run_tag", choices=["hs_only", "ms_only", "both"], default=RUN_TAG)
    parser.add_argument("--out_x_npy", default=None)
    parser.add_argument("--out_wl_sort_idx", default=None)
    parser.add_argument("--iters", type=int, default=ITERS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--use_rtm_prior", type=int, choices=[0, 1], default=1 if USE_RTM_PRIOR else 0)
    parser.add_argument("--use_hs_term", type=int, choices=[0, 1], default=None)
    parser.add_argument("--use_ms_term", type=int, choices=[0, 1], default=None)
    args = parser.parse_args()
    main(
        prisma_path=args.prisma_path,
        s2_path=args.s2_path,
        wl_path=args.wl_path,
        srf_xlsx=args.srf_xlsx,
        inv_torchscript_path=args.inv_torchscript_path,
        fwd_torchscript_path=args.fwd_torchscript_path,
        scalers_pkl=args.scalers_pkl,
        run_tag=args.run_tag,
        out_x_npy=args.out_x_npy,
        out_wl_sort_idx=args.out_wl_sort_idx,
        use_rtm_prior=bool(args.use_rtm_prior),
        use_hs_term=(None if args.use_hs_term is None else bool(args.use_hs_term)),
        use_ms_term=(None if args.use_ms_term is None else bool(args.use_ms_term)),
        iters=args.iters,
        lr=args.lr,
    )
