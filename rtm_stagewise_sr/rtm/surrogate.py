#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train differentiable PROSAIL surrogates (forward: params->spectrum, inverse: spectrum->params)
with stronger spectral priors and modern training tricks, to plug into ISPDiff.

Key upgrades vs v1:
- Forward net: lightweight Transformer encoder for long-range spectral dependencies
- Inverse net: Conv1D encoder + self-attention head
- Loss: MSE + SAM + Cycle (L1 + SAM) + Spectral Smoothness + optional FFT magnitude loss
- Cosine LR schedule, grad clipping, LayerNorms, mixed precision
- Extra metrics: MAE, R2, Spearman corr (per batch estimate)

Usage (same I/O as before):
  python train_surrogate_v2.py \
    --params_csv InputPROSAIL_params.csv \
    --spectra_csv PROSAIL_reflectance.csv \
    --srf_csv SENSOR_SRF.csv \
    --hs_channels 168 \
    --out_dir ./surrogate_out_v2
"""

import argparse
import json
import os
import random
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

# ------------------------------
# Utilities
# ------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def read_params_and_spectra(params_csv: str, spectra_csv: str):
    params = pd.read_csv(params_csv)
    spectra = pd.read_csv(spectra_csv)
    if len(params) != len(spectra):
        raise ValueError(f"Row mismatch: {len(params)} params vs {len(spectra)} spectra")
    X_params = params.values.astype(np.float32)
    Y_spec = spectra.values.astype(np.float32)
    return params.columns.tolist(), X_params, spectra.columns.tolist(), Y_spec


def load_folding_matrix(args, spectra_dim: int):
    """Return folding matrix of shape (hs_channels, spectra_dim)."""
    if args.no_folding:
        if args.hs_channels != spectra_dim:
            raise ValueError("--no_folding requires hs_channels == spectra dimension")
        return np.eye(spectra_dim, dtype=np.float32)

    if args.folding_csv is not None:
        M = pd.read_csv(args.folding_csv, header=None).values.astype(np.float32)
        if M.shape != (args.hs_channels, spectra_dim):
            raise ValueError(f"Folding matrix shape {M.shape} != ({args.hs_channels},{spectra_dim})")
        return M

    if args.srf_csv is not None:
        srf = pd.read_csv(args.srf_csv)
        if srf.shape[1] < 2:
            raise ValueError("SRF CSV must have wavelength column + >=1 band columns")
        wl = srf.iloc[:, 0].values
        bands = srf.columns[1:]
        if len(bands) != args.hs_channels:
            print(f"[warn] SRF has {len(bands)} bands; overriding hs_channels to match.")
            args.hs_channels = len(bands)
        wl_target = np.arange(400, 2501, 1, dtype=np.float32)  # 2101
        M = []
        for b in bands:
            resp = srf[b].values.astype(np.float32)
            resp_i = np.interp(wl_target, wl, resp)
            if resp_i.sum() <= 0:
                raise ValueError(f"Band {b} SRF sums to zero after interp")
            resp_i = resp_i / (resp_i.sum() + 1e-8)
            M.append(resp_i)
        M = np.stack(M, axis=0).astype(np.float32)
        if M.shape[1] != spectra_dim:
            raise ValueError(f"Built SRF folding width {M.shape[1]} != spectra_dim {spectra_dim}")
        return M

    if args.hs_channels == spectra_dim:
        return np.eye(spectra_dim, dtype=np.float32)
    raise ValueError("Provide either --no_folding (with matching dims), --srf_csv, or --folding_csv")


def spectral_angle_mapper(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8):
    """SAM in radians over last dimension."""
    num = (a * b).sum(dim=-1)
    den = torch.linalg.norm(a, dim=-1) * torch.linalg.norm(b, dim=-1) + eps
    cosang = torch.clamp(num / den, -1.0, 1.0)
    return torch.arccos(cosang)


# ------------------------------
# Datasets
# ------------------------------
class PairedDataset(Dataset):
    def __init__(self, X_params, Y_spec):
        self.X = torch.from_numpy(X_params)
        self.Y = torch.from_numpy(Y_spec)
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, i):
        return self.X[i], self.Y[i]


# ------------------------------
# Models
# ------------------------------
class SpecLayerNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.ln = nn.LayerNorm(d)
    def forward(self, x):
        return self.ln(x)


class TransformerForward(nn.Module):
    """Params -> Spectrum (folded)."""
    def __init__(self, input_dim: int, output_dim: int, hidden: int = 512, nhead: int = 4, nlayers: int = 2):
        super().__init__()
        self.mlp_in = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=nhead, dim_feedforward=hidden*2,
                                                   batch_first=True, activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.proj = nn.Sequential(
            nn.Linear(hidden, output_dim),
            nn.Sigmoid()  # reflectance in [0,1]
        )
        # simple smoothing
        self.smooth = nn.Conv1d(1, 1, kernel_size=5, padding=2, bias=False)
        with torch.no_grad():
            self.smooth.weight.fill_(1/5.0)
            self.smooth.weight.requires_grad_(False)

    def forward(self, x):
        # x: [B, P]
        h = self.mlp_in(x)[:, None, :]           # [B,1,H]
        h = self.encoder(h)[:, 0, :]             # [B,H]
        s = self.proj(h)                         # [B,L]
        s = self.smooth(s.unsqueeze(1)).squeeze(1)
        return s


class InverseConvAttn(nn.Module):
    """Spectrum (folded) -> Params."""
    def __init__(self, input_dim: int, output_dim: int, hidden: int = 512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2, stride=2),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2, stride=2),
            nn.GELU(),
        )
        self.attn = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        conv_out = (input_dim + 3) // 4
        self.head = nn.Sequential(
            nn.Linear(128 * conv_out, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, s):
        z = self.conv(s.unsqueeze(1))                       # [B,128,L/4]
        z = z.transpose(1, 2)                               # [B,T,C]
        z,_ = self.attn(z, z, z)
        z = torch.flatten(z, 1)
        p = self.head(z)
        return p


# ------------------------------
# Training
# ------------------------------
@dataclass
class TrainConfig:
    params_csv: str
    spectra_csv: str
    srf_csv: str | None
    folding_csv: str | None
    no_folding: bool
    hs_channels: int
    out_dir: str
    batch_size: int = 1024
    epochs: int = 180
    lr: float = 5e-4
    weight_decay: float = 1e-5
    hidden: int = 512
    lambda_cycle: float = 0.2
    lambda_sam: float = 0.3
    lambda_smooth: float = 0.02
    lambda_fft: float = 0.0
    val_split: float = 0.1
    seed: int = 42
    amp: bool = True


def batch_metrics(y_pred, y_true):
    # RMSE per sample
    rmse = torch.sqrt(F.mse_loss(y_pred, y_true, reduction='none').mean(dim=-1)).mean().item()
    mae  = F.l1_loss(y_pred, y_true, reduction='none').mean(dim=-1).mean().item()
    # R^2 (approx batch)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - y_true.mean()) ** 2) + 1e-8
    r2 = 1 - (ss_res / ss_tot)
    # Spearman (approx with corr of ranks along last dim)
    yp = y_pred.detach()
    yt = y_true.detach()
    yp_rank = torch.argsort(torch.argsort(yp, dim=-1), dim=-1).float()
    yt_rank = torch.argsort(torch.argsort(yt, dim=-1), dim=-1).float()
    spearman = torch.mean(torch.sum((yp_rank - yp_rank.mean(-1, keepdim=True)) * (yt_rank - yt_rank.mean(-1, keepdim=True)), dim=-1)
                          / (torch.linalg.norm(yp_rank - yp_rank.mean(-1, keepdim=True), dim=-1) * torch.linalg.norm(yt_rank - yt_rank.mean(-1, keepdim=True), dim=-1) + 1e-8))
    return rmse, mae, float(r2.item()), float(spearman.item())


def train(args: TrainConfig):
    ensure_dir(args.out_dir)
    set_seed(args.seed)

    # Load CSVs
    param_cols, X_params_raw, spec_cols, Y_spec_full = read_params_and_spectra(args.params_csv, args.spectra_csv)
    spectra_dim = Y_spec_full.shape[1]

    # Folding
    M = load_folding_matrix(args, spectra_dim)  # (hs_channels, spectra_dim)
    Y_fold = (Y_spec_full @ M.T).astype(np.float32)  # [N, hs_channels]

    # Scale
    p_scaler = MinMaxScaler()
    s_scaler = MinMaxScaler()
    X_params = p_scaler.fit_transform(X_params_raw)
    Y_spec = s_scaler.fit_transform(Y_fold)

    # Split
    X_tr, X_val, Y_tr, Y_val = train_test_split(X_params, Y_spec, test_size=args.val_split, random_state=args.seed)

    # Dataloaders
    tr_ds = PairedDataset(X_tr, Y_tr)
    va_ds = PairedDataset(X_val, Y_val)
    num_workers = min(8, os.cpu_count() or 2)
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    va_loader = DataLoader(va_ds, batch_size=4096, shuffle=False, num_workers=num_workers//2 or 1, pin_memory=True)

    # Models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fwd = TransformerForward(input_dim=X_params.shape[1], output_dim=args.hs_channels, hidden=args.hidden).to(device)
    inv = InverseConvAttn(input_dim=args.hs_channels, output_dim=X_params.shape[1], hidden=args.hidden).to(device)

    # Opt + Sched
    opt = torch.optim.AdamW(list(fwd.parameters()) + list(inv.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
    scaler = GradScaler(enabled=args.amp)

    mse = nn.MSELoss()
    l1 = nn.L1Loss()

    best_sam = float('inf')
    log = []

    for epoch in range(1, args.epochs + 1):
        fwd.train(); inv.train()
        tr_loss = 0.0
        for xb, yb in tr_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            # param noise for robustness
            xb_noisy = xb + 0.01 * torch.randn_like(xb)

            with autocast(enabled=args.amp):
                # forward & inverse
                y_hat = fwd(xb_noisy)
                x_hat = inv(yb)

                # cycle
                y_cyc = fwd(x_hat)
                x_cyc = inv(y_hat)

                # base losses
                loss_fwd = mse(y_hat, yb)
                loss_inv = mse(x_hat, xb)
                loss_cyc = l1(y_cyc, yb) + l1(x_cyc, xb)

                # SAM
                loss_sam = spectral_angle_mapper(y_hat, yb).mean()

                # spectral smoothness (adjacent bands)
                smooth_loss = torch.mean((y_hat[:, 1:] - y_hat[:, :-1]) ** 2)

                # optional FFT magnitude loss
                if args.lambda_fft > 0:
                    y_fft = torch.fft.rfft(y_hat, dim=-1).abs()
                    t_fft = torch.fft.rfft(yb, dim=-1).abs()
                    fft_loss = F.mse_loss(y_fft, t_fft)
                else:
                    fft_loss = torch.tensor(0.0, device=device)

                # dynamic lambda for cycle (decay over epochs: start high -> lower later)
                lam_cycle = args.lambda_cycle * 0.5 * (1.0 + np.cos(np.pi * epoch / args.epochs))

                loss = loss_fwd + loss_inv \
                       + lam_cycle * loss_cyc \
                       + args.lambda_sam * loss_sam \
                       + args.lambda_smooth * smooth_loss \
                       + args.lambda_fft * fft_loss

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            # grad clip after unscale
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(list(fwd.parameters()) + list(inv.parameters()), 1.0)
            scaler.step(opt)
            scaler.update()
            tr_loss += loss.item() * xb.size(0)

        tr_loss /= len(tr_ds)
        scheduler.step()

        # Validation
        fwd.eval(); inv.eval()
        with torch.no_grad():
            sam_list = []
            rmse_list = []
            metrics_list = []
            for xb, yb in va_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                y_hat = fwd(xb)
                sam = spectral_angle_mapper(y_hat, yb)
                sam_list.append(sam)
                rmse = torch.sqrt(F.mse_loss(y_hat, yb, reduction='none').mean(dim=-1))
                rmse_list.append(rmse)
                metrics_list.append(batch_metrics(y_hat, yb))
            sam_val = torch.cat(sam_list).mean().item()
            rmse_val = torch.cat(rmse_list).mean().item()
            # average batch-wise metrics
            if metrics_list:
                r_rmse = np.mean([m[0] for m in metrics_list]).item()
                r_mae  = np.mean([m[1] for m in metrics_list]).item()
                r_r2   = float(np.mean([m[2] for m in metrics_list]))
                r_sp   = float(np.mean([m[3] for m in metrics_list]))
            else:
                r_rmse = rmse_val; r_mae = 0.; r_r2 = 0.; r_sp = 0.

        log.append({"epoch": epoch, "train_loss": tr_loss, "val_SAM(rad)": sam_val, "val_RMSE": rmse_val,
                    "RMSE_batch": r_rmse, "MAE_batch": r_mae, "R2_batch": r_r2, "Spearman_batch": r_sp,
                    "lr": scheduler.get_last_lr()[0]})
        print(f"Epoch {epoch:03d} | train {tr_loss:.5f} | val SAM {sam_val:.4f} rad | val RMSE {rmse_val:.5f} | R2 {r_r2:.3f} | lr {scheduler.get_last_lr()[0]:.2e}")

        # Track best by SAM
        if sam_val < best_sam:
            best_sam = sam_val
            save_all(args, fwd, inv, p_scaler, s_scaler, M, param_cols, spec_cols, log, best=True)

    # Final save
    save_all(args, fwd, inv, p_scaler, s_scaler, M, param_cols, spec_cols, log, best=False)


def save_all(args, fwd, inv, p_scaler, s_scaler, M, param_cols, spec_cols, log, best: bool):
    tag = "best" if best else "final"
    torch.save({
        'fwd': fwd.state_dict(),
        'inv': inv.state_dict(),
        'param_cols': param_cols,
        'spec_cols': spec_cols,
        'config': asdict(args),
        'tag': tag,
    }, os.path.join(args.out_dir, f"surrogate_models_{tag}.pth"))

    joblib.dump({'param_scaler': p_scaler, 'spec_scaler': s_scaler}, os.path.join(args.out_dir, 'scalers.pkl'))
    np.savetxt(os.path.join(args.out_dir, 'folding_used.csv'), M, delimiter=',')

    with open(os.path.join(args.out_dir, 'config.json'), 'w') as f:
        json.dump(asdict(args), f, indent=2)
    with open(os.path.join(args.out_dir, 'val_metrics.json'), 'w') as f:
        json.dump(log, f, indent=2)


# ------------------------------
# CLI
# ------------------------------

def build_argparser():
    p = argparse.ArgumentParser(description="Train PROSAIL forward/inverse surrogates (v2)")
    p.add_argument('--params_csv', type=str, default='./InputPROSAIL_params.csv')
    p.add_argument('--spectra_csv', type=str, default='./PROSAIL_to_PRISMA_168_SORTED.csv')
    p.add_argument('--srf_csv', type=str, default=None, help='CSV with wavelength + band SRFs (columns).')
    p.add_argument('--folding_csv', type=str, default=None, help='Precomputed folding matrix (hs_channels x 2101).')
    p.add_argument('--no_folding', action='store_true', help='Use identity (requires hs_channels==2101).')
    p.add_argument('--hs_channels', type=int, default=168)
    p.add_argument('--out_dir', type=str, default='./surrogate_out_v3')
    p.add_argument('--batch_size', type=int, default=1024)
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--weight_decay', type=float, default=1e-5)
    p.add_argument('--hidden', type=int, default=512)
    p.add_argument('--lambda_cycle', type=float, default=0.2)
    p.add_argument('--lambda_sam', type=float, default=0.3)
    p.add_argument('--lambda_smooth', type=float, default=0.02)
    p.add_argument('--lambda_fft', type=float, default=0.0)
    p.add_argument('--val_split', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no_amp', action='store_true', help='Disable mixed precision')
    return p


if __name__ == '__main__':
    ap = build_argparser()
    a = ap.parse_args()
    cfg = TrainConfig(
        params_csv=a.params_csv,
        spectra_csv=a.spectra_csv,
        srf_csv=a.srf_csv,
        folding_csv=a.folding_csv,
        no_folding=a.no_folding,
        hs_channels=a.hs_channels,
        out_dir=a.out_dir,
        batch_size=a.batch_size,
        epochs=a.epochs,
        lr=a.lr,
        weight_decay=a.weight_decay,
        hidden=a.hidden,
        lambda_cycle=a.lambda_cycle,
        lambda_sam=a.lambda_sam,
        lambda_smooth=a.lambda_smooth,
        lambda_fft=a.lambda_fft,
        val_split=a.val_split,
        seed=a.seed,
        amp=(not a.no_amp),
    )
    train(cfg)
