#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os

import torch

from .surrogate import InverseConvAttn, TransformerForward


def _get(cfg: dict, key: str, default):
    value = cfg.get(key, default)
    return default if value is None else value


def export_surrogate(
    *,
    ckpt_path: str,
    out_fwd_ts: str,
    out_inv_ts: str,
    device: str = "cpu",
) -> None:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})
    param_cols = ckpt["param_cols"]
    spec_cols = ckpt["spec_cols"]

    params_dim = len(param_cols)
    hs_channels = int(_get(cfg, "hs_channels", len(spec_cols)))
    hidden = int(_get(cfg, "hidden", 512))
    nhead = 4
    nlayers = 2

    fwd = TransformerForward(
        input_dim=params_dim,
        output_dim=hs_channels,
        hidden=hidden,
        nhead=nhead,
        nlayers=nlayers,
    ).to(device).eval()
    inv = InverseConvAttn(
        input_dim=hs_channels,
        output_dim=params_dim,
        hidden=hidden,
    ).to(device).eval()

    fwd.load_state_dict(ckpt["fwd"], strict=True)
    inv.load_state_dict(ckpt["inv"], strict=True)

    dummy_params = torch.randn(1, params_dim, device=device)
    dummy_spec = torch.randn(1, hs_channels, device=device)
    fwd_ts = torch.jit.trace(fwd, dummy_params, check_trace=False)
    inv_ts = torch.jit.trace(inv, dummy_spec, check_trace=False)

    os.makedirs(os.path.dirname(os.path.abspath(out_fwd_ts)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(out_inv_ts)), exist_ok=True)
    torch.jit.save(fwd_ts, out_fwd_ts)
    torch.jit.save(inv_ts, out_inv_ts)
    print(f"Saved: {out_fwd_ts}")
    print(f"Saved: {out_inv_ts}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export train_surrogate.py checkpoints to TorchScript.")
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--out_fwd_ts", required=True)
    parser.add_argument("--out_inv_ts", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    export_surrogate(
        ckpt_path=args.ckpt_path,
        out_fwd_ts=args.out_fwd_ts,
        out_inv_ts=args.out_inv_ts,
        device=args.device,
    )


if __name__ == "__main__":
    main()
