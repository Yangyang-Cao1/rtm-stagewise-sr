#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_band_folding.py  (PRISMA version)

Generate a Gaussian folding matrix for PRISMA 168 bands (center wavelengths only),
mapping RTM/PROSAIL spectra (400–2500 nm @ 1 nm, 2101 dims) -> PRISMA 168 bands.

Output:
  - CSV folding matrix M with shape (168, 2101)
  - Optional preview plot (.png)

IMPORTANT:
  - This script SORTS PRISMA wavelengths ascending and generates M in that sorted order.
  - In your fusion script, you MUST apply the same argsort index to reorder PRISMA HS bands
    (Y_hs[:, :, idx]) so that everything stays consistent.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

def sigma_piecewise_nm(w_nm: float) -> float:
    """
    Piecewise sigma (nm) for PRISMA bandpass approximation (no official SRF available).
    Good default for agriculture scenes; tweak if you want sharper/smoother folding.
    """
    if w_nm < 900.0:
        return 8.0
    elif w_nm < 1800.0:
        return 12.0
    else:
        return 16.0

def build_folding_matrix_prisma(
    wl_center_nm_sorted: np.ndarray,
    wl_min: int = 400,
    wl_max: int = 2500,
    use_piecewise_sigma: bool = True,
    sigma_const_nm: float = 12.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    wl_center_nm_sorted: (B,) sorted ascending center wavelengths (nm)
    Returns:
      M: (B, 2101) float32, each row sums to 1
      wl_full: (2101,) wavelengths 400..2500
    """
    wl_full = np.arange(wl_min, wl_max + 1, 1.0, dtype=np.float32)  # 2101
    B = wl_center_nm_sorted.shape[0]
    M = np.zeros((B, wl_full.shape[0]), dtype=np.float32)

    for i, c in enumerate(wl_center_nm_sorted.astype(np.float32)):
        if use_piecewise_sigma:
            sig = sigma_piecewise_nm(float(c))
        else:
            sig = float(sigma_const_nm)

        # Gaussian bandpass weights
        w = np.exp(-0.5 * ((wl_full - c) / sig) ** 2).astype(np.float32)
        s = float(w.sum())
        if s > 0:
            w /= s
        M[i, :] = w

    return M, wl_full

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wl_npy", type=str, default="filtered_wavelengths.npy",
                    help="PRISMA center wavelengths (.npy), length=168")
    ap.add_argument("--out_csv", type=str, default="M_folding_PRISMA168.csv",
                    help="Output folding matrix CSV (168x2101)")
    ap.add_argument("--save_idx", type=str, default="prisma_wl_sort_idx.npy",
                    help="Save argsort index for wl sorting (used to reorder PRISMA HS bands)")
    ap.add_argument("--no_plot", action="store_true",
                    help="Disable saving preview plot")
    ap.add_argument("--plot_png", type=str, default="M_folding_PRISMA168_preview.png",
                    help="Preview plot path (png)")

    # sigma control
    ap.add_argument("--use_piecewise_sigma", action="store_true", default=True,
                    help="Use piecewise sigma (recommended). Default: True")
    ap.add_argument("--use_const_sigma", action="store_true",
                    help="Use constant sigma instead of piecewise")
    ap.add_argument("--sigma", type=float, default=12.0,
                    help="Constant sigma (nm) if --use_const_sigma is set")

    args = ap.parse_args()

    wl_center_nm = np.load(args.wl_npy).astype(np.float32)  # (168,)
    if wl_center_nm.ndim != 1:
        raise ValueError(f"wl_npy must be 1D array, got shape {wl_center_nm.shape}")

    # Sort wavelengths ascending (PRISMA wl may not be strictly increasing)
    idx = np.argsort(wl_center_nm)
    wl_sorted = wl_center_nm[idx]

    # Save idx for use in fusion script (reorder Y_hs with same idx)
    np.save(args.save_idx, idx)
    print(f"Saved sort index: {args.save_idx}  (apply Y_hs[:,:,idx] in fusion)")

    use_piecewise = True
    sigma_const = float(args.sigma)
    if args.use_const_sigma:
        use_piecewise = False

    M, wl_full = build_folding_matrix_prisma(
        wl_center_nm_sorted=wl_sorted,
        wl_min=400,
        wl_max=2500,
        use_piecewise_sigma=use_piecewise,
        sigma_const_nm=sigma_const
    )

    np.savetxt(args.out_csv, M, delimiter=",")
    print(f"Saved folding CSV: {args.out_csv}  shape={M.shape}")
    print("Wavelength grid for RTM assumed: 400..2500 nm @1nm (2101 dims).")

    # Optional preview plot
    if not args.no_plot:
        plt.figure(figsize=(9, 5))
        # plot a few rows
        step = max(1, M.shape[0] // 10)
        for i in range(0, M.shape[0], step):
            plt.plot(wl_full, M[i, :], linewidth=1)
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Relative response (row-normalized)")
        plt.title("PRISMA folding matrix preview (Gaussian bandpass)")
        plt.tight_layout()
        plt.savefig(args.plot_png, dpi=200)
        print(f"Saved preview plot: {args.plot_png}")

if __name__ == "__main__":
    main()
