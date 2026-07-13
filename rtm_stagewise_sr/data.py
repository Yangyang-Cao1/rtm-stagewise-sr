#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from dataclasses import dataclass
from typing import Any

import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)
DEFAULT_PRISMA_PATH = os.path.normpath(
    os.path.join(REPO_ROOT, "datasets_crop/dataset_onepatch/prisma_patches/ALL_POINTS_prisma.npy")
)
DEFAULT_S2_PATH = os.path.normpath(
    os.path.join(REPO_ROOT, "datasets_crop/dataset_onepatch/s2_patches/ALL_POINTS_s2.npy")
)
DEFAULT_WL_PATH = os.path.join(REPO_ROOT, "artifacts/filtered_wavelengths.npy")


@dataclass
class CoreData:
    hs_hwc_sorted: np.ndarray
    ms_hwc: np.ndarray
    wavelengths_sorted_nm: np.ndarray
    wavelengths_original_nm: np.ndarray
    sort_idx: np.ndarray
    metadata: dict[str, Any]


def _load_npy(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return np.load(path)


def normalize_unit_range(array: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    array = array.astype(np.float32, copy=False)
    arr_min = float(array.min())
    arr_max = float(array.max())
    return (array - arr_min) / (arr_max - arr_min + eps)


def ensure_hwc(array: np.ndarray, expected_channels: int, name: str) -> np.ndarray:
    if array.ndim != 3:
        raise ValueError(f"{name} must be 3D, got shape={array.shape}")
    if array.shape[-1] == expected_channels:
        return array.astype(np.float32, copy=False)
    if array.shape[0] == expected_channels:
        return np.transpose(array, (1, 2, 0)).astype(np.float32, copy=False)
    raise ValueError(
        f"{name} does not match expected channels={expected_channels}. "
        f"Observed shape={array.shape}"
    )


def sort_wavelengths_and_hs(
    wavelengths_nm: np.ndarray,
    hs_hwc: np.ndarray,
    *band_aligned_arrays: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    wavelengths_nm = np.asarray(wavelengths_nm, dtype=np.float32).reshape(-1)
    hs_hwc = np.asarray(hs_hwc, dtype=np.float32)
    if hs_hwc.ndim != 3:
        raise ValueError(f"hs_hwc must be HWC, got shape={hs_hwc.shape}")
    if hs_hwc.shape[-1] != wavelengths_nm.shape[0]:
        raise ValueError(
            f"HS band count and wavelength count mismatch: hs={hs_hwc.shape[-1]}, wl={wavelengths_nm.shape[0]}"
        )

    sort_idx = np.argsort(wavelengths_nm)
    wavelengths_sorted = wavelengths_nm[sort_idx]
    hs_sorted = hs_hwc[..., sort_idx]

    sorted_extras: list[np.ndarray] = []
    for arr in band_aligned_arrays:
        arr_np = np.asarray(arr)
        if arr_np.shape[-1] != wavelengths_nm.shape[0]:
            raise ValueError(
                f"Band-aligned extra does not match wavelength count. "
                f"shape={arr_np.shape}, wl={wavelengths_nm.shape[0]}"
            )
        sorted_extras.append(arr_np[..., sort_idx])
    return wavelengths_sorted, hs_sorted, sort_idx.astype(np.int64), sorted_extras


def unsort_bands(sorted_array: np.ndarray, sort_idx: np.ndarray) -> np.ndarray:
    inverse_idx = np.argsort(np.asarray(sort_idx))
    return np.asarray(sorted_array)[..., inverse_idx]


def load_core_data(
    prisma_path: str = DEFAULT_PRISMA_PATH,
    s2_path: str = DEFAULT_S2_PATH,
    wavelength_path: str = DEFAULT_WL_PATH,
    normalize: bool = True,
) -> CoreData:
    hs_raw = _load_npy(prisma_path)
    ms_raw = _load_npy(s2_path)
    wl_raw = _load_npy(wavelength_path).astype(np.float32).reshape(-1)

    hs_hwc = ensure_hwc(hs_raw, expected_channels=wl_raw.shape[0], name="PRISMA")
    ms_hwc = ensure_hwc(ms_raw, expected_channels=9, name="Sentinel-2")

    if normalize:
        hs_hwc = normalize_unit_range(hs_hwc)
        ms_hwc = normalize_unit_range(ms_hwc)

    wl_sorted, hs_sorted, sort_idx, _ = sort_wavelengths_and_hs(wl_raw, hs_hwc)

    metadata = {
        "prisma_path": os.path.abspath(prisma_path),
        "s2_path": os.path.abspath(s2_path),
        "wavelength_path": os.path.abspath(wavelength_path),
        "hs_shape_sorted_hwc": list(hs_sorted.shape),
        "ms_shape_hwc": list(ms_hwc.shape),
        "n_hs_bands": int(hs_sorted.shape[-1]),
        "n_ms_bands": int(ms_hwc.shape[-1]),
        "normalization": "global_minmax" if normalize else "none",
    }
    return CoreData(
        hs_hwc_sorted=hs_sorted,
        ms_hwc=ms_hwc,
        wavelengths_sorted_nm=wl_sorted,
        wavelengths_original_nm=wl_raw,
        sort_idx=sort_idx,
        metadata=metadata,
    )
