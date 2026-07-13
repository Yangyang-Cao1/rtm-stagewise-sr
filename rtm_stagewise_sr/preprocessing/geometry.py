#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
from rasterio.windows import Window
from rasterio.windows import bounds as win_bounds
from rasterio.windows import from_bounds


def normalize_s2(arr: np.ndarray) -> np.ndarray:
    mx = np.nanmax(arr)
    if mx > 100:
        return arr.astype(np.float32) / 10000.0
    return arr.astype(np.float32)


def normalize_hsi(
    arr: np.ndarray,
    *,
    nodata_value: float | None = None,
) -> tuple[np.ndarray, dict[str, float | str | bool | None]]:
    arr_f = arr.astype(np.float32, copy=True)
    finite_mask = np.isfinite(arr_f)
    nodata_mask = ~finite_mask
    if nodata_value is not None and np.isfinite(float(nodata_value)):
        nodata_mask |= arr_f == float(nodata_value)
    # Catch integer-coded nodata such as EnMAP's -32768 even when nodata metadata is absent.
    nodata_mask |= arr_f <= -1.0e4

    valid_mask = finite_mask & (~nodata_mask)
    valid_values = arr_f[valid_mask]
    scale_factor = 1.0
    if valid_values.size > 0 and float(np.nanmax(valid_values)) > 100.0:
        scale_factor = 10000.0
        arr_f = arr_f / scale_factor
        valid_values = arr_f[valid_mask]

    arr_f[nodata_mask] = 0.0
    summary: dict[str, float | str | bool | None] = {
        "scale_policy": "divide_by_10000_if_max_gt_100",
        "scale_factor": float(scale_factor),
        "nodata_value": (None if nodata_value is None else float(nodata_value)),
        "nodata_replaced_with_zero": True,
        "valid_min_after_scale": (None if valid_values.size == 0 else float(np.min(valid_values))),
        "valid_max_after_scale": (None if valid_values.size == 0 else float(np.max(valid_values))),
    }
    return arr_f.astype(np.float32, copy=False), summary


def ceil_to_multiple(x: int, m: int) -> int:
    return int(math.ceil(float(x) / float(m)) * m)


def rect_from_center(x: float, y: float, side_m: float) -> tuple[float, float, float, float]:
    half = 0.5 * float(side_m)
    return (float(x - half), float(y - half), float(x + half), float(y + half))


def intersect_area(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    left = max(float(a[0]), float(b[0]))
    bottom = max(float(a[1]), float(b[1]))
    right = min(float(a[2]), float(b[2]))
    top = min(float(a[3]), float(b[3]))
    if right <= left or top <= bottom:
        return 0.0
    return float((right - left) * (top - bottom))


def window_to_cover_bounds_centered(
    bounds_xy: tuple[float, float, float, float],
    transform,
    width: int,
    height: int,
    align: int,
) -> Window:
    win0 = from_bounds(*bounds_xy, transform=transform)
    col_start = int(math.floor(win0.col_off))
    row_start = int(math.floor(win0.row_off))
    col_end = int(math.ceil(win0.col_off + win0.width))
    row_end = int(math.ceil(win0.row_off + win0.height))
    raw_w = max(1, col_end - col_start)
    raw_h = max(1, row_end - row_start)
    width_aligned = min(width, ceil_to_multiple(raw_w, align))
    height_aligned = min(height, ceil_to_multiple(raw_h, align))
    center_col = float(win0.col_off + 0.5 * win0.width)
    center_row = float(win0.row_off + 0.5 * win0.height)
    col0 = int(round(center_col - 0.5 * width_aligned))
    row0 = int(round(center_row - 0.5 * height_aligned))
    col0 = min(max(0, col0), max(0, width - width_aligned))
    row0 = min(max(0, row0), max(0, height - height_aligned))
    return Window(int(col0), int(row0), int(width_aligned), int(height_aligned))


def pixel_bounds(transform, row: int, col: int) -> tuple[float, float, float, float]:
    return tuple(float(v) for v in win_bounds(Window(int(col), int(row), 1, 1), transform))


def footprint_pixels(
    *,
    ds,
    patch_window: Window,
    plot_bounds_xy: tuple[float, float, float, float],
) -> list[dict[str, object]]:
    candidate = from_bounds(*plot_bounds_xy, transform=ds.transform)
    col_start = max(0, int(math.floor(candidate.col_off)) - 1)
    row_start = max(0, int(math.floor(candidate.row_off)) - 1)
    col_end = min(ds.width, int(math.ceil(candidate.col_off + candidate.width)) + 1)
    row_end = min(ds.height, int(math.ceil(candidate.row_off + candidate.height)) + 1)
    area_total = float((plot_bounds_xy[2] - plot_bounds_xy[0]) * (plot_bounds_xy[3] - plot_bounds_xy[1]))
    pixels: list[dict[str, object]] = []
    area_sum = 0.0
    for row in range(row_start, row_end):
        for col in range(col_start, col_end):
            pix_bounds = pixel_bounds(ds.transform, row=row, col=col)
            overlap = intersect_area(plot_bounds_xy, pix_bounds)
            if overlap <= 0.0:
                continue
            area_sum += overlap
            pixels.append(
                {
                    "rc_global": [int(row), int(col)],
                    "rc_in_patch": [int(row - patch_window.row_off), int(col - patch_window.col_off)],
                    "overlap_area_m2": float(overlap),
                    "weight": float(overlap / max(area_total, 1e-12)),
                    "pixel_bbox_xy_lbrt": [float(v) for v in pix_bounds],
                }
            )
    if not pixels:
        raise RuntimeError("No overlapping pixels found for plot footprint.")
    norm = max(area_sum, 1e-12)
    for item in pixels:
        item["weight"] = float(item["overlap_area_m2"] / norm)
    return pixels


def bbox_from_patch_rows_cols(rows: list[int], cols: list[int], height: int, width: int, margin: int) -> list[int]:
    min_row = max(0, min(rows) - int(margin))
    max_row = min(height - 1, max(rows) + int(margin))
    min_col = max(0, min(cols) - int(margin))
    max_col = min(width - 1, max(cols) + int(margin))
    box_h = max_row - min_row + 1
    box_w = max_col - min_col + 1
    size = max(box_h, box_w)
    center_row = 0.5 * (min_row + max_row)
    center_col = 0.5 * (min_col + max_col)
    y0 = int(round(center_row - 0.5 * size))
    x0 = int(round(center_col - 0.5 * size))
    y0 = min(max(0, y0), max(0, height - size))
    x0 = min(max(0, x0), max(0, width - size))
    y1 = min(height, y0 + size)
    x1 = min(width, x0 + size)
    return [int(y0), int(x0), int(y1), int(x1)]


def _stretch(image_hwc: np.ndarray, pmin: float = 2.0, pmax: float = 98.0) -> np.ndarray:
    def _valid_values(band_hw: np.ndarray) -> np.ndarray:
        values = band_hw[np.isfinite(band_hw)]
        if values.size == 0:
            return values
        nodata_mask = values <= -1.0e4
        if np.any(nodata_mask) and np.any(~nodata_mask):
            values = values[~nodata_mask]
        return values

    out = np.zeros_like(image_hwc, dtype=np.float32)
    for idx in range(image_hwc.shape[-1]):
        band = image_hwc[..., idx].astype(np.float32)
        valid_values = _valid_values(band)
        if valid_values.size == 0:
            continue
        lo, hi = np.percentile(valid_values, [pmin, pmax])
        if not np.isfinite(lo) or not np.isfinite(hi):
            continue
        if hi <= lo:
            hi = float(np.max(valid_values))
            lo = float(np.min(valid_values))
            if hi <= lo:
                continue
        stretched = np.clip((band - lo) / (hi - lo + 1e-12), 0.0, 1.0)
        stretched[~np.isfinite(band)] = 0.0
        stretched[band <= -1.0e4] = 0.0
        out[..., idx] = stretched
    return out
