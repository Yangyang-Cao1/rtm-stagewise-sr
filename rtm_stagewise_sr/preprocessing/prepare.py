#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.windows import Window
from rasterio.windows import bounds as win_bounds
from rasterio.windows import from_bounds
from rasterio.windows import transform as win_transform

from .geometry import (
    _stretch,
    bbox_from_patch_rows_cols,
    footprint_pixels,
    normalize_hsi,
    normalize_s2,
    rect_from_center,
    window_to_cover_bounds_centered,
)


DEFAULT_ALIGN = 128
DEFAULT_MARGIN_M = 60.0
DEFAULT_PLOT_SIDE_M = 15.0
DEFAULT_ROI_MARGIN_S2_PX = 24
DEFAULT_PATCH_NAME = "ALL_POINTS"


@dataclass
class FieldPoint:
    id: str
    lon: float
    lat: float
    csv_row_index: int
    plot_side_m: float
    plot_id_original: str | None


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _find_band_idx(wavelengths_nm: np.ndarray, target_nm: float) -> int:
    return int(np.argmin(np.abs(wavelengths_nm.astype(np.float64) - float(target_nm))))


def _infer_hsi_sensor_label(source_hsi_path: str | None) -> str:
    path_text = (source_hsi_path or "").lower()
    if "enmap" in path_text:
        return "EnMAP"
    if "prisma" in path_text:
        return "PRISMA"
    return "HSI"


def _select_hsi_rgb(
    hsi_patch_hwc: np.ndarray,
    wavelengths_path: str | None,
) -> np.ndarray:
    if wavelengths_path and os.path.exists(wavelengths_path):
        wavelengths_nm = np.load(wavelengths_path).astype(np.float32).reshape(-1)
        if wavelengths_nm.shape[0] == hsi_patch_hwc.shape[-1]:
            idx_r = _find_band_idx(wavelengths_nm, 664.5)
            idx_g = _find_band_idx(wavelengths_nm, 560.0)
            idx_b = _find_band_idx(wavelengths_nm, 496.6)
            return _stretch(hsi_patch_hwc[..., [idx_r, idx_g, idx_b]])
    fallback_idx = [min(hsi_patch_hwc.shape[-1] - 1, idx) for idx in [28, 16, 9]]
    return _stretch(hsi_patch_hwc[..., fallback_idx])


def _make_unique_ids(raw_ids: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    output: list[str] = []
    for raw in raw_ids:
        count = counts.get(raw, 0)
        counts[raw] = count + 1
        output.append(raw if count == 0 else f"{raw}__dup{count + 1}")
    return output


def load_valid_points_from_field_csv(csv_path: str, plot_side_m: float) -> tuple[list[FieldPoint], dict[str, object]]:
    frame = pd.read_csv(csv_path, header=None)
    header_numeric = pd.to_numeric(frame.iloc[0, :], errors="coerce")
    band_cols = np.where(np.isfinite(header_numeric.to_numpy()))[0]
    if band_cols.size < 2:
        raise ValueError(f"Could not infer spectral columns from field CSV: {csv_path}")

    raw_points: list[FieldPoint] = []
    dropped_rows: list[int] = []
    raw_ids: list[str] = []
    for row_idx in range(1, len(frame)):
        row = frame.iloc[row_idx, :]
        lat = pd.to_numeric(row.iloc[6], errors="coerce") if frame.shape[1] > 6 else np.nan
        lon = pd.to_numeric(row.iloc[7], errors="coerce") if frame.shape[1] > 7 else np.nan
        spec = pd.to_numeric(row.iloc[band_cols], errors="coerce").to_numpy(dtype=float)
        has_valid_spec = np.isfinite(spec).sum() >= 2 and not np.all(np.isnan(spec))
        if not np.isfinite(lat) or not np.isfinite(lon) or not has_valid_spec:
            dropped_rows.append(int(row_idx))
            continue
        plot_id_original = None
        if frame.shape[1] > 8 and pd.notna(row.iloc[8]):
            text = str(row.iloc[8]).strip()
            plot_id_original = text if text else None
        raw_id = plot_id_original or f"row_{row_idx:02d}"
        raw_ids.append(raw_id)
        raw_points.append(
            FieldPoint(
                id=raw_id,
                lon=float(lon),
                lat=float(lat),
                csv_row_index=int(row_idx),
                plot_side_m=float(plot_side_m),
                plot_id_original=plot_id_original,
            )
        )

    unique_ids = _make_unique_ids(raw_ids)
    points = [
        FieldPoint(
            id=unique_ids[idx],
            lon=point.lon,
            lat=point.lat,
            csv_row_index=point.csv_row_index,
            plot_side_m=point.plot_side_m,
            plot_id_original=point.plot_id_original,
        )
        for idx, point in enumerate(raw_points)
    ]
    summary = {
        "csv_path": os.path.abspath(csv_path),
        "total_rows_including_header": int(len(frame)),
        "candidate_data_rows": int(len(frame) - 1),
        "valid_point_count": int(len(points)),
        "dropped_row_indices": dropped_rows,
        "spectral_band_cols": [int(v) for v in band_cols.tolist()],
        "field_wavelength_min_nm": float(np.nanmin(header_numeric.iloc[band_cols].to_numpy(dtype=float))),
        "field_wavelength_max_nm": float(np.nanmax(header_numeric.iloc[band_cols].to_numpy(dtype=float))),
    }
    return points, summary


def save_diagnostics(
    *,
    prisma_patch_hwc: np.ndarray,
    source_hsi_path: str,
    wavelengths_path: str | None,
    diag_out: str,
    s2_patch_hwc: np.ndarray,
    points_meta: list[dict[str, object]],
    roi_bbox_s2: list[int],
    roi_bbox_prisma: list[int],
    title_prefix: str,
) -> None:
    ensure_dir(diag_out)
    with np.errstate(invalid="ignore"):
        prisma_rgb = _select_hsi_rgb(prisma_patch_hwc, wavelengths_path)
    hsi_sensor_label = _infer_hsi_sensor_label(source_hsi_path)
    s2_rgb = _stretch(s2_patch_hwc[..., [2, 1, 0]])

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(prisma_rgb)
    py0, px0, py1, px1 = roi_bbox_prisma
    ax.add_patch(plt.Rectangle((px0, py0), px1 - px0, py1 - py0, fill=False, edgecolor="cyan", linewidth=1.8))
    for point in points_meta:
        row, col = point["prisma_rc_in_patch"]
        ax.scatter([col], [row], s=16, c="yellow", edgecolors="black", linewidths=0.4)
    ax.set_title(f"{title_prefix} | centered {hsi_sensor_label} patch")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(diag_out, "prisma_patch_field_points.png"), dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(s2_rgb)
    y0, x0, y1, x1 = roi_bbox_s2
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="cyan", linewidth=2.0))
    for point in points_meta:
        row, col = point["s2_rc_in_patch"]
        ax.scatter([col], [row], s=18, c="yellow", edgecolors="black", linewidths=0.4)
        ax.text(col + 1.5, row + 1.5, point["id"], fontsize=6, color="white")
    ax.set_title(f"{title_prefix} | centered S2 patch")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(diag_out, "s2_patch_field_points.png"), dpi=220)
    plt.close(fig)

    roi = s2_rgb[y0:y1, x0:x1, :]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(roi)
    for point in points_meta:
        row, col = point["s2_rc_in_patch"]
        if y0 <= row < y1 and x0 <= col < x1:
            ax.scatter([col - x0], [row - y0], s=20, c="yellow", edgecolors="black", linewidths=0.4)
    ax.set_title(f"{title_prefix} | field-point ROI")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(diag_out, "s2_patch_field_roi.png"), dpi=220)
    plt.close(fig)

    prisma_roi = prisma_rgb[py0:py1, px0:px1, :]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(prisma_roi)
    for point in points_meta:
        row, col = point["prisma_rc_in_patch"]
        if py0 <= row < py1 and px0 <= col < px1:
            ax.scatter([col - px0], [row - py0], s=18, c="yellow", edgecolors="black", linewidths=0.4)
    ax.set_title(f"{title_prefix} | field-point ROI on {hsi_sensor_label}")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(diag_out, "prisma_patch_field_roi.png"), dpi=220)
    plt.close(fig)


def generate_dataset(
    *,
    prisma_path: str,
    s2_path: str,
    field_csv_path: str,
    out_root: str,
    wavelengths_path: str | None = None,
    patch_name: str = DEFAULT_PATCH_NAME,
    align: int = DEFAULT_ALIGN,
    margin_m: float = DEFAULT_MARGIN_M,
    plot_side_m: float = DEFAULT_PLOT_SIDE_M,
    roi_margin_s2_px: int = DEFAULT_ROI_MARGIN_S2_PX,
) -> dict[str, object]:
    prisma_out = ensure_dir(os.path.join(out_root, "prisma_patches"))
    s2_out = ensure_dir(os.path.join(out_root, "s2_patches"))
    meta_out = ensure_dir(os.path.join(out_root, "meta"))
    diag_out = ensure_dir(os.path.join(out_root, "diagnostics"))

    points, csv_summary = load_valid_points_from_field_csv(field_csv_path, plot_side_m=plot_side_m)
    if len(points) == 0:
        raise ValueError(f"No valid field points found in CSV: {field_csv_path}")

    with rasterio.open(prisma_path) as prisma, rasterio.open(s2_path) as s2:
        if prisma.crs != s2.crs:
            raise ValueError(f"CRS mismatch: PRISMA={prisma.crs}, S2={s2.crs}")
        crs_text = prisma.crs.to_string()
        prisma_transform = prisma.transform
        s2_transform = s2.transform

        transformer = Transformer.from_crs("EPSG:4326", prisma.crs, always_xy=True)
        points_xy: list[dict[str, object]] = []
        all_left, all_bottom, all_right, all_top = [], [], [], []
        for point in points:
            x, y = transformer.transform(point.lon, point.lat)
            plot_bounds = rect_from_center(float(x), float(y), point.plot_side_m)
            points_xy.append(
                {
                    "id": point.id,
                    "field_plot_id": point.plot_id_original,
                    "field_csv_row_index": int(point.csv_row_index),
                    "lonlat": [float(point.lon), float(point.lat)],
                    "xy": [float(x), float(y)],
                    "plot_side_m": float(point.plot_side_m),
                    "plot_bbox_xy_lbrt": [float(v) for v in plot_bounds],
                }
            )
            all_left.append(plot_bounds[0])
            all_bottom.append(plot_bounds[1])
            all_right.append(plot_bounds[2])
            all_top.append(plot_bounds[3])

        union_bounds = (
            float(min(all_left) - margin_m),
            float(min(all_bottom) - margin_m),
            float(max(all_right) + margin_m),
            float(max(all_top) + margin_m),
        )
        prisma_window = window_to_cover_bounds_centered(
            union_bounds,
            transform=prisma.transform,
            width=prisma.width,
            height=prisma.height,
            align=align,
        )
        prisma_patch_raw = prisma.read(window=prisma_window)
        prisma_patch, hsi_norm_summary = normalize_hsi(prisma_patch_raw, nodata_value=prisma.nodata)
        bbox_final = win_bounds(prisma_window, prisma.transform)
        s2_window = from_bounds(*bbox_final, transform=s2.transform).round_offsets().round_lengths()
        s2_window = Window(
            int(s2_window.col_off),
            int(s2_window.row_off),
            int(s2_window.width),
            int(s2_window.height),
        )
        s2_patch = normalize_s2(s2.read(window=s2_window))

        points_meta: list[dict[str, object]] = []
        roi_rows: list[int] = []
        roi_cols: list[int] = []
        prisma_roi_rows: list[int] = []
        prisma_roi_cols: list[int] = []
        for point in points_xy:
            pr_row, pr_col = prisma.index(point["xy"][0], point["xy"][1])
            s2_row, s2_col = s2.index(point["xy"][0], point["xy"][1])
            s2_pixels = footprint_pixels(
                ds=s2,
                patch_window=s2_window,
                plot_bounds_xy=tuple(point["plot_bbox_xy_lbrt"]),
            )
            prisma_pixels = footprint_pixels(
                ds=prisma,
                patch_window=prisma_window,
                plot_bounds_xy=tuple(point["plot_bbox_xy_lbrt"]),
            )
            roi_rows.extend(int(pixel["rc_in_patch"][0]) for pixel in s2_pixels)
            roi_cols.extend(int(pixel["rc_in_patch"][1]) for pixel in s2_pixels)
            prisma_roi_rows.extend(int(pixel["rc_in_patch"][0]) for pixel in prisma_pixels)
            prisma_roi_cols.extend(int(pixel["rc_in_patch"][1]) for pixel in prisma_pixels)
            points_meta.append(
                {
                    **point,
                    "prisma_rc_global": [int(pr_row), int(pr_col)],
                    "prisma_rc_in_patch": [int(pr_row - prisma_window.row_off), int(pr_col - prisma_window.col_off)],
                    "s2_rc_global": [int(s2_row), int(s2_col)],
                    "s2_rc_in_patch": [int(s2_row - s2_window.row_off), int(s2_col - s2_window.col_off)],
                    "prisma_plot_pixels": prisma_pixels,
                    "s2_plot_pixels": s2_pixels,
                }
            )

        roi_bbox = bbox_from_patch_rows_cols(
            rows=roi_rows,
            cols=roi_cols,
            height=int(s2_patch.shape[1]),
            width=int(s2_patch.shape[2]),
            margin=roi_margin_s2_px,
        )
        prisma_roi_bbox = bbox_from_patch_rows_cols(
            rows=prisma_roi_rows,
            cols=prisma_roi_cols,
            height=int(prisma_patch.shape[1]),
            width=int(prisma_patch.shape[2]),
            margin=max(4, int(np.ceil(float(roi_margin_s2_px) / 3.0))),
        )

    prisma_npy = os.path.join(prisma_out, f"{patch_name}_prisma.npy")
    s2_npy = os.path.join(s2_out, f"{patch_name}_s2.npy")
    np.save(prisma_npy, prisma_patch)
    np.save(s2_npy, s2_patch)

    meta = {
        "name": patch_name,
        "dataset_variant": "centered_plot15_areaweighted_from_csv",
        "window_strategy": "centered_bbox_aligned",
        "footprint_sampling": "exact_rect_area_overlap",
        "source_csv_path": os.path.abspath(field_csv_path),
        "source_prisma_path": os.path.abspath(prisma_path),
        "source_hsi_path": os.path.abspath(prisma_path),
        "source_hsi_sensor": _infer_hsi_sensor_label(prisma_path),
        "source_hsi_scale_policy": hsi_norm_summary["scale_policy"],
        "source_hsi_scale_factor": hsi_norm_summary["scale_factor"],
        "source_hsi_nodata_value": hsi_norm_summary["nodata_value"],
        "source_hsi_nodata_replaced_with_zero": hsi_norm_summary["nodata_replaced_with_zero"],
        "source_hsi_valid_min_after_scale": hsi_norm_summary["valid_min_after_scale"],
        "source_hsi_valid_max_after_scale": hsi_norm_summary["valid_max_after_scale"],
        "source_s2_path": os.path.abspath(s2_path),
        "margin_m": float(margin_m),
        "align_multiple": int(align),
        "plot_side_m": float(plot_side_m),
        "roi_margin_s2_px": int(roi_margin_s2_px),
        "crs": crs_text,
        "prisma_window": [int(prisma_window.col_off), int(prisma_window.row_off), int(prisma_window.width), int(prisma_window.height)],
        "s2_window": [int(s2_window.col_off), int(s2_window.row_off), int(s2_window.width), int(s2_window.height)],
        "bbox_left_bottom_right_top": [float(v) for v in bbox_final],
        "prisma_patch_transform": list(win_transform(prisma_window, prisma_transform)),
        "s2_patch_transform": list(win_transform(s2_window, s2_transform)),
        "prisma_shape": list(prisma_patch.shape),
        "s2_shape": list(s2_patch.shape),
        "field_points_roi_s2_rc": roi_bbox,
        "field_points_roi_prisma_rc": prisma_roi_bbox,
        "valid_point_count": int(len(points_meta)),
        "csv_summary": csv_summary,
        "points": points_meta,
        "prisma_npy": os.path.join("prisma_patches", f"{patch_name}_prisma.npy"),
        "s2_npy": os.path.join("s2_patches", f"{patch_name}_s2.npy"),
    }
    meta_json = os.path.join(meta_out, f"{patch_name}.json")
    with open(meta_json, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)

    save_diagnostics(
        prisma_patch_hwc=np.transpose(prisma_patch, (1, 2, 0)),
        source_hsi_path=prisma_path,
        wavelengths_path=wavelengths_path,
        diag_out=diag_out,
        s2_patch_hwc=np.transpose(s2_patch, (1, 2, 0)),
        points_meta=points_meta,
        roi_bbox_s2=roi_bbox,
        roi_bbox_prisma=prisma_roi_bbox,
        title_prefix=os.path.basename(os.path.dirname(os.path.abspath(out_root))),
    )
    dataset_summary = {
        "dataset_root": os.path.abspath(out_root),
        "meta_json": os.path.abspath(meta_json),
        "prisma_shape": list(prisma_patch.shape),
        "s2_shape": list(s2_patch.shape),
        "field_points_roi_s2_rc": roi_bbox,
        "field_points_roi_prisma_rc": prisma_roi_bbox,
        "valid_point_count": int(len(points_meta)),
        "dropped_row_indices": csv_summary["dropped_row_indices"],
        "source_hsi_scale_factor": hsi_norm_summary["scale_factor"],
        "source_hsi_valid_max_after_scale": hsi_norm_summary["valid_max_after_scale"],
    }
    with open(os.path.join(diag_out, "dataset_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(dataset_summary, handle, ensure_ascii=False, indent=2)
    return dataset_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate centered onepatch dataset from a field CSV.")
    parser.add_argument("--prisma_path", required=True)
    parser.add_argument("--s2_path", required=True)
    parser.add_argument("--field_csv_path", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--wavelengths_path", default=None)
    parser.add_argument("--patch_name", default=DEFAULT_PATCH_NAME)
    parser.add_argument("--align", type=int, default=DEFAULT_ALIGN)
    parser.add_argument("--margin_m", type=float, default=DEFAULT_MARGIN_M)
    parser.add_argument("--plot_side_m", type=float, default=DEFAULT_PLOT_SIDE_M)
    parser.add_argument("--roi_margin_s2_px", type=int, default=DEFAULT_ROI_MARGIN_S2_PX)
    args = parser.parse_args()
    summary = generate_dataset(
        prisma_path=args.prisma_path,
        s2_path=args.s2_path,
        field_csv_path=args.field_csv_path,
        out_root=args.out_root,
        wavelengths_path=args.wavelengths_path,
        patch_name=args.patch_name,
        align=args.align,
        margin_m=args.margin_m,
        plot_side_m=args.plot_side_m,
        roi_margin_s2_px=args.roi_margin_s2_px,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
