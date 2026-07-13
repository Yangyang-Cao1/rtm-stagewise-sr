#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import DEFAULT_S2_PATH, DEFAULT_WL_PATH, ensure_hwc


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)
DEFAULT_LEGACY_NO_RTM_PATH = os.path.normpath(os.path.join(BASE_DIR, "../datasets_crop/legacy_no_rtm.npy"))
DEFAULT_META_JSON = os.path.join(REPO_ROOT, "datasets_crop/dataset_onepatch/meta/ALL_POINTS.json")


def _valid_stretch_values(array: np.ndarray) -> np.ndarray:
    values = array[np.isfinite(array)]
    if values.size == 0:
        return values
    nodata_mask = values <= -1.0e4
    if np.any(nodata_mask) and np.any(~nodata_mask):
        values = values[~nodata_mask]
    return values


def _stretch(image_hwc: np.ndarray, pmin: float = 2.0, pmax: float = 98.0) -> np.ndarray:
    output = np.zeros_like(image_hwc, dtype=np.float32)
    for channel_idx in range(image_hwc.shape[-1]):
        channel = image_hwc[..., channel_idx].astype(np.float32)
        valid_values = _valid_stretch_values(channel)
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
        stretched = np.clip((channel - lo) / (hi - lo + 1e-12), 0.0, 1.0)
        stretched[~np.isfinite(channel)] = 0.0
        stretched[channel <= -1.0e4] = 0.0
        output[..., channel_idx] = stretched
    return output


def _stretch_single(band_hw: np.ndarray, pmin: float = 2.0, pmax: float = 98.0) -> np.ndarray:
    band_hw = band_hw.astype(np.float32)
    valid_values = _valid_stretch_values(band_hw)
    if valid_values.size == 0:
        return np.zeros_like(band_hw, dtype=np.float32)
    lo, hi = np.percentile(valid_values, [pmin, pmax])
    if not np.isfinite(lo) or not np.isfinite(hi):
        return np.zeros_like(band_hw, dtype=np.float32)
    if hi <= lo:
        hi = float(np.max(valid_values))
        lo = float(np.min(valid_values))
        if hi <= lo:
            return np.zeros_like(band_hw, dtype=np.float32)
    stretched = np.clip((band_hw - lo) / (hi - lo + 1e-12), 0.0, 1.0)
    stretched[~np.isfinite(band_hw)] = 0.0
    stretched[band_hw <= -1.0e4] = 0.0
    return stretched


def _find_band_idx(wavelengths_nm: np.ndarray, target_nm: float) -> int:
    return int(np.argmin(np.abs(wavelengths_nm.astype(np.float64) - float(target_nm))))


def _infer_hsi_sensor_label(source_hsi_path: str | None) -> str:
    path_text = (source_hsi_path or "").lower()
    if "enmap" in path_text:
        return "EnMAP"
    if "prisma" in path_text:
        return "PRISMA"
    return "HSI"


def _crop_center(image_hwc: np.ndarray, roi_size: int) -> tuple[np.ndarray, list[int]]:
    height, width = image_hwc.shape[:2]
    roi_size = int(max(16, roi_size))
    center_y, center_x = height // 2, width // 2
    y0 = max(0, center_y - roi_size // 2)
    x0 = max(0, center_x - roi_size // 2)
    y1 = min(height, y0 + roi_size)
    x1 = min(width, x0 + roi_size)
    return image_hwc[y0:y1, x0:x1, ...], [int(y0), int(x0), int(y1), int(x1)]


def _crop_bbox(image_hwc: np.ndarray, roi_bbox: list[int]) -> np.ndarray:
    y0, x0, y1, x1 = [int(v) for v in roi_bbox]
    return image_hwc[y0:y1, x0:x1, ...]


def _map_roi_between_grids(
    roi_bbox_src: list[int],
    src_shape_hw: tuple[int, int],
    dst_shape_hw: tuple[int, int],
) -> list[int]:
    y0, x0, y1, x1 = [int(v) for v in roi_bbox_src]
    src_h, src_w = int(src_shape_hw[0]), int(src_shape_hw[1])
    dst_h, dst_w = int(dst_shape_hw[0]), int(dst_shape_hw[1])
    scale_y = float(dst_h) / float(max(src_h, 1))
    scale_x = float(dst_w) / float(max(src_w, 1))
    dy0 = int(np.floor(y0 * scale_y))
    dx0 = int(np.floor(x0 * scale_x))
    dy1 = int(np.ceil(y1 * scale_y))
    dx1 = int(np.ceil(x1 * scale_x))
    dy0 = min(max(0, dy0), max(0, dst_h - 1))
    dx0 = min(max(0, dx0), max(0, dst_w - 1))
    dy1 = min(max(dy0 + 1, dy1), dst_h)
    dx1 = min(max(dx0 + 1, dx1), dst_w)
    return [int(dy0), int(dx0), int(dy1), int(dx1)]


def _field_points_roi(meta_json_path: str, image_shape: tuple[int, int, int], roi_size: int) -> tuple[list[dict], list[int]] | tuple[None, None]:
    if not meta_json_path or not os.path.exists(meta_json_path):
        return None, None
    with open(meta_json_path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    points = meta.get("points", [])
    roi_bbox = meta.get("field_points_roi_s2_rc")
    if isinstance(roi_bbox, list) and len(roi_bbox) == 4:
        return points, [int(v) for v in roi_bbox]

    rows: list[int] = []
    cols: list[int] = []
    for point in points:
        footprint = point.get("s2_plot_pixels")
        if isinstance(footprint, list) and len(footprint) > 0:
            rows.extend(int(pixel["rc_in_patch"][0]) for pixel in footprint)
            cols.extend(int(pixel["rc_in_patch"][1]) for pixel in footprint)
        elif "s2_rc_in_patch" in point:
            row, col = point["s2_rc_in_patch"]
            rows.append(int(row))
            cols.append(int(col))
    if not rows or not cols:
        return points, None
    height, width = image_shape[:2]
    margin = int(max(16, roi_size // 4))
    min_row = max(0, min(rows) - margin)
    max_row = min(height - 1, max(rows) + margin)
    min_col = max(0, min(cols) - margin)
    max_col = min(width - 1, max(cols) + margin)
    box_h = max_row - min_row + 1
    box_w = max_col - min_col + 1
    size = max(box_h, box_w, int(max(roi_size, 16)))
    center_row = 0.5 * (min_row + max_row)
    center_col = 0.5 * (min_col + max_col)
    y0 = int(round(center_row - 0.5 * size))
    x0 = int(round(center_col - 0.5 * size))
    y0 = min(max(0, y0), max(0, height - size))
    x0 = min(max(0, x0), max(0, width - size))
    y1 = min(height, y0 + size)
    x1 = min(width, x0 + size)
    return points, [int(y0), int(x0), int(y1), int(x1)]


def visualize_sr(
    pred_path: str,
    output_dir: str,
    wavelengths_path: str = DEFAULT_WL_PATH,
    ms_path: str = DEFAULT_S2_PATH,
    prisma_path: str | None = None,
    legacy_no_rtm_path: str = DEFAULT_LEGACY_NO_RTM_PATH,
    comparison_path: str | None = None,
    comparison_label: str = "Reference",
    meta_json_path: str = DEFAULT_META_JSON,
    experiment_name: str = "stagewise",
    stage_name: str = "stage1",
    iter_value: int | None = None,
    use_rtm: bool = False,
    keep_ms_spectral: bool = True,
    roi_size: int = 96,
    roi_mode: str = "field_points",
) -> dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)

    wavelengths_nm = np.load(wavelengths_path).astype(np.float32).reshape(-1)
    sort_idx = np.argsort(wavelengths_nm)
    wavelengths_sorted = wavelengths_nm[sort_idx]
    pred_hwc = ensure_hwc(np.load(pred_path), expected_channels=wavelengths_sorted.shape[0], name="pred")
    ms_hwc = ensure_hwc(np.load(ms_path), expected_channels=9, name="ms")
    meta = None
    if meta_json_path and os.path.exists(meta_json_path):
        with open(meta_json_path, "r", encoding="utf-8") as handle:
            meta = json.load(handle)
    source_hsi_path = None if meta is None else (meta.get("source_hsi_path") or meta.get("source_prisma_path"))
    hsi_panel_title = f"{_infer_hsi_sensor_label(source_hsi_path)} RGB"
    prisma_rgb_hwc = None
    prisma_points_meta = None
    prisma_roi_bbox = None
    if prisma_path and os.path.exists(prisma_path):
        prisma_hwc = ensure_hwc(np.load(prisma_path), expected_channels=wavelengths_sorted.shape[0], name="prisma")
        prisma_hwc = prisma_hwc[..., sort_idx]
        idx_pr_r = _find_band_idx(wavelengths_sorted, 664.5)
        idx_pr_g = _find_band_idx(wavelengths_sorted, 560.0)
        idx_pr_b = _find_band_idx(wavelengths_sorted, 496.6)
        prisma_rgb_hwc = _stretch(prisma_hwc[..., [idx_pr_r, idx_pr_g, idx_pr_b]])
    reference_hwc = None
    reference_label = None
    reference_path = comparison_path
    if reference_path is None and legacy_no_rtm_path and os.path.exists(legacy_no_rtm_path):
        reference_path = legacy_no_rtm_path
        reference_label = "Legacy no-RTM"
    elif reference_path is not None:
        reference_label = comparison_label
    if reference_path is not None and os.path.exists(reference_path):
        legacy_hwc = ensure_hwc(
            np.load(reference_path),
            expected_channels=wavelengths_sorted.shape[0],
            name="reference",
        )[..., sort_idx]
        reference_hwc = legacy_hwc

    idx_r = _find_band_idx(wavelengths_sorted, 664.5)
    idx_g = _find_band_idx(wavelengths_sorted, 560.0)
    idx_b = _find_band_idx(wavelengths_sorted, 496.6)
    idx_nir = _find_band_idx(wavelengths_sorted, 842.0)

    panels: list[tuple[str, np.ndarray]] = []
    if prisma_rgb_hwc is not None:
        panels.append((hsi_panel_title, prisma_rgb_hwc))
    panels.append(("Sentinel-2 RGB", _stretch(ms_hwc[..., [2, 1, 0]])))
    if reference_hwc is not None:
        panels.append((reference_label or "Reference", _stretch(reference_hwc[..., [idx_r, idx_g, idx_b]])))
    panels.append(("Current result", _stretch(pred_hwc[..., [idx_r, idx_g, idx_b]])))

    points_meta = None
    roi_bbox = None
    if roi_mode == "field_points":
        points_meta, roi_bbox = _field_points_roi(meta_json_path, pred_hwc.shape, roi_size=roi_size)
    if roi_bbox is None:
        _, roi_bbox = _crop_center(pred_hwc, roi_size=roi_size)
    if prisma_rgb_hwc is not None:
        prisma_points_meta = points_meta
        prisma_roi_meta = (meta or {}).get("field_points_roi_prisma_rc")
        if isinstance(prisma_roi_meta, list) and len(prisma_roi_meta) == 4 and roi_mode == "field_points":
            prisma_roi_bbox = [int(v) for v in prisma_roi_meta]
        else:
            prisma_roi_bbox = _map_roi_between_grids(
                roi_bbox,
                src_shape_hw=pred_hwc.shape[:2],
                dst_shape_hw=prisma_rgb_hwc.shape[:2],
            )

    iter_text = "NA" if iter_value is None else str(iter_value)
    title = (
        f"{experiment_name} | {stage_name} | iter={iter_text} | "
        f"use_rtm={use_rtm} | keep_ms_stage2={keep_ms_spectral}"
    )

    figure, axes = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 4.8))
    if len(panels) == 1:
        axes = [axes]
    for axis, (panel_title, image_hwc) in zip(axes, panels):
        axis.imshow(image_hwc)
        panel_roi_bbox = prisma_roi_bbox if panel_title == hsi_panel_title else roi_bbox
        if panel_roi_bbox is not None:
            y0, x0, y1, x1 = panel_roi_bbox
            axis.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="cyan", linewidth=1.8))
        if points_meta:
            for point in points_meta:
                rc_key = "prisma_rc_in_patch" if panel_title == hsi_panel_title else "s2_rc_in_patch"
                row, col = point.get(rc_key, [None, None])
                if row is not None and col is not None:
                    axis.scatter([col], [row], s=12, c="yellow", edgecolors="black", linewidths=0.3)
        axis.set_title(panel_title)
        axis.axis("off")
    figure.suptitle(title, fontsize=12)
    figure.tight_layout(rect=[0, 0, 1, 0.95])
    rgb_overview_path = os.path.join(output_dir, "rgb_overview.png")
    figure.savefig(rgb_overview_path, dpi=220)
    plt.close(figure)

    figure, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 4.2))
    if len(panels) == 1:
        axes = [axes]
    for axis, (panel_title, image_hwc) in zip(axes, panels):
        panel_roi_bbox = prisma_roi_bbox if panel_title == hsi_panel_title else roi_bbox
        axis.imshow(_crop_bbox(image_hwc, panel_roi_bbox))
        axis.set_title(f"{panel_title} ROI")
        axis.axis("off")
    figure.suptitle(f"{title} | ROI", fontsize=12)
    figure.tight_layout(rect=[0, 0, 1, 0.94])
    roi_path = os.path.join(output_dir, "rgb_roi_zoom.png")
    figure.savefig(roi_path, dpi=220)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3 if reference_hwc is not None else 2, figsize=(4.0 * (3 if reference_hwc is not None else 2), 4.2))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    band_panels = [
        ("Sentinel-2 B8", _stretch_single(ms_hwc[..., 6])),
    ]
    if reference_hwc is not None:
        band_panels.append((reference_label or "Reference", _stretch_single(reference_hwc[..., idx_nir])))
    band_panels.append(("Current result", _stretch_single(pred_hwc[..., idx_nir])))
    for axis, (panel_title, band_hw) in zip(axes, band_panels):
        axis.imshow(band_hw, cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(panel_title)
        axis.axis("off")
    figure.suptitle(f"{title} | NIR {wavelengths_sorted[idx_nir]:.1f}nm", fontsize=12)
    figure.tight_layout(rect=[0, 0, 1, 0.94])
    nir_path = os.path.join(output_dir, "band_compare_nir.png")
    figure.savefig(nir_path, dpi=220)
    plt.close(figure)

    info = {
        "rgb_overview": rgb_overview_path,
        "rgb_roi_zoom": roi_path,
        "band_compare_nir": nir_path,
        "title": title,
        "roi_mode": roi_mode,
        "roi_bbox_y0x0y1x1": roi_bbox,
        "prisma_roi_bbox_y0x0y1x1": prisma_roi_bbox,
        "hsi_panel_title": hsi_panel_title,
        "source_hsi_path": source_hsi_path,
        "prisma_path": (os.path.abspath(prisma_path) if prisma_path else None),
        "meta_json_path": (os.path.abspath(meta_json_path) if meta_json_path else None),
    }
    with open(os.path.join(output_dir, "visualize_info.json"), "w", encoding="utf-8") as handle:
        json.dump(info, handle, ensure_ascii=False, indent=2)
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a reconstructed SR HSI result.")
    parser.add_argument("--pred_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--wavelengths_path", default=DEFAULT_WL_PATH)
    parser.add_argument("--ms_path", default=DEFAULT_S2_PATH)
    parser.add_argument("--prisma_path", default=None)
    parser.add_argument("--legacy_no_rtm_path", default=DEFAULT_LEGACY_NO_RTM_PATH)
    parser.add_argument("--comparison_path", default=None)
    parser.add_argument("--comparison_label", default="Reference")
    parser.add_argument("--meta_json_path", default=DEFAULT_META_JSON)
    parser.add_argument("--experiment_name", default="stagewise")
    parser.add_argument("--stage_name", default="stage1")
    parser.add_argument("--iter_value", type=int, default=None)
    parser.add_argument("--use_rtm", action="store_true")
    parser.add_argument("--no_keep_ms_spectral", action="store_true")
    parser.add_argument("--roi_size", type=int, default=96)
    parser.add_argument("--roi_mode", choices=["field_points", "center"], default="field_points")
    args = parser.parse_args()

    output = visualize_sr(
        pred_path=args.pred_path,
        output_dir=args.output_dir,
        wavelengths_path=args.wavelengths_path,
        ms_path=args.ms_path,
        prisma_path=args.prisma_path,
        legacy_no_rtm_path=args.legacy_no_rtm_path,
        comparison_path=args.comparison_path,
        comparison_label=args.comparison_label,
        meta_json_path=args.meta_json_path,
        experiment_name=args.experiment_name,
        stage_name=args.stage_name,
        iter_value=args.iter_value,
        use_rtm=bool(args.use_rtm),
        keep_ms_spectral=(not bool(args.no_keep_ms_spectral)),
        roi_size=args.roi_size,
        roi_mode=args.roi_mode,
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
