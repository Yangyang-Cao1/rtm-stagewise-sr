#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
from typing import Any

import numpy as np
import pandas as pd

from .data import DEFAULT_WL_PATH, ensure_hwc


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)
META_JSON = os.path.join(REPO_ROOT, "datasets_crop/dataset_onepatch/meta/ALL_POINTS.json")
FIELD_CSV = os.path.normpath(os.path.join(BASE_DIR, "../datasets_crop/field_measurements.csv"))
LEGACY_NO_RTM_PATH = os.path.normpath(os.path.join(BASE_DIR, "../datasets_crop/legacy_no_rtm.npy"))
NIR_MIN_NM = 760.0
NIR_MAX_NM = 1300.0


def insert_nan_for_gaps(
    wavelengths_nm: np.ndarray,
    spectrum: np.ndarray,
    gap_threshold: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    gap = np.diff(wavelengths_nm)
    breaks = np.where(gap > gap_threshold)[0]
    wl_plot = wavelengths_nm.copy()
    spec_plot = spectrum.copy()
    offset = 0
    for break_idx in breaks:
        insert_at = break_idx + 1 + offset
        x_mid = 0.5 * (wl_plot[insert_at - 1] + wl_plot[insert_at])
        wl_plot = np.insert(wl_plot, insert_at, x_mid)
        spec_plot = np.insert(spec_plot, insert_at, np.nan)
        offset += 1
    return wl_plot, spec_plot


def _interp_to_target(wl_src: np.ndarray, y_src: np.ndarray, wl_tgt: np.ndarray) -> np.ndarray:
    valid = np.isfinite(wl_src) & np.isfinite(y_src)
    if valid.sum() < 2:
        return np.full_like(wl_tgt, np.nan, dtype=np.float32)
    return np.interp(wl_tgt, wl_src[valid], y_src[valid], left=np.nan, right=np.nan).astype(np.float32)


def _load_points(meta_json_path: str = META_JSON) -> list[dict[str, Any]]:
    with open(meta_json_path, "r", encoding="utf-8") as handle:
        return json.load(handle)["points"]


def _read_field_csv(field_csv_path: str) -> tuple[np.ndarray, list[dict[str, Any]]]:
    frame = pd.read_csv(field_csv_path, header=None)
    header_numeric = pd.to_numeric(frame.iloc[0, :], errors="coerce")
    band_cols = np.where(np.isfinite(header_numeric.to_numpy()))[0]
    if band_cols.size < 2:
        raise ValueError(f"Could not infer spectral columns from field CSV: {field_csv_path}")

    wl_field_src = header_numeric.iloc[band_cols].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    for row_idx in range(1, len(frame)):
        row = frame.iloc[row_idx, :]
        plot_id = None
        if frame.shape[1] > 8:
            value = row.iloc[8]
            if pd.notna(value):
                text = str(value).strip()
                plot_id = text if text else None
        lat = float(pd.to_numeric(row.iloc[6], errors="coerce")) if frame.shape[1] > 6 else float("nan")
        lon = float(pd.to_numeric(row.iloc[7], errors="coerce")) if frame.shape[1] > 7 else float("nan")
        spectrum = pd.to_numeric(row.iloc[band_cols], errors="coerce").to_numpy(dtype=float)
        valid_spec = np.isfinite(spectrum).sum() >= 2 and not np.all(np.isnan(spectrum))
        rows.append(
            {
                "field_csv_row_index": int(row_idx),
                "field_plot_id": plot_id,
                "lat": lat,
                "lon": lon,
                "spectrum": spectrum,
                "valid_spectrum": bool(valid_spec),
            }
        )
    return wl_field_src.astype(np.float32), rows


def _align_points_with_field_rows(
    points: list[dict[str, Any]],
    field_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows_by_index = {int(row["field_csv_row_index"]): row for row in field_rows}
    aligned_points: list[dict[str, Any]] = []
    aligned_rows: list[dict[str, Any]] = []

    if points and all("field_csv_row_index" in point for point in points):
        for point in points:
            row = rows_by_index.get(int(point["field_csv_row_index"]))
            if row is None or not bool(row.get("valid_spectrum", False)):
                continue
            aligned_points.append(point)
            aligned_rows.append(row)
        if aligned_points:
            return aligned_points, aligned_rows

    valid_rows = [row for row in field_rows if bool(row.get("valid_spectrum", False))]
    n = min(len(points), len(valid_rows))
    return list(points[:n]), valid_rows[-n:]


def _load_field_spectra(
    wavelengths_sorted_nm: np.ndarray,
    points: list[dict[str, Any]],
    field_csv_path: str = FIELD_CSV,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    wl_field_src, field_rows = _read_field_csv(field_csv_path)
    aligned_points, aligned_rows = _align_points_with_field_rows(points, field_rows)
    if len(aligned_points) == 0:
        raise ValueError(f"No valid field spectra could be aligned from CSV: {field_csv_path}")
    field_interp = np.stack(
        [_interp_to_target(wl_field_src, row["spectrum"], wavelengths_sorted_nm) for row in aligned_rows],
        axis=0,
    )
    return aligned_points, field_interp.astype(np.float32)


def _extract_point_spectra(cube_hwc: np.ndarray, points: list[dict[str, Any]]) -> np.ndarray:
    spectra = []
    for point in points:
        footprint = point.get("s2_plot_pixels")
        if isinstance(footprint, list) and len(footprint) > 0:
            acc = np.zeros((cube_hwc.shape[-1],), dtype=np.float32)
            total = 0.0
            for pixel in footprint:
                row, col = pixel["rc_in_patch"]
                if 0 <= int(row) < cube_hwc.shape[0] and 0 <= int(col) < cube_hwc.shape[1]:
                    weight = float(pixel.get("weight", 0.0))
                    acc += weight * cube_hwc[int(row), int(col), :].astype(np.float32)
                    total += weight
            if total > 0.0:
                spectra.append((acc / total).astype(np.float32))
                continue
        row, col = point["s2_rc_in_patch"]
        spectra.append(cube_hwc[int(row), int(col), :].astype(np.float32))
    return np.stack(spectra, axis=0).astype(np.float32)


def collect_40point_data(
    checkpoint_path: str,
    wavelengths_path: str = DEFAULT_WL_PATH,
    meta_json_path: str = META_JSON,
    field_csv_path: str = FIELD_CSV,
    legacy_no_rtm_path: str = LEGACY_NO_RTM_PATH,
) -> dict[str, Any]:
    wavelengths_raw = np.load(wavelengths_path).astype(np.float32).reshape(-1)
    sort_idx = np.argsort(wavelengths_raw)
    wavelengths_sorted = wavelengths_raw[sort_idx]
    points = _load_points(meta_json_path=meta_json_path)
    points, field_specs = _load_field_spectra(
        wavelengths_sorted,
        points=points,
        field_csv_path=field_csv_path,
    )

    pred_cube = ensure_hwc(np.load(checkpoint_path), expected_channels=wavelengths_sorted.shape[0], name="pred")
    legacy_specs = None
    if legacy_no_rtm_path and os.path.exists(legacy_no_rtm_path):
        legacy_cube = ensure_hwc(np.load(legacy_no_rtm_path), expected_channels=wavelengths_sorted.shape[0], name="legacy")
        legacy_cube = legacy_cube[..., sort_idx]
        legacy_specs = _extract_point_spectra(legacy_cube, points)

    pred_specs = _extract_point_spectra(pred_cube, points)
    return {
        "points": points,
        "wl_sorted": wavelengths_sorted,
        "field_specs": field_specs,
        "legacy_no_rtm_specs": legacy_specs,
        "pred_specs": pred_specs,
        "meta_json_path": os.path.abspath(meta_json_path),
        "field_csv_path": os.path.abspath(field_csv_path),
        "legacy_no_rtm_path": (os.path.abspath(legacy_no_rtm_path) if legacy_no_rtm_path else None),
    }


def _mae(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        pred = pred[:, mask]
        target = target[:, mask]
    return float(np.mean(np.abs(pred - target)))


def _bias(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        pred = pred[:, mask]
        target = target[:, mask]
    return float(np.mean(pred - target))


def _sam_mean_rad(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        pred = pred[:, mask]
        target = target[:, mask]
    dot = np.sum(pred * target, axis=1)
    pred_norm = np.sqrt(np.sum(pred * pred, axis=1) + 1e-12)
    target_norm = np.sqrt(np.sum(target * target, axis=1) + 1e-12)
    cosine = np.clip(dot / (pred_norm * target_norm + 1e-12), -1.0, 1.0)
    return float(np.mean(np.arccos(cosine)))


def _second_diff_energy(pred: np.ndarray) -> float:
    first_diff = pred[:, 1:] - pred[:, :-1]
    second_diff = first_diff[:, 1:] - first_diff[:, :-1]
    return float(np.mean(second_diff ** 2))


def evaluate_prediction(
    checkpoint_path: str,
    consistency_summary: dict[str, Any] | None = None,
    save_path: str | None = None,
    wavelengths_path: str = DEFAULT_WL_PATH,
    meta_json_path: str = META_JSON,
    field_csv_path: str = FIELD_CSV,
    legacy_no_rtm_path: str = LEGACY_NO_RTM_PATH,
    nir_min_nm: float = NIR_MIN_NM,
    nir_max_nm: float = NIR_MAX_NM,
) -> dict[str, Any]:
    data = collect_40point_data(
        checkpoint_path=checkpoint_path,
        wavelengths_path=wavelengths_path,
        meta_json_path=meta_json_path,
        field_csv_path=field_csv_path,
        legacy_no_rtm_path=legacy_no_rtm_path,
    )
    pred = data["pred_specs"]
    field = data["field_specs"]
    wavelengths_sorted = data["wl_sorted"]
    nir_mask = (wavelengths_sorted >= float(nir_min_nm)) & (wavelengths_sorted <= float(nir_max_nm))
    non_nir_mask = ~nir_mask

    metrics = {
        "field_mae_allbands": _mae(pred, field),
        "field_mae_nir": _mae(pred, field, mask=nir_mask),
        "field_mae_non_nir": _mae(pred, field, mask=non_nir_mask),
        "field_bias_nir": _bias(pred, field, mask=nir_mask),
        "field_sam_allbands": _sam_mean_rad(pred, field),
        "field_sam_nir": _sam_mean_rad(pred, field, mask=nir_mask),
        "field_sam_unit": "rad",
        "osc_2diff_energy": _second_diff_energy(pred),
        "checkpoint": os.path.abspath(checkpoint_path),
        "n_points": int(pred.shape[0]),
        "n_bands": int(pred.shape[1]),
        "nir_range_nm": [float(nir_min_nm), float(nir_max_nm)],
        "nir_active_bands": int(nir_mask.sum()),
        "meta_json_path": os.path.abspath(meta_json_path),
        "field_csv_path": os.path.abspath(field_csv_path),
    }
    if consistency_summary:
        if "hs_mse" in consistency_summary:
            metrics["hs_reproj"] = float(consistency_summary["hs_mse"])
        if "ms_mse" in consistency_summary:
            metrics["ms_reproj"] = float(consistency_summary["ms_mse"])
        for key in [
            "stage1_end_iter",
            "stage1_stable_detected",
            "stage1_stability_rule",
            "best_iter",
            "last_iter",
            "result_iter_used_for_metrics",
            "result_iter_used_for_plot",
            "result_iter_used_for_array_output",
            "selection_interval",
            "min_select_iter",
            "total_best_global_iter",
        ]:
            if key in consistency_summary:
                metrics[key] = consistency_summary[key]
    if save_path is not None:
        with open(save_path, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, ensure_ascii=False, indent=2)
    return metrics
