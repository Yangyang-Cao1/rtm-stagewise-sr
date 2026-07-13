#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass

import numpy as np


PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PACKAGE_DIR)
ROOT = os.path.abspath(
    os.environ.get("RTM_SR_DATASETS_ROOT", os.path.join(REPO_ROOT, "datasets_crop"))
)
ARTIFACTS_DIR = os.path.abspath(
    os.environ.get("RTM_SR_ARTIFACTS_ROOT", os.path.join(REPO_ROOT, "artifacts"))
)
PYTHON = sys.executable

PROSAIL_PARAMS = os.path.abspath(
    os.environ.get("RTM_SR_PROSAIL_PARAMS", os.path.join(ARTIFACTS_DIR, "InputPROSAIL_params.csv"))
)
PROSAIL_SPECTRA = os.path.abspath(
    os.environ.get("RTM_SR_PROSAIL_SPECTRA", os.path.join(ARTIFACTS_DIR, "PROSAIL_reflectance.csv"))
)
S2_SRF_XLSX = os.path.abspath(
    os.environ.get("RTM_SR_S2_SRF_XLSX", os.path.join(ARTIFACTS_DIR, "Sentinel2SRF2024-4.0.xlsx"))
)

SURROGATE_EPOCHS = 180
GENERATED_DATASET_DIRNAME = "dataset_onepatch_centered_plot15_areaweighted_v2"
OUTPUT_RUNS_DIRNAME = "runs_stagewise_rtm"
RUN_SUFFIX = "stagewise_rtm"
MANIFEST_NAME = "batch_manifest.json"


@dataclass
class DatasetSpec:
    name: str
    dataset_root: str
    input_root: str
    prisma_tif: str
    s2_tif: str
    field_csv: str
    wavelengths_npy: str
    generated_root: str
    output_base: str


def _s2_priority(filename: str) -> int | None:
    if not filename.endswith(".tif"):
        return None
    if filename.startswith("S2_L2A_Median_CloudFree_"):
        return 0
    if filename.startswith("S2_single_date_"):
        return 1
    if filename.startswith("S2_"):
        return 2
    return None


def _run(cmd: list[str], *, cwd: str | None = None) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd)


def _discover_datasets(selected_names: set[str] | None = None) -> list[DatasetSpec]:
    specs: list[DatasetSpec] = []
    for name in sorted(os.listdir(ROOT)):
        if name.startswith("_") or (selected_names is not None and name not in selected_names):
            continue
        dataset_root = os.path.join(ROOT, name)
        project_b4 = os.path.join(dataset_root, "project_B4")
        if not os.path.isdir(dataset_root):
            continue
        input_root = project_b4 if os.path.isdir(project_b4) else dataset_root
        hsi_tif = None
        s2_candidates: list[tuple[int, str]] = []
        field_csv_candidates: list[str] = []
        wavelengths_npy = None
        for entry in sorted(os.listdir(input_root)):
            full = os.path.join(input_root, entry)
            if os.path.isdir(full):
                continue
            if entry.endswith(".csv"):
                field_csv_candidates.append(full)
            elif entry.startswith(("prisma_filtered_bands_", "enmap_filtered_bands_")) and entry.endswith(".tif"):
                hsi_tif = full
            elif entry.startswith(("filtered_wavelengths_", "enmap_filtered_wavelengths_")) and entry.endswith(".npy"):
                wavelengths_npy = full
            else:
                s2_priority = _s2_priority(entry)
                if s2_priority is not None:
                    s2_candidates.append((s2_priority, full))
        s2_tif = sorted(s2_candidates, key=lambda item: (item[0], item[1]))[0][1] if s2_candidates else None
        field_csv = field_csv_candidates[0] if field_csv_candidates else None
        if not (hsi_tif and s2_tif and field_csv and wavelengths_npy):
            if selected_names is not None:
                raise FileNotFoundError(f"Incomplete dataset inputs for {name}: {input_root}")
            continue
        specs.append(
            DatasetSpec(
                name=name,
                dataset_root=dataset_root,
                input_root=input_root,
                prisma_tif=hsi_tif,
                s2_tif=s2_tif,
                field_csv=field_csv,
                wavelengths_npy=wavelengths_npy,
                generated_root=os.path.join(dataset_root, GENERATED_DATASET_DIRNAME),
                output_base=os.path.join(dataset_root, OUTPUT_RUNS_DIRNAME),
            )
        )
    return specs


def _wavelength_tag(wavelengths_path: str) -> str:
    wl = np.load(wavelengths_path).astype(np.float32).reshape(-1)
    digest = hashlib.md5(wl.tobytes()).hexdigest()[:10]
    stem = os.path.splitext(os.path.basename(wavelengths_path))[0]
    return f"{stem}_b{wl.shape[0]}_{digest}"


def _prepare_surrogate(wavelengths_path: str) -> dict[str, str]:
    tag = _wavelength_tag(wavelengths_path)
    out_dir = os.path.join(ROOT, "_shared_rtm_surrogates", tag)
    fwd_ts = os.path.join(out_dir, "fwd_net.pt")
    inv_ts = os.path.join(out_dir, "inv_net.pt")
    scalers = os.path.join(out_dir, "scalers.pkl")
    if os.path.exists(fwd_ts) and os.path.exists(inv_ts) and os.path.exists(scalers):
        return {
            "tag": tag,
            "out_dir": out_dir,
            "fwd_ts": fwd_ts,
            "inv_ts": inv_ts,
            "scalers": scalers,
            "folding_csv": os.path.join(out_dir, "folding.csv"),
        }

    os.makedirs(out_dir, exist_ok=True)
    folding_csv = os.path.join(out_dir, "folding.csv")
    sort_idx_path = os.path.join(out_dir, "prisma_wl_sort_idx.npy")
    _run(
        [
            PYTHON, "-m", "rtm_stagewise_sr.rtm.folding",
            "--wl_npy", wavelengths_path,
            "--out_csv", folding_csv,
            "--save_idx", sort_idx_path,
            "--no_plot",
        ],
        cwd=REPO_ROOT,
    )

    hs_channels = int(np.load(wavelengths_path).reshape(-1).shape[0])
    _run(
        [
            PYTHON, "-m", "rtm_stagewise_sr.rtm.surrogate",
            "--params_csv", PROSAIL_PARAMS,
            "--spectra_csv", PROSAIL_SPECTRA,
            "--folding_csv", folding_csv,
            "--hs_channels", str(hs_channels),
            "--out_dir", out_dir,
            "--epochs", str(SURROGATE_EPOCHS),
        ],
        cwd=REPO_ROOT,
    )
    _run(
        [
            PYTHON, "-m", "rtm_stagewise_sr.rtm.export",
            "--ckpt_path", os.path.join(out_dir, "surrogate_models_best.pth"),
            "--out_fwd_ts", fwd_ts,
            "--out_inv_ts", inv_ts,
            "--device", "cpu",
        ],
        cwd=REPO_ROOT,
    )
    return {
        "tag": tag,
        "out_dir": out_dir,
        "fwd_ts": fwd_ts,
        "inv_ts": inv_ts,
        "scalers": scalers,
        "folding_csv": folding_csv,
    }


def _generate_dataset(spec: DatasetSpec) -> str:
    meta_json = os.path.join(spec.generated_root, "meta", "ALL_POINTS.json")
    if os.path.exists(meta_json):
        try:
            with open(meta_json, "r", encoding="utf-8") as handle:
                meta = json.load(handle)
            source_hsi_path = meta.get("source_hsi_path") or meta.get("source_prisma_path")
            scale_policy = meta.get("source_hsi_scale_policy")
            current_path = os.path.abspath(spec.prisma_tif)
            source_text = current_path.lower()
            needs_hsi_scale_meta = "enmap" in source_text
            if (
                os.path.abspath(str(source_hsi_path)) == current_path
                and ((not needs_hsi_scale_meta) or scale_policy == "divide_by_10000_if_max_gt_100")
            ):
                return meta_json
            print(f"[regen] stale or legacy onepatch detected for {spec.name}; regenerating {spec.generated_root}")
        except Exception as exc:
            print(f"[regen] could not validate existing onepatch for {spec.name}: {exc}")
    _run(
        [
            PYTHON, "-m", "rtm_stagewise_sr.preprocessing.prepare",
            "--prisma_path", spec.prisma_tif,
            "--s2_path", spec.s2_tif,
            "--field_csv_path", spec.field_csv,
            "--wavelengths_path", spec.wavelengths_npy,
            "--out_root", spec.generated_root,
        ],
        cwd=REPO_ROOT,
    )
    return meta_json


def _find_completed_run(output_base: str, meta_json: str | None = None) -> str | None:
    if not os.path.isdir(output_base):
        return None
    meta_mtime = os.path.getmtime(meta_json) if meta_json and os.path.exists(meta_json) else None
    candidates = []
    for entry in sorted(os.listdir(output_base)):
        run_dir = os.path.join(output_base, entry)
        final_selection = os.path.join(run_dir, "final_selection.json")
        if not (os.path.isdir(run_dir) and os.path.exists(final_selection)):
            continue
        if meta_mtime is not None and os.path.getmtime(final_selection) < meta_mtime:
            print(f"[stale-run] ignoring outdated completed run {run_dir} because {meta_json} is newer")
            continue
        candidates.append(run_dir)
    return candidates[-1] if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all supported PRISMA and EnMAP datasets.")
    parser.add_argument("datasets", nargs="*", help="Optional dataset directory names; default: auto-discover all")
    args = parser.parse_args()
    selected_names = set(args.datasets) if args.datasets else None
    specs = _discover_datasets(selected_names)
    if not specs:
        raise RuntimeError("No datasets discovered under datasets_crop.")

    manifest: list[dict[str, object]] = []
    surrogate_cache: dict[str, dict[str, str]] = {}
    for spec in specs:
        meta_json = _generate_dataset(spec)
        tag = _wavelength_tag(spec.wavelengths_npy)
        if tag not in surrogate_cache:
            surrogate_cache[tag] = _prepare_surrogate(spec.wavelengths_npy)
        surrogate = surrogate_cache[tag]
        completed_run = _find_completed_run(spec.output_base, meta_json)
        if completed_run is None:
            _run(
                [
                    PYTHON, "-m", "rtm_stagewise_sr.pipeline",
                    "--dataset_name", spec.name,
                    "--dataset_root", spec.generated_root,
                    "--prisma_path", os.path.join(spec.generated_root, "prisma_patches", "ALL_POINTS_prisma.npy"),
                    "--s2_path", os.path.join(spec.generated_root, "s2_patches", "ALL_POINTS_s2.npy"),
                    "--meta_json", meta_json,
                    "--field_csv", spec.field_csv,
                    "--wavelengths_path", spec.wavelengths_npy,
                    "--rtm_inv_path", surrogate["inv_ts"],
                    "--rtm_fwd_path", surrogate["fwd_ts"],
                    "--rtm_scalers_path", surrogate["scalers"],
                    "--srf_xlsx", S2_SRF_XLSX,
                    "--run_suffix", RUN_SUFFIX,
                    "--output_base", spec.output_base,
                ],
                cwd=REPO_ROOT,
            )
            completed_run = _find_completed_run(spec.output_base, meta_json)
        manifest.append(
            {
                "dataset_name": spec.name,
                "dataset_root": spec.dataset_root,
                "generated_root": spec.generated_root,
                "meta_json": meta_json,
                "wavelengths_path": spec.wavelengths_npy,
                "surrogate_tag": tag,
                "surrogate_dir": surrogate["out_dir"],
                "output_base": spec.output_base,
                "completed_run": completed_run,
            }
        )

    manifest_path = os.path.join(ROOT, MANIFEST_NAME)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
