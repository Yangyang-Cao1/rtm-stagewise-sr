#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .evaluation import collect_40point_data, insert_nan_for_gaps


def plot_40points(
    checkpoint_path: str,
    run_dir: str,
    title: str,
    output_basename: str = "spectra_40points.png",
    wavelengths_path: str | None = None,
    meta_json_path: str | None = None,
    field_csv_path: str | None = None,
    legacy_no_rtm_path: str | None = None,
    legacy_label: str = "Legacy no-RTM",
) -> str:
    os.makedirs(run_dir, exist_ok=True)
    kwargs = {}
    if wavelengths_path is not None:
        kwargs["wavelengths_path"] = wavelengths_path
    if meta_json_path is not None:
        kwargs["meta_json_path"] = meta_json_path
    if field_csv_path is not None:
        kwargs["field_csv_path"] = field_csv_path
    if legacy_no_rtm_path is not None:
        kwargs["legacy_no_rtm_path"] = legacy_no_rtm_path
    data = collect_40point_data(checkpoint_path=checkpoint_path, **kwargs)
    points = data["points"]
    wl_sorted = data["wl_sorted"]
    field_specs = data["field_specs"]
    legacy_specs = data["legacy_no_rtm_specs"]
    pred_specs = data["pred_specs"]
    n_points = len(points)

    ncols = min(5, max(1, n_points))
    nrows = int(np.ceil(float(n_points) / float(ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.5 * nrows), sharex=True, sharey=False)
    axes = np.atleast_1d(axes).ravel()
    for idx in range(n_points):
        axis = axes[idx]
        wl_plot, field_plot = insert_nan_for_gaps(wl_sorted, field_specs[idx], gap_threshold=30.0)
        axis.plot(wl_plot, field_plot, label="Field", color="#1f77b4", linewidth=1.5, zorder=3)
        wl_plot, pred_plot = insert_nan_for_gaps(wl_sorted, pred_specs[idx], gap_threshold=30.0)
        axis.plot(
            wl_plot,
            pred_plot,
            label="Current result",
            color="#2ca02c",
            linewidth=1.8,
            alpha=0.9,
            zorder=2,
        )
        if legacy_specs is not None:
            wl_plot, legacy_plot = insert_nan_for_gaps(wl_sorted, legacy_specs[idx], gap_threshold=30.0)
            axis.plot(
                wl_plot,
                legacy_plot,
                label=legacy_label,
                color="#ff7f0e",
                linewidth=1.4,
                linestyle="--",
                marker="o",
                markersize=1.8,
                markevery=max(1, int(len(wl_sorted) / 24)),
                zorder=4,
            )
        point_title = points[idx].get("id", f"pt{idx:02d}")
        if legacy_specs is not None:
            overlap = np.nanmax(np.abs(legacy_specs[idx] - pred_specs[idx]))
            if float(overlap) < 1e-8:
                point_title = f"{point_title} | overlap"
        axis.set_title(point_title, fontsize=9)
        axis.grid(alpha=0.3)
    for idx in range(n_points, len(axes)):
        axes[idx].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=11)
    fig.suptitle(title, fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    output_path = os.path.join(run_dir, output_basename)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot 40-point spectra for a prediction.")
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--output_basename", default="spectra_40points.png")
    parser.add_argument("--wavelengths_path", default=None)
    parser.add_argument("--meta_json_path", default=None)
    parser.add_argument("--field_csv_path", default=None)
    parser.add_argument("--legacy_no_rtm_path", default=None)
    parser.add_argument("--legacy_label", default="Legacy no-RTM")
    args = parser.parse_args()
    print(
        plot_40points(
            args.checkpoint_path,
            args.run_dir,
            args.title,
            output_basename=args.output_basename,
            wavelengths_path=args.wavelengths_path,
            meta_json_path=args.meta_json_path,
            field_csv_path=args.field_csv_path,
            legacy_no_rtm_path=args.legacy_no_rtm_path,
            legacy_label=args.legacy_label,
        )
    )


if __name__ == "__main__":
    main()
