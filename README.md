# RTM Stagewise Hyperspectral Super-Resolution

Source-only implementation of a stagewise hyperspectral super-resolution
workflow using low-resolution HSI, high-resolution Sentinel-2 MSI, field
spectra, and a PROSAIL-based RTM teacher.

Satellite imagery, field observations, model weights, generated patches, and
experiment results are not included.

## Method

1. Train a no-RTM reconstruction to establish spatial detail.
2. Construct a conservative low-frequency vegetation baseline.
3. Train an RTM-constrained spectral teacher.
4. Learn wavelength-dependent fusion between baseline and teacher.
5. Apply light segment-aware spectral smoothing.


## Code layout

```text
rtm_stagewise_sr/
├── pipeline.py          # complete single-dataset workflow
├── batch.py             # unified PRISMA/EnMAP batch discovery
├── training.py          # Stage 1 and Stage 2 optimization
├── data.py              # array loading and wavelength ordering
├── operators.py         # spatial and spectral forward operators
├── losses.py            # reconstruction and regularization losses
├── anchor.py            # RTM projection utilities
├── evaluation.py        # field-spectrum evaluation
├── plots.py             # field-spectrum plots
├── visualization.py     # image and band visualization
├── preprocessing/
│   ├── prepare.py       # field CSV to area-weighted onepatch dataset
│   └── geometry.py      # geometry and normalization primitives
└── rtm/
    ├── folding.py       # target-wavelength folding matrix
    ├── surrogate.py     # PROSAIL surrogate training
    ├── export.py        # TorchScript export
    └── teacher.py       # RTM-only teacher reconstruction
```

There is one batch entry point for both supported sensor layouts. Historical
sweep scripts and experiment-specific duplicate entry points are excluded.

## Installation

Python 3.10 is recommended.

```bash
conda env create -f environment.yml
conda activate rtm-stagewise-sr
```

If necessary, install the PyTorch build matching the local CUDA driver
separately.

## Local inputs

Put private inputs anywhere outside Git. The default local locations are:

```text
datasets_crop/                    # satellite and field datasets
artifacts/
├── InputPROSAIL_params.csv
├── PROSAIL_reflectance.csv
└── Sentinel2SRF2024-4.0.xlsx
```

Both directories are ignored. Override them with environment variables:

```bash
export RTM_SR_DATASETS_ROOT=/path/to/datasets_crop
export RTM_SR_ARTIFACTS_ROOT=/path/to/artifacts
export RTM_SR_PROSAIL_PARAMS=/path/to/InputPROSAIL_params.csv
export RTM_SR_PROSAIL_SPECTRA=/path/to/PROSAIL_reflectance.csv
export RTM_SR_S2_SRF_XLSX=/path/to/Sentinel2SRF2024-4.0.xlsx
```

Supported layouts are:

```text
datasets_crop/<name>/project_B4/     # PRISMA layout
datasets_crop/<name>/                # EnMAP single-date layout
```

The batch command detects the layout automatically.

## Commands

Prepare one dataset:

```bash
python -m rtm_stagewise_sr.preprocessing.prepare \
  --prisma_path /path/to/hsi.tif \
  --s2_path /path/to/s2.tif \
  --field_csv_path /path/to/field.csv \
  --wavelengths_path /path/to/wavelengths.npy \
  --out_root /path/to/generated_dataset
```

Run one prepared dataset:

```bash
python -m rtm_stagewise_sr.pipeline \
  --dataset_name example \
  --dataset_root /path/to/generated_dataset \
  --prisma_path /path/to/ALL_POINTS_prisma.npy \
  --s2_path /path/to/ALL_POINTS_s2.npy \
  --meta_json /path/to/ALL_POINTS.json \
  --field_csv /path/to/field.csv \
  --wavelengths_path /path/to/wavelengths.npy \
  --rtm_inv_path /path/to/inv_net.pt \
  --rtm_fwd_path /path/to/fwd_net.pt \
  --rtm_scalers_path /path/to/scalers.pkl \
  --srf_xlsx /path/to/Sentinel2SRF2024-4.0.xlsx \
  --output_base /path/to/output
```

Auto-discover and run every supported dataset:

```bash
python -m rtm_stagewise_sr.batch
```

Run selected dataset directories only:

```bash
python -m rtm_stagewise_sr.batch dataset_a dataset_b
```

## License

This project is released under the [MIT License](LICENSE).
