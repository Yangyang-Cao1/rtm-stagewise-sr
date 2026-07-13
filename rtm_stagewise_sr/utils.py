#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import json
import os
import shutil
from datetime import datetime
from typing import Any

import numpy as np


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def timestamp_string() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_run_dir(base_dir: str, label: str) -> str:
    ensure_dir(base_dir)
    run_dir = os.path.join(base_dir, f"{timestamp_string()}_{label}")
    os.makedirs(run_dir, exist_ok=False)
    return run_dir


def save_json(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def save_sort_index(path: str, sort_idx: np.ndarray) -> None:
    np.save(path, np.asarray(sort_idx, dtype=np.int64))


def append_history_csv(path: str, rows: list[dict[str, Any]]) -> None:
    if len(rows) == 0:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def copy_code_snapshot(destination_dir: str, source_files: list[str]) -> str:
    snapshot_dir = os.path.join(destination_dir, "code_snapshot")
    os.makedirs(snapshot_dir, exist_ok=True)
    for src in source_files:
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(snapshot_dir, os.path.basename(src)))
    return snapshot_dir


def select_device(device_arg: str | None = None) -> str:
    if device_arg:
        return device_arg
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
