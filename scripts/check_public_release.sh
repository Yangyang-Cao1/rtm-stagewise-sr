#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

echo "Checking for unexpectedly large files..."
large_files="$(find . -type f -not -path './.git/*' -size +10M -print)"
if [[ -n "$large_files" ]]; then
    echo "$large_files"
    echo "Release check failed: files larger than 10 MiB were found."
    exit 1
fi

echo "Checking for data and model artifacts..."
artifact_files="$(find . -type f -not -path './.git/*' \( \
    -iname '*.tif' -o -iname '*.tiff' -o -iname '*.npy' -o -iname '*.npz' -o \
    -iname '*.pt' -o -iname '*.pth' -o -iname '*.ckpt' -o -iname '*.pkl' -o \
    -iname '*.joblib' -o -iname '*.xlsx' -o -iname '*.csv' \
\) -print)"
if [[ -n "$artifact_files" ]]; then
    echo "$artifact_files"
    echo "Release check failed: data or model artifacts were found."
    exit 1
fi

echo "Checking for local absolute paths..."
if rg -n '/home/|/Users/|[A-Za-z]:\\Users\\' \
    --glob '*.py' --glob '*.md' --glob '*.sh' --glob '*.yml' --glob '*.txt' \
    --glob '!**/scripts/check_public_release.sh' .; then
    echo "Release check failed: local absolute paths were found."
    exit 1
fi

echo "Checking for common secret assignments..."
if rg -n -i '(api[_-]?key|access[_-]?token|password|client[_-]?secret)\s*[:=]\s*["'"'][^"'"']+["'"']' \
    --glob '*.py' --glob '*.json' --glob '*.yml' --glob '*.yaml' --glob '*.env' \
    --glob '!**/scripts/check_public_release.sh' .; then
    echo "Release check failed: a possible embedded secret was found."
    exit 1
fi

echo "Public-release checks passed."
