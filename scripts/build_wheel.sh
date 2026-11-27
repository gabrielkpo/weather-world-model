#!/usr/bin/env bash
# Build the Kaggle dataset folder for install mode "wheel" of notebooks/vjepa2_qg_kaggle.ipynb.
# Output: dist/kaggle-wheel/ with the package wheel, build-info.json (code commit) and
# records/ (the committed small-JEPA step-2 records the notebook compares against).
# Usage: scripts/build_wheel.sh   (from a clean checkout of the commit to measure)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "working tree not clean: commit first, the record must name the exact code" >&2
  exit 1
fi
out=dist/kaggle-wheel
rm -rf "$out"
mkdir -p "$out/records"
uv build --wheel --out-dir "$out"
cp results/qg/qg-stationary*.json "$out/records/"
rm -f "$out"/records/*.partial.json
printf '{"code_commit": "%s", "source": "wheel"}\n' "$(git rev-parse HEAD)" > "$out/build-info.json"
ls -l "$out" "$out/records"
