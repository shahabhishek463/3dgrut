#!/usr/bin/env bash
# Inside the devcontainer: 3DGUT (MCMC) on <scan_root>/splat, PLY -> NuRec USDZ, then the combined twin USD.
# Usage: train_splat.sh <scan_root> [n_iterations=30000] [extra hydra overrides...]
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(readlink -f "$1")"; ITERS="${2:-30000}"; shift; shift || true
cd "$HERE/.."
python train.py --config-name apps/colmap_3dgut_mcmc.yaml \
    path="$ROOT/splat" out_dir="$ROOT/splat_runs" experiment_name=office \
    n_iterations="$ITERS" "checkpoint.iterations=[$ITERS]" export_ply.enabled=true \
    dataset.load_exif=false "$@"
PLY="$(ls -t "$ROOT"/splat_runs/office/*/export_last.ply | head -1)"
python -m threedgrut.export.scripts.ply_to_usd "$PLY" --output_file "$ROOT/final/office_splat.usdz"
python "$HERE/compose_twin.py" "$ROOT/final"
