#!/usr/bin/env bash
# The NVIDIA workflow's 3DGUT step (apps/colmap_3dgut_mcmc.yaml) on <root>/colmap, then PLY -> NuRec USDZ and the
# combined twin scene. Usage: train_splat.sh <scan_root> [n_iterations=30000] [extra hydra overrides...]
# USDZ via ply_to_usd, not export_usd(z).enabled: in-training export writes a broken reference (CLAUDE.md §3).
set -e
T="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="$(readlink -f "$1")"; ITERS="${2:-30000}"; shift; shift || true
cd "$T/.."
python train.py --config-name apps/colmap_3dgut_mcmc.yaml \
    path="$ROOT/colmap" out_dir="$ROOT/splat_runs" experiment_name=office \
    n_iterations="$ITERS" "checkpoint.iterations=[$ITERS]" export_ply.enabled=true \
    dataset.load_exif=false "$@"
PLY="$(ls -t "$ROOT"/splat_runs/office/*/export_last.ply | head -1)"
python -m threedgrut.export.scripts.ply_to_usd "$PLY" --output_file "$ROOT/final/office_splat.usdz"
python "$T/isaac/compose_twin.py" "$ROOT/final"
