#!/usr/bin/env bash
# Inside the devcontainer. Scanned zones -> collision USD + splat dataset: build_twin.sh <scan_root> <zone1> [zone2 ...]
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(readlink -f "$1")"; shift
ZONES=(); for z in "$@"; do ZONES+=("$ROOT/$z"); done
for z in "${ZONES[@]}"; do [ -f "$z/poses.npy" ] || python "$HERE/reconstruct.py" "$z"; done
if [ ${#ZONES[@]} -gt 1 ]; then python "$HERE/merge_zones.py" "$ROOT/merged" "${ZONES[@]}"; MESH="$ROOT/merged/fused.ply"
else MESH="${ZONES[0]}/mesh.ply"; fi
python "$HERE/finalize.py" "$MESH" "$ROOT/final" > /dev/null
python "$HERE/preview.py" "$ROOT/final/office_visual.ply" "$ROOT/final/preview.png"
python "$HERE/mesh_to_usd.py" "$ROOT/final"
python "$HERE/to_colmap.py" "$ROOT" "$@"
echo "next: check $ROOT/final/preview.png, then mantra_twin/train_splat.sh $ROOT"
