#!/usr/bin/env bash
# Inside the devcontainer. Recorded zones -> collider + the COLMAP dataset 3DGUT trains on (twin/README.md).
# Usage: build_twin.sh <scan_root> <zone1> [zone2 ...]   (zones in walking order; SKIP_COLMAP=1 uses the SLAM poses)
set -e
T="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(readlink -f "$1")"; shift
ZONES=(); for z in "$@"; do ZONES+=("$ROOT/$z"); done

echo "== 1/4 SLAM per zone"
for z in "${ZONES[@]}"; do [ -f "$z/poses.npy" ] || python "$T/slam/reconstruct.py" "$z"; done
if [ ${#ZONES[@]} -gt 1 ]; then python "$T/slam/merge_zones.py" "$ROOT/merged" "${ZONES[@]}"; MESH="$ROOT/merged/fused.ply"
else MESH="${ZONES[0]}/mesh.ply"; fi

echo "== 2/4 collision mesh (levelled: floor z=0, metres)"
python "$T/collision/finalize.py" "$MESH" "$ROOT/final" ${EXCLUDE_BOX:+--exclude-box $EXCLUDE_BOX} > /dev/null
python "$T/collision/preview.py" "$ROOT/final/office_visual.ply" "$ROOT/final/preview.png"
python "$T/collision/mesh_to_usd.py" "$ROOT/final"

echo "== 3/4 keyframes -> colmap/images"
python "$T/colmap/keyframes.py" "$ROOT" "$@"

echo "== 4/4 COLMAP (NVIDIA workflow) + metric alignment"
if [ "${SKIP_COLMAP:-0}" = 1 ]; then python "$T/colmap/slam_to_colmap.py" "$ROOT"
else
  bash "$T/colmap/run_colmap.sh" "$ROOT" && python "$T/colmap/align_metric.py" "$ROOT" \
    || { echo "!! COLMAP route failed: falling back to the SLAM poses"; python "$T/colmap/slam_to_colmap.py" "$ROOT"; }
fi
echo "poses: $(cat "$ROOT/colmap/poses_source.txt")   next: check $ROOT/final/preview.png, then twin/splat/train_splat.sh $ROOT"
