#!/usr/bin/env bash
# The NVIDIA workflow's COLMAP step (blog: feature_extractor -> exhaustive_matcher -> mapper) on <root>/colmap/images.
# Option names are this COLMAP's (4.1): the blog's SiftExtraction.max_image_size / SiftMatching.use_gpu were renamed.
# Usage: run_colmap.sh <scan_root> [--sequential]   (sequential matching for > ~800 images)
set -e
ROOT="$(readlink -f "$1")"; C="$ROOT/colmap"
MATCHER=exhaustive_matcher; [ "${2:-}" = "--sequential" ] && MATCHER=sequential_matcher
# The OAK's calibrated PINHOLE (images are undistorted) as COLMAP's starting intrinsics; single camera as in the blog.
PARAMS=$(python -c "import json; print(','.join(f'{v:.4f}' for v in json.load(open('$C/keyframes.json'))['camera']['params']))")
rm -rf "$C/database.db" "$C/sparse_raw"; mkdir -p "$C/sparse_raw"
colmap feature_extractor --database_path "$C/database.db" --image_path "$C/images" \
    --ImageReader.single_camera 1 --ImageReader.camera_model PINHOLE --ImageReader.camera_params "$PARAMS" \
    --FeatureExtraction.max_image_size 2000 --FeatureExtraction.use_gpu 1 \
    --SiftExtraction.estimate_affine_shape 1 --SiftExtraction.domain_size_pooling 1
colmap "$MATCHER" --database_path "$C/database.db" --FeatureMatching.use_gpu 1
colmap mapper --database_path "$C/database.db" --image_path "$C/images" --output_path "$C/sparse_raw"
echo "COLMAP models: $(ls "$C/sparse_raw" | tr '\n' ' ')-> next: python twin/colmap/align_metric.py $ROOT"
