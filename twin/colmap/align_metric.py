"""COLMAP's model -> the twin frame (metres, floor z=0, Z up): `colmap model_aligner` onto the OAK's measured camera
positions. Picks the biggest sparse_raw model, writes colmap/sparse/0, checks it; exit 1 = use slam_to_colmap.py."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from threedgrut.datasets.utils import qvec_to_so3, read_colmap_extrinsics_binary


def centres_and_rotations(images_bin: Path):
    out = {}
    for im in read_colmap_extrinsics_binary(str(images_bin)).values():
        R = qvec_to_so3(im.qvec)
        out[im.name] = (-R.T @ np.asarray(im.tvec), R.T)  # camera centre, camera-to-world rotation
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--max-error", type=float, default=0.10, help="RANSAC inlier distance for the alignment, m")
    ap.add_argument("--min-registered", type=float, default=0.6, help="fail below this share of images registered")
    a = ap.parse_args()
    c = a.root.expanduser() / "colmap"
    kf = json.loads((c / "keyframes.json").read_text())
    slam = {im["name"]: np.array(im["c2w"]) for im in kf["images"]}
    models = [m for m in (c / "sparse_raw").iterdir() if (m / "images.bin").exists()]
    if not models:
        sys.exit("COLMAP produced no model: use python twin/colmap/slam_to_colmap.py")
    best = max(models, key=lambda m: len(read_colmap_extrinsics_binary(str(m / "images.bin"))))
    out = c / "sparse" / "0"
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*"):
        f.unlink()
    subprocess.run(["colmap", "model_aligner", "--input_path", str(best), "--output_path", str(out),
                    "--ref_images_path", str(c / "ref_positions.txt"), "--ref_is_gps", "0",
                    "--alignment_type", "custom", "--alignment_max_error", str(a.max_error)], check=True)
    got = centres_and_rotations(out / "images.bin")
    common = [n for n in got if n in slam]
    pos = np.array([np.linalg.norm(got[n][0] - slam[n][:3, 3]) for n in common])
    ang = np.array([np.degrees(np.arccos(np.clip((np.trace(got[n][1].T @ slam[n][:3, :3]) - 1) / 2, -1, 1)))
                    for n in common])
    reg = len(got) / len(slam)
    rep = {"model": best.name, "registered": len(got), "keyframes": len(slam), "registered_share": round(reg, 3),
           "position_err_m": {"median": round(float(np.median(pos)), 4), "p90": round(float(np.percentile(pos, 90)), 4)},
           "rotation_err_deg": {"median": round(float(np.median(ang)), 2), "p90": round(float(np.percentile(ang, 90)), 2)}}
    (c / "align_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    ok = reg >= a.min_registered and rep["position_err_m"]["median"] < 0.10 and rep["rotation_err_deg"]["median"] < 5
    (c / "poses_source.txt").write_text("colmap\n" if ok else "colmap-FAILED\n")
    print("ALIGNED: colmap/sparse/0 is in the twin frame" if ok else "ALIGNMENT FAILED: use slam_to_colmap.py")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
