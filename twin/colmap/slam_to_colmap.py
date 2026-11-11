"""Backup when COLMAP can't register the images: write colmap/sparse/0 straight from the SLAM poses (keyframes.json),
init points sampled from the scan mesh. Usage: python twin/colmap/slam_to_colmap.py <scan_root>"""
import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def rot_to_qvec(R):
    """Rotation matrix -> COLMAP (qw, qx, qy, qz)."""
    q = np.empty(4)
    t = np.trace(R)
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        q[:] = 0.25 / s, (R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    return q if q[0] >= 0 else -q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--points", type=int, default=300000, help="initial Gaussians sampled from the visual mesh")
    a = ap.parse_args()
    root = a.root.expanduser()
    c = root / "colmap"
    kf = json.loads((c / "keyframes.json").read_text())
    sp = c / "sparse" / "0"
    sp.mkdir(parents=True, exist_ok=True)
    for f in sp.glob("*.bin"):
        f.unlink()  # 3DGRUT reads .bin before .txt
    cam = kf["camera"]
    (sp / "cameras.txt").write_text(f"1 PINHOLE {cam['width']} {cam['height']} {' '.join(map(str, cam['params']))}\n")
    lines = []
    for i, im in enumerate(kf["images"], 1):
        w2c = np.linalg.inv(np.array(im["c2w"]))
        q, t = rot_to_qvec(w2c[:3, :3]), w2c[:3, 3]
        lines.append(f"{i} {' '.join(f'{v:.9f}' for v in q)} {' '.join(f'{v:.6f}' for v in t)} 1 {im['name']}\n\n")
    (sp / "images.txt").write_text("# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n" + "".join(lines))
    pc = o3d.io.read_triangle_mesh(str(root / "final" / "office_visual.ply")).sample_points_uniformly(a.points)
    xyz, rgb = np.asarray(pc.points), (np.asarray(pc.colors) * 255).round().astype(int)
    with open(sp / "points3D.txt", "w") as f:
        f.write("# POINT3D_ID X Y Z R G B ERROR TRACK[]\n")
        for i, (p, col) in enumerate(zip(xyz, rgb), 1):
            f.write(f"{i} {p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {col[0]} {col[1]} {col[2]} 0\n")
    (c / "poses_source.txt").write_text("slam\n")
    print(f"sparse/0 from SLAM poses: {len(lines)} images, {len(xyz)} init points -> {sp}")


if __name__ == "__main__":
    main()
