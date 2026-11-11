"""SLAM keyframes -> a COLMAP text dataset for 3DGRUT, in the twin's world frame (metric, Z up, = the collision mesh).
Usage: python mantra_twin/to_colmap.py <scan_root> <zone> [zone ...]   (after build_twin.sh) -> <scan_root>/splat/"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from reconstruct import frame_files


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


def color_calib(zone: Path):
    """K, distortion, size of the saved colour images (1080p OAK; else the depth intrinsics, no distortion)."""
    f = zone / "intrinsic_color.json"
    if f.exists():
        j = json.loads(f.read_text())
        return np.array(j["K"], float), np.array(j["dist"], float), (j["width"], j["height"])
    j = json.loads((zone / "intrinsic.json").read_text())
    return np.array(j["intrinsic_matrix"], float).reshape(3, 3).T, np.zeros(5), (j["width"], j["height"])


def sharpness(path):
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    g = cv2.resize(g, (640, int(640 * g.shape[0] / g.shape[1])), interpolation=cv2.INTER_AREA)
    return cv2.Laplacian(g, cv2.CV_32F).var()


def pick_keyframes(poses, valid, colors, move=0.10, turn_deg=6.0, window=5):
    """After every `move` m or `turn_deg` of motion, the sharpest valid frame among the next `window`."""
    keep, last = [], None
    i = 0
    while i < len(poses):
        if not valid[i]:
            i += 1
            continue
        if last is not None:
            d = np.linalg.inv(poses[last]) @ poses[i]
            ang = np.degrees(np.arccos(np.clip((np.trace(d[:3, :3]) - 1) / 2, -1, 1)))
            if np.linalg.norm(d[:3, 3]) < move and ang < turn_deg:
                i += 1
                continue
        cands = [j for j in range(i, min(i + window, len(poses))) if valid[j]]
        best = max(cands, key=lambda j: sharpness(colors[j]))
        keep.append(best)
        last, i = best, best + 1
    sh = np.array([sharpness(colors[k]) for k in keep])
    return [k for k, s in zip(keep, sh) if s > 0.35 * np.median(sh)]  # drop the motion-blurred ones


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="scan root holding the zones, merged/ and final/")
    ap.add_argument("zones", nargs="+")
    ap.add_argument("--move", type=float, default=0.10, help="metres between keyframes")
    ap.add_argument("--turn", type=float, default=6.0, help="degrees between keyframes")
    ap.add_argument("--points", type=int, default=300000, help="initial Gaussians sampled from the visual mesh")
    a = ap.parse_args()
    root = a.root.expanduser()
    zones = [root / z for z in a.zones]
    T_level = np.array(json.loads((root / "final" / "finalize.json").read_text())["T_scan_to_world"])
    zj = root / "merged" / "zones.json"
    T_zone = {Path(p).name: np.array(T) for p, T in zip(*(lambda j: (j["zones"], j["T_world"]))(
        json.loads(zj.read_text())))} if zj.exists() and len(zones) > 1 else {zones[0].name: np.eye(4)}

    out = root / "splat"
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "sparse" / "0").mkdir(parents=True, exist_ok=True)
    K, dist, (w, h) = color_calib(zones[0])
    maps = cv2.initUndistortRectifyMap(K, dist, None, K, (w, h), cv2.CV_32FC1) if np.any(dist) else None
    lines, n_img = [], 0
    for z in zones:
        stride = json.loads((z / "slam.json").read_text())["stride"]
        poses, valid = np.load(z / "poses.npy"), np.load(z / "valid.npy")
        colors, _ = frame_files(z, stride)
        keys = pick_keyframes(poses, valid, colors, a.move, a.turn)
        for k in keys:
            img = cv2.imread(str(colors[k]))
            if img.shape[1] != w:
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            if maps is not None:
                img = cv2.remap(img, maps[0], maps[1], cv2.INTER_LINEAR)
            name = f"{z.name}_{k:06d}.jpg"
            cv2.imwrite(str(out / "images" / name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            c2w = T_level @ T_zone[z.name] @ poses[k]
            w2c = np.linalg.inv(c2w)
            q, t = rot_to_qvec(w2c[:3, :3]), w2c[:3, 3]
            n_img += 1
            lines.append(f"{n_img} {' '.join(f'{v:.9f}' for v in q)} {' '.join(f'{v:.6f}' for v in t)} 1 {name}\n\n")
        print(f"{z.name}: {len(keys)} keyframes of {len(poses)} frames")

    sp = out / "sparse" / "0"
    (sp / "cameras.txt").write_text(f"# PINHOLE, undistorted\n1 PINHOLE {w} {h} {K[0, 0]} {K[1, 1]} {K[0, 2]} {K[1, 2]}\n")
    (sp / "images.txt").write_text("# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n" + "".join(lines))
    mesh = o3d.io.read_triangle_mesh(str(root / "final" / "office_visual.ply"))
    pc = mesh.sample_points_uniformly(a.points)
    xyz, rgb = np.asarray(pc.points), (np.asarray(pc.colors) * 255).round().astype(int)
    with open(sp / "points3D.txt", "w") as f:
        f.write("# POINT3D_ID X Y Z R G B ERROR TRACK[]\n")
        for i, (p, c) in enumerate(zip(xyz, rgb), 1):
            f.write(f"{i} {p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[0]} {c[1]} {c[2]} 0\n")
    print(f"{n_img} images, {len(xyz)} init points -> {out}  (train: mantra_twin/train_splat.sh {root})")


if __name__ == "__main__":
    main()
