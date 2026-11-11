"""SLAM recording -> the NVIDIA workflow's COLMAP input: <root>/colmap/images/ (sharp, undistorted 1080p keyframes),
keyframes.json (calibrated PINHOLE + SLAM pose of each, twin frame) and ref_positions.txt (for the metric alignment)."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from twin.slam.reconstruct import frame_files


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


def pick(poses, valid, colors, move=0.10, turn_deg=6.0, window=5):
    """After every `move` m or `turn_deg` of motion, the sharpest valid frame among the next `window`."""
    keep, last, i = [], None, 0
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


def zone_transforms(root: Path, zones):
    """Each zone's SLAM frame -> the twin frame (levelled, floor z=0): finalize's T @ merge's T."""
    T_level = np.array(json.loads((root / "final" / "finalize.json").read_text())["T_scan_to_world"])
    zj = root / "merged" / "zones.json"
    if zj.exists() and len(zones) > 1:
        j = json.loads(zj.read_text())
        T_zone = {Path(p).name: np.array(T) for p, T in zip(j["zones"], j["T_world"])}
    else:
        T_zone = {zones[0].name: np.eye(4)}
    return {z.name: T_level @ T_zone[z.name] for z in zones}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="scan root holding the zones, merged/ and final/")
    ap.add_argument("zones", nargs="+")
    ap.add_argument("--move", type=float, default=0.10, help="metres between keyframes")
    ap.add_argument("--turn", type=float, default=6.0, help="degrees between keyframes")
    a = ap.parse_args()
    root = a.root.expanduser()
    zones = [root / z for z in a.zones]
    T = zone_transforms(root, zones)
    out = root / "colmap"
    (out / "images").mkdir(parents=True, exist_ok=True)
    for old in (out / "images").glob("*.jpg"):
        old.unlink()
    K, dist, (w, h) = color_calib(zones[0])
    maps = cv2.initUndistortRectifyMap(K, dist, None, K, (w, h), cv2.CV_32FC1) if np.any(dist) else None
    images, refs = [], []
    for z in zones:
        stride = json.loads((z / "slam.json").read_text())["stride"]
        poses, valid = np.load(z / "poses.npy"), np.load(z / "valid.npy")
        colors, _ = frame_files(z, stride)
        keys = pick(poses, valid, colors, a.move, a.turn)
        for k in keys:
            img = cv2.imread(str(colors[k]))
            if img.shape[1] != w:
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            if maps is not None:
                img = cv2.remap(img, maps[0], maps[1], cv2.INTER_LINEAR)
            name = f"{z.name}_{k:06d}.jpg"
            cv2.imwrite(str(out / "images" / name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            c2w = T[z.name] @ poses[k]
            images.append({"name": name, "c2w": c2w.tolist()})
            refs.append(f"{name} {c2w[0, 3]:.5f} {c2w[1, 3]:.5f} {c2w[2, 3]:.5f}\n")
        print(f"{z.name}: {len(keys)} keyframes of {len(poses)} frames")
    cam = {"model": "PINHOLE", "width": w, "height": h, "params": [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]}
    (out / "keyframes.json").write_text(json.dumps({"camera": cam, "images": images}))
    (out / "ref_positions.txt").write_text("".join(refs))
    print(f"{len(images)} images -> {out / 'images'}")


if __name__ == "__main__":
    main()
