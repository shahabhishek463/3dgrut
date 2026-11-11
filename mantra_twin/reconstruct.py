"""Per-zone dense RGB-D SLAM (Open3D t.pipelines.slam, CUDA): poses.npy, mesh.ply, cloud.ply in the zone folder.
Usage: python mantra_twin/reconstruct.py /data/office_scan/<zone> [--voxel 0.015] [--show]"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import open3d.core as o3c


def load_intrinsic(zone: Path) -> np.ndarray:
    j = json.loads((zone / "intrinsic.json").read_text())
    return np.array(j["intrinsic_matrix"], dtype=np.float64).reshape(3, 3).T  # stored column-major


def frame_files(zone: Path, stride: int):
    color = sorted((zone / "color").glob("*.jpg")) or sorted((zone / "color").glob("*.png"))
    depth = sorted((zone / "depth").glob("*.png"))
    if len(color) != len(depth) or not color:
        raise SystemExit(f"{zone}: {len(color)} color vs {len(depth)} depth frames")
    return color[::stride], depth[::stride]


class Tracker:
    """Frame-to-model dense SLAM; feed RGB-D images in order with step()."""

    def __init__(self, K: np.ndarray, rows: int, cols: int, voxel: float, depth_max: float, blocks: int,
                 device: str, scale: float = 1000.0):
        self.dev, self.scale, self.depth_max = o3c.Device(device), scale, depth_max
        Kt = o3c.Tensor(K, o3c.Dtype.Float64)
        self.T = o3c.Tensor(np.eye(4))
        self.model = o3d.t.pipelines.slam.Model(voxel, 16, blocks, self.T, self.dev)
        self.inp = o3d.t.pipelines.slam.Frame(rows, cols, Kt, self.dev)
        self.ray = o3d.t.pipelines.slam.Frame(rows, cols, Kt, self.dev)
        self.i, self.lost = 0, 0

    def step(self, color: o3d.t.geometry.Image, depth: o3d.t.geometry.Image) -> bool:
        self.inp.set_data_from_image("depth", depth.to(self.dev))
        self.inp.set_data_from_image("color", color.to(self.dev))
        ok = True
        if self.i > 0:
            try:
                r = self.model.track_frame_to_model(self.inp, self.ray, self.scale, self.depth_max, 0.07)
                step = r.transformation.cpu().numpy()
                # a jump > 15 cm or > 15 deg between frames at walking speed means tracking failed
                ok = np.linalg.norm(step[:3, 3]) < 0.15 and np.arccos(np.clip((np.trace(step[:3, :3]) - 1) / 2, -1, 1)) < 0.26
                if ok:
                    self.T = self.T @ r.transformation
            except RuntimeError:
                ok = False
        self.lost += not ok
        self.model.update_frame_pose(self.i, self.T)
        if ok:
            self.model.integrate(self.inp, self.scale, self.depth_max, 8.0)
        self.model.synthesize_model_frame(self.ray, self.scale, 0.3, self.depth_max, 8.0, False)
        self.i += 1
        return ok

    def pose(self) -> np.ndarray:
        return self.T.cpu().numpy()


def load_color(path: Path, cols: int, rows: int) -> o3d.t.geometry.Image:
    """RGB image at depth size (the saved colour may be 1080p while depth is 720p)."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr.shape[1] != cols:
        bgr = cv2.resize(bgr, (cols, rows), interpolation=cv2.INTER_AREA)
    return o3d.t.geometry.Image(np.ascontiguousarray(bgr[:, :, ::-1]))


def run_slam(zone: Path, voxel: float, depth_max: float, stride: int, blocks: int, device: str):
    colors, depths = frame_files(zone, stride)
    scale = float(json.loads((zone / "meta.json").read_text()).get("depth_scale", 1000.0)) \
        if (zone / "meta.json").exists() else 1000.0
    d0 = o3d.t.io.read_image(str(depths[0]))
    tr = Tracker(load_intrinsic(zone), d0.rows, d0.columns, voxel, depth_max, blocks, device, scale)
    poses, valid, t0 = [], [], time.time()
    for i, (c, d) in enumerate(zip(colors, depths)):
        valid.append(tr.step(load_color(c, d0.columns, d0.rows), o3d.t.io.read_image(str(d))))
        poses.append(tr.pose())
        if i % 100 == 0:
            print(f"  frame {i}/{len(colors)}  lost {tr.lost}  {i / max(time.time() - t0, 1e-3):.1f} fps", flush=True)
    return tr.model, np.stack(poses), np.array(valid), tr.lost, len(colors)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zone", type=Path)
    ap.add_argument("--voxel", type=float, default=0.015, help="TSDF voxel size, m")
    ap.add_argument("--depth-max", type=float, default=4.0)
    ap.add_argument("--stride", type=int, default=1, help="use every Nth frame")
    ap.add_argument("--blocks", type=int, default=60000, help="voxel-block capacity (16^3 voxels each)")
    ap.add_argument("--device", default="CUDA:0" if o3d.core.cuda.is_available() else "CPU:0")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    zone = a.zone.expanduser()

    model, poses, valid, lost, n = run_slam(zone, a.voxel, a.depth_max, a.stride, a.blocks, a.device)
    np.save(zone / "poses.npy", poses)
    np.save(zone / "valid.npy", valid)
    (zone / "slam.json").write_text(json.dumps({"voxel": a.voxel, "depth_max": a.depth_max, "stride": a.stride,
                                                "frames": n, "lost": int(lost)}, indent=1))
    model.voxel_grid.save(str(zone / "vbg.npz"))
    mesh = model.voxel_grid.extract_triangle_mesh().to_legacy()
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(zone / "mesh.ply"), mesh)
    pcd = model.voxel_grid.extract_point_cloud().to_legacy()
    o3d.io.write_point_cloud(str(zone / "cloud.ply"), pcd)
    path_len = float(np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1).sum())
    loop_gap = float(np.linalg.norm(poses[-1, :3, 3] - poses[0, :3, 3]))
    ext = mesh.get_axis_aligned_bounding_box().get_extent()
    print(f"{zone.name}: {n} frames, lost {lost} ({100 * lost / n:.1f}%), path {path_len:.1f} m, "
          f"end-to-start {loop_gap:.2f} m, mesh {len(mesh.triangles)} tris, extent {np.round(ext, 2)} m")
    if lost / n > 0.05:
        print("WARNING: > 5% frames lost tracking: walk slower / more texture, or rescan this zone.")
    if a.show:
        o3d.visualization.draw_geometries([mesh], window_name=zone.name)


if __name__ == "__main__":
    main()
