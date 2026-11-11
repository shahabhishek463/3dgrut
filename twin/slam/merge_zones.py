"""Register zones to the first one (FPFH+RANSAC, then colored ICP) and re-fuse every frame into one TSDF.
Usage: python twin/slam/merge_zones.py <out_dir> <zone1> <zone2> ... [--voxel 0.015]; writes fused.ply + zones.json"""
import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import open3d.core as o3c

from twin.slam.reconstruct import frame_files, load_color, load_intrinsic


def prep(pcd, v):
    d = pcd.voxel_down_sample(v)
    d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=2 * v, max_nn=30))
    f = o3d.pipelines.registration.compute_fpfh_feature(d, o3d.geometry.KDTreeSearchParamHybrid(radius=5 * v, max_nn=100))
    return d, f


def register(src, dst, v=0.05):
    """Transform taking src into dst's frame, plus the fine fitness / rmse."""
    reg = o3d.pipelines.registration
    s, sf = prep(src, v)
    d, df = prep(dst, v)
    g = reg.registration_ransac_based_on_feature_matching(
        s, d, sf, df, True, 1.5 * v, reg.TransformationEstimationPointToPoint(False), 3,
        [reg.CorrespondenceCheckerBasedOnEdgeLength(0.9), reg.CorrespondenceCheckerBasedOnDistance(1.5 * v)],
        reg.RANSACConvergenceCriteria(400000, 0.999))
    T = g.transformation
    for r in (0.04, 0.02):  # coarse-to-fine colored ICP
        s2, d2 = src.voxel_down_sample(r), dst.voxel_down_sample(r)
        for p in (s2, d2):
            p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=2 * r, max_nn=30))
        T = reg.registration_colored_icp(s2, d2, 1.5 * r, T, reg.TransformationEstimationForColoredICP(),
                                         reg.ICPConvergenceCriteria(1e-6, 1e-6, 50)).transformation
    ev = reg.evaluate_registration(src.voxel_down_sample(0.02), dst.voxel_down_sample(0.02), 0.04, T)
    return T, ev.fitness, ev.inlier_rmse


def fuse(zones, T_world, voxel, depth_max, blocks, device):
    dev = o3c.Device(device)
    vbg = o3d.t.geometry.VoxelBlockGrid(("tsdf", "weight", "color"), (o3c.float32, o3c.float32, o3c.float32),
                                        ((1), (1), (3)), voxel, 16, blocks, dev)
    for z, Tz in zip(zones, T_world):
        K = o3c.Tensor(load_intrinsic(z), o3c.float64)
        stride = json.loads((z / "slam.json").read_text())["stride"]
        poses, valid = np.load(z / "poses.npy"), np.load(z / "valid.npy")
        colors, depths = frame_files(z, stride)
        for c, d, P, ok in zip(colors, depths, poses, valid):
            if not ok:
                continue
            depth = o3d.t.io.read_image(str(d)).to(dev)
            color = load_color(c, depth.columns, depth.rows).to(dev)
            ext = o3c.Tensor(np.linalg.inv(Tz @ P), o3c.float64)  # world -> camera
            coords = vbg.compute_unique_block_coordinates(depth, K, ext, 1000.0, depth_max)
            vbg.integrate(coords, depth, color, K, K, ext, 1000.0, depth_max)
        print(f"  fused {z.name}")
    return vbg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("zones", type=Path, nargs="+", help="in walking order; each overlaps the one before")
    ap.add_argument("--voxel", type=float, default=0.015)
    ap.add_argument("--depth-max", type=float, default=4.0)
    ap.add_argument("--blocks", type=int, default=80000, help="voxel-block capacity; 80k ~ 6.5 GB GPU at 20 B/voxel")
    ap.add_argument("--device", default="CUDA:0" if o3d.core.cuda.is_available() else "CPU:0")
    a = ap.parse_args()
    zones = [z.expanduser() for z in a.zones]
    out = a.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)

    clouds = [o3d.io.read_point_cloud(str(z / "cloud.ply")) for z in zones]
    T_world, report, world = [np.eye(4)], [], clouds[0]
    for z, c in zip(zones[1:], clouds[1:]):
        T, fit, rmse = register(c, world)
        T_world.append(T)
        report.append({"zone": z.name, "fitness": round(fit, 3), "rmse_m": round(rmse, 4)})
        print(f"{z.name} -> world: fitness {fit:.2f}, rmse {1000 * rmse:.1f} mm"
              + ("   <- WEAK: zones may not overlap enough" if fit < 0.15 else ""))
        world = world + c.transform(T)
    vbg = fuse(zones, T_world, a.voxel, a.depth_max, a.blocks, a.device)
    mesh = vbg.extract_triangle_mesh().to_legacy()
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(out / "fused.ply"), mesh)
    (out / "zones.json").write_text(json.dumps({"zones": [str(z) for z in zones],
                                                "T_world": [t.tolist() for t in T_world], "registration": report}, indent=1))
    print(f"fused mesh {len(mesh.triangles)} tris -> {out / 'fused.ply'}")


if __name__ == "__main__":
    main()
