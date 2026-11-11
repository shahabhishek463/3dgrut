"""Level a scan (floor -> z=0, main wall -> X axis), drop floating junk, write visual + collision meshes.
Usage: python mantra_twin/finalize.py <mesh.ply> <out_dir> [--visual-tris 300000] [--collision-tris 30000]"""
import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def rot_a_to_b(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c))


def planes(pcd, n=8, dist=0.02):
    out, rest = [], pcd
    for _ in range(n):
        if len(rest.points) < 500:
            break
        model, idx = rest.segment_plane(dist, 3, 2000)
        out.append((np.array(model), len(idx), np.asarray(rest.points)[idx]))
        rest = rest.select_by_index(idx, invert=True)
    return out


def level_transform(mesh, up_hint):
    """Floor = the lowest big plane whose normal is within 25 deg of up_hint (camera -Y at the first frame)."""
    pcd = mesh.sample_points_uniformly(200000).voxel_down_sample(0.03)
    total = len(pcd.points)
    cands = []
    for m, cnt, pts in planes(pcd):
        nrm = m[:3] / np.linalg.norm(m[:3])
        if nrm @ up_hint < 0:
            nrm = -nrm
        if nrm @ up_hint > np.cos(np.radians(25)) and cnt > 0.03 * total:
            cands.append((float(pts.mean(0) @ up_hint), nrm, pts))
    if not cands:
        raise SystemExit("no floor plane found: scan more of the floor")
    _, up, floor_pts = min(cands, key=lambda c: c[0])
    R = rot_a_to_b(up, np.array([0, 0, 1.0]))
    T = np.eye(4)
    T[:3, :3] = R
    T[2, 3] = -float((floor_pts @ R.T)[:, 2].mean())
    # yaw: the biggest vertical plane (a wall) faces +-X
    lv = pcd.transform(T)
    walls = [(cnt, m[:3]) for m, cnt, _ in planes(lv) if abs(m[2]) / np.linalg.norm(m[:3]) < 0.2]
    if walls:
        nx = max(walls, key=lambda w: w[0])[1]
        yaw = -np.arctan2(nx[1], nx[0])
        Rz = np.eye(4)
        Rz[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
        T = Rz @ T
    return T


def clean(mesh, min_tris):
    cl, counts, _ = mesh.cluster_connected_triangles()
    cl, counts = np.asarray(cl), np.asarray(counts)
    mesh.remove_triangles_by_mask(counts[cl] < min_tris)
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()
    return mesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--visual-tris", type=int, default=300000)
    ap.add_argument("--collision-tris", type=int, default=30000)
    ap.add_argument("--min-cluster", type=int, default=800, help="drop connected pieces smaller than this")
    ap.add_argument("--exclude-box", type=float, nargs=4, action="append", metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
                    help="levelled-frame floor area whose geometry is cut (e.g. the real bench the sim models itself)")
    ap.add_argument("--up", type=float, nargs=3, default=[0, -1, 0], help="up in the scan frame (camera -Y)")
    a = ap.parse_args()
    out = a.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)

    mesh = o3d.io.read_triangle_mesh(str(a.mesh.expanduser()))
    n0 = len(mesh.triangles)
    T = level_transform(mesh, np.array(a.up, float))
    mesh.transform(T)
    for x0, y0, x1, y1 in a.exclude_box or []:
        v = np.asarray(mesh.vertices)
        inside = (v[:, 0] > x0) & (v[:, 0] < x1) & (v[:, 1] > y0) & (v[:, 1] < y1) & (v[:, 2] > 0.02)  # keep the floor
        mesh.remove_vertices_by_mask(inside)
    mesh = clean(mesh, a.min_cluster)
    vis = mesh.simplify_quadric_decimation(a.visual_tris) if len(mesh.triangles) > a.visual_tris else mesh
    vis.compute_vertex_normals()
    col = mesh.simplify_vertex_clustering(0.03).simplify_quadric_decimation(a.collision_tris)
    col = clean(col, 50)
    col.remove_non_manifold_edges()
    col.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(out / "office_visual.ply"), vis)
    o3d.io.write_triangle_mesh(str(out / "office_collision.ply"), col)
    bb = vis.get_axis_aligned_bounding_box()
    info = {"source": str(a.mesh), "T_scan_to_world": T.tolist(), "tris_in": n0, "exclude_boxes": a.exclude_box or [],
            "visual_tris": len(vis.triangles), "collision_tris": len(col.triangles),
            "bbox_min": np.round(bb.min_bound, 3).tolist(), "bbox_max": np.round(bb.max_bound, 3).tolist()}
    (out / "finalize.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
