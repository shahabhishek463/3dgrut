"""Scale check: Shift+click two points (e.g. the ends of the 1 m tape mark), close the window, read the distance.
Usage: python twin/collision/measure.py <mesh.ply> [--expect 1.0]"""
import argparse

import numpy as np
import open3d as o3d

ap = argparse.ArgumentParser()
ap.add_argument("mesh")
ap.add_argument("--expect", type=float, help="true length in metres")
a = ap.parse_args()
pcd = o3d.io.read_triangle_mesh(a.mesh).sample_points_uniformly(2_000_000)
v = o3d.visualization.VisualizerWithEditing()
v.create_window("Shift+click 2 points, then close (Q)")
v.add_geometry(pcd)
v.run()
v.destroy_window()
idx = v.get_picked_points()
if len(idx) < 2:
    raise SystemExit("pick two points")
p = np.asarray(pcd.points)[idx[-2:]]
d = float(np.linalg.norm(p[1] - p[0]))
msg = f"distance {d:.3f} m"
if a.expect:
    msg += f"  vs {a.expect:.3f} m expected: error {100 * (d - a.expect) / a.expect:+.1f}%"
print(msg)
