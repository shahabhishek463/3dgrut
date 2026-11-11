"""Quick look without a display: top view (colour) + front view (height-coded) PNG of a mesh.
Usage: python mantra_twin/preview.py <mesh.ply> <out.png>"""
import sys

import cv2
import numpy as np
import open3d as o3d

m = o3d.io.read_triangle_mesh(sys.argv[1])
p = m.sample_points_uniformly(1_500_000)
xyz, rgb = np.asarray(p.points), (np.asarray(p.colors) * 255).astype(np.uint8)[:, ::-1]
lo, hi = xyz.min(0), xyz.max(0)
s = 800 / max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2])


def splat(u, v, depth, col, w, h):
    img = np.full((h, w, 3), 255, np.uint8)
    order = np.argsort(depth)  # far first, near overwrites
    img[v[order], u[order]] = col[order]
    return img


w, h = int((hi[0] - lo[0]) * s) + 1, int((hi[1] - lo[1]) * s) + 1
cut = xyz[:, 2] < lo[2] + 0.85 * (hi[2] - lo[2])  # hide the ceiling in the top view
top = splat(((xyz[cut, 0] - lo[0]) * s).astype(int), (h - 1 - (xyz[cut, 1] - lo[1]) * s).astype(int),
            xyz[cut, 2], rgb[cut], w, h)
hz = int((hi[2] - lo[2]) * s) + 1
zc = cv2.applyColorMap(((xyz[:, 2] - lo[2]) / (hi[2] - lo[2]) * 255).astype(np.uint8)[:, None], cv2.COLORMAP_TURBO)[:, 0]
front = splat(((xyz[:, 0] - lo[0]) * s).astype(int), (hz - 1 - (xyz[:, 2] - lo[2]) * s).astype(int), -xyz[:, 1], zc, w, hz)
cv2.putText(top, f"top  {hi[0]-lo[0]:.1f} x {hi[1]-lo[1]:.1f} m", (10, 30), 0, 0.8, (0, 0, 0), 2)
cv2.putText(front, f"front (colour = height, z {lo[2]:.2f}..{hi[2]:.2f} m)", (10, 30), 0, 0.8, (0, 0, 0), 2)
cv2.imwrite(sys.argv[2], np.vstack([top, np.zeros((6, w, 3), np.uint8), front]))
