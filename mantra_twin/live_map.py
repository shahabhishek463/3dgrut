"""Live dense map while recording (GPU): a worker thread runs Tracker on the newest frame, drops the rest.
Gives a top-down view (map + camera trail) for the recorder and live_cloud.ply at the end."""
import threading
import time

import cv2
import numpy as np
import open3d as o3d

from reconstruct import Tracker


class LiveMap:
    def __init__(self, K, rows, cols, voxel=0.03, depth_max=4.0, blocks=40000, device=None):
        device = device or ("CUDA:0" if o3d.core.cuda.is_available() else "CPU:0")
        self.tr = Tracker(np.asarray(K, float), rows, cols, voxel, depth_max, blocks, device)
        self.device, self.trail, self.ok = device, [], True
        self._new, self._lock, self._stop = None, threading.Lock(), False
        self._pts = np.zeros((0, 3))
        self._cols = np.zeros((0, 3))
        self._last_extract = 0.0
        self.fps = 0.0
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def push(self, rgb_bgr: np.ndarray, depth_mm: np.ndarray):
        with self._lock:
            self._new = (np.ascontiguousarray(rgb_bgr[:, :, ::-1]), depth_mm)

    def _run(self):
        n, t0 = 0, time.time()
        while not self._stop:
            with self._lock:
                job, self._new = self._new, None
            if job is None:
                time.sleep(0.005)
                continue
            rgb, depth = job
            self.ok = self.tr.step(o3d.t.geometry.Image(rgb), o3d.t.geometry.Image(depth.astype(np.uint16)))
            self.trail.append(self.tr.pose()[:3, 3].copy())
            n += 1
            self.fps = n / max(time.time() - t0, 1e-3)
            if time.time() - self._last_extract > 2.0:  # refresh the preview cloud every 2 s
                pc = self.tr.model.voxel_grid.extract_point_cloud().to_legacy()
                self._pts, self._cols = np.asarray(pc.points), np.asarray(pc.colors)
                self._last_extract = time.time()

    def top_view(self, size=360) -> np.ndarray:
        """Map seen from above in the first frame's camera axes (x right, z forward); red dot = camera."""
        img = np.full((size, size, 3), 30, np.uint8)
        pts, cols, trail = self._pts, self._cols, np.array(self.trail[-2000:]) if self.trail else np.zeros((0, 3))
        if len(pts) == 0:
            return img
        keep = pts[:, 1] > -1.6  # drop the ceiling (camera y points down)
        xz = np.concatenate([pts[keep][:, [0, 2]], trail[:, [0, 2]]]) if len(trail) else pts[keep][:, [0, 2]]
        lo, hi = xz.min(0) - 0.3, xz.max(0) + 0.3
        s = (size - 1) / max(hi - lo)

        def px(p):
            return np.clip(((p[:, [0, 2]] - lo) * s).astype(int), 0, size - 1)

        uv = px(pts[keep])
        img[size - 1 - uv[:, 1], uv[:, 0]] = (cols[keep][:, ::-1] * 255).astype(np.uint8)
        if len(trail):
            tv = px(trail)
            cv2.polylines(img, [np.stack([tv[:, 0], size - 1 - tv[:, 1]], 1).astype(np.int32)], False, (0, 200, 255), 1)
            cv2.circle(img, (int(tv[-1, 0]), int(size - 1 - tv[-1, 1])), 5, (0, 0, 255) if not self.ok else (0, 255, 0), -1)
        cv2.putText(img, f"map {len(pts) // 1000}k pts  {self.fps:.0f} fps  {self.device}", (6, 18), 0, 0.45, (255, 255, 255), 1)
        if not self.ok:
            cv2.putText(img, "TRACKING LOST: go back to a seen spot", (6, 38), 0, 0.45, (0, 0, 255), 1)
        return img

    def close(self, out_ply=None):
        self._stop = True
        self.th.join(timeout=5)
        if out_ply:
            o3d.io.write_point_cloud(str(out_ply), self.tr.model.voxel_grid.extract_point_cloud().to_legacy())
        return self.tr.lost, self.tr.i
