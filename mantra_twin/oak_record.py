"""Record one scan zone as an Open3D RGB-D dataset: color/*.jpg, depth/*.png (uint16 mm), intrinsic.json.
Colour 1080p, depth 720p; live GPU map + live_cloud.ply unless --no-live-map. Usage: python oak_record.py <zone>."""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import depthai as dai
import numpy as np

from oak_common import H, W, ir_on, build_pipeline, lens_position, write_intrinsic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zone", help="folder name, e.g. bench, desks, corridor")
    ap.add_argument("--out", default="/data/office_scan" if Path("/data").is_dir() else "~/Robotics/data/office_scan")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--max-minutes", type=float, default=6)
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--no-live-map", action="store_true", help="record only (no open3d needed)")
    a = ap.parse_args()

    root = Path(a.out).expanduser() / a.zone
    if root.exists() and any(root.iterdir()):
        raise SystemExit(f"{root} already has data; pick a new zone name or delete it first.")
    (root / "color").mkdir(parents=True, exist_ok=True)
    (root / "depth").mkdir(exist_ok=True)
    pool = ThreadPoolExecutor(8)

    with dai.Device() as dev:
        if dev.getUsbSpeed().name not in ("SUPER", "SUPER_PLUS"):
            raise SystemExit(f"USB {dev.getUsbSpeed().name}: plug the OAK into a USB 3 port first.")
        lens = lens_position(dev)
        K = write_intrinsic(dev, root)
        live = None
        if not a.no_live_map:
            from live_map import LiveMap
            live = LiveMap(K, H, W)
            print(f"live map on {live.device}")
        print(f"IR dot projector {'on' if ir_on(dev) else 'not available'}")
        dev.startPipeline(build_pipeline(fps=a.fps, lens_pos=lens))
        q = dev.getOutputQueue("rgbd", maxSize=8, blocking=False)
        n, paused, t0, inv_hist = 0, False, time.time(), []
        print(f"recording zone '{a.zone}' -> {root}   SPACE pause/resume, q stop")
        while time.time() - t0 < a.max_minutes * 60:
            msg = q.tryGet()
            if msg is None:
                time.sleep(0.003)
                continue
            rgb, depth = msg["rgb"].getCvFrame(), msg["depth"].getFrame()
            inv = float((depth == 0).mean())
            inv_hist = (inv_hist + [inv])[-15:]
            if not paused:
                name = f"{n:06d}"
                pool.submit(cv2.imwrite, str(root / "color" / f"{name}.jpg"), rgb, [cv2.IMWRITE_JPEG_QUALITY, 92])
                pool.submit(cv2.imwrite, str(root / "depth" / f"{name}.png"), depth.astype(np.uint16))
                n += 1
                if live:
                    live.push(cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA), depth)
            if not a.no_preview:
                rgb_s = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
                vis = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=255 / 4000), cv2.COLORMAP_TURBO)
                vis[depth == 0] = 0
                both = cv2.resize(np.hstack([rgb_s, vis]), None, fx=0.5, fy=0.5)
                bad = np.mean(inv_hist) > 0.65  # ~45% is normal in a room: > 4 m and the stereo edge band are empty
                txt = f"{'PAUSED ' if paused else 'REC '}{n} frames  {time.time() - t0:5.0f}s  no-depth {100 * np.mean(inv_hist):.0f}%"
                cv2.putText(both, txt + ("  <- aim at texture / step back" if bad else ""), (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if bad or paused else (0, 255, 0), 2)
                if live:
                    both = np.hstack([both, live.top_view(both.shape[0])])
                cv2.imshow(f"scan: {a.zone}", both)
                k = cv2.waitKey(1)
                if k == ord("q"):
                    break
                if k == ord(" "):
                    paused = not paused
    pool.shutdown(wait=True)
    if live:
        lost, done = live.close(root / "live_cloud.ply")
        print(f"live map: {done} frames mapped, {lost} lost tracking -> {root / 'live_cloud.ply'}")
    meta = {"zone": a.zone, "frames": n, "fps": a.fps, "seconds": round(time.time() - t0, 1),
            "depth_scale": 1000.0, "lens_position": lens, "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    (root / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"saved {n} frames to {root}")


if __name__ == "__main__":
    main()
