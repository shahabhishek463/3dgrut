"""OAK-D link check: USB speed, calibration, fps and invalid-depth share over N seconds.
Usage: python twin/capture/oak_check.py [--seconds 10] [--show]"""
import argparse
import time

import cv2
import depthai as dai
import numpy as np

from twin.capture.oak_common import ir_on, H, W, build_pipeline, lens_position


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=10)
    ap.add_argument("--snapshot", help="save the last RGB|depth pair to this PNG")
    ap.add_argument("--show", action="store_true", help="live RGB | depth window (q quits)")
    a = ap.parse_args()

    found = dai.Device.getAllAvailableDevices()
    if not found:
        raise SystemExit("No OAK found. Plug it into a USB 3 port; check the udev rule (README step 2).")
    with dai.Device() as dev:
        speed = dev.getUsbSpeed().name
        cal = dev.readCalibration()
        K = cal.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, W, H)
        lens = lens_position(dev)
        print(f"device   {dev.getDeviceName()}  mxid {dev.getMxId()}  USB {speed}  IMU {dev.getConnectedIMU()}")
        print(f"cameras  {[c.name for c in dev.getConnectedCameras()]}  RGB lens pos {lens}")
        print(f"K @{W}x{H}  fx {K[0][0]:.1f} fy {K[1][1]:.1f} cx {K[0][2]:.1f} cy {K[1][2]:.1f}  "
              f"baseline {abs(cal.getBaselineDistance()):.2f} cm")
        if speed not in ("SUPER", "SUPER_PLUS"):
            print(f"WARNING: USB {speed}. Use a blue USB 3 port directly on the PC (not the hub).")
        print(f"IR dot projector {'on' if ir_on(dev) else 'not available'}")
        dev.startPipeline(build_pipeline(lens_pos=lens))
        q = dev.getOutputQueue("rgbd", maxSize=4, blocking=False)
        n, invalid, t0 = 0, [], time.time()
        while time.time() - t0 < a.seconds:
            msg = q.tryGet()
            if msg is None:
                time.sleep(0.005)
                continue
            depth = msg["depth"].getFrame()
            n += 1
            invalid.append(float((depth == 0).mean()))
            last = (msg["rgb"].getCvFrame(), depth)
            if a.show:
                vis = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=255 / 4000), cv2.COLORMAP_TURBO)
                vis[depth == 0] = 0
                both = np.hstack([cv2.resize(msg["rgb"].getCvFrame(), (W, H)), vis])
                cv2.imshow("OAK  RGB | depth (black = no depth)", cv2.resize(both, None, fx=0.5, fy=0.5))
                if cv2.waitKey(1) == ord("q"):
                    break
        dt = time.time() - t0
    if a.snapshot and n:
        rgb, depth = last
        rgb = cv2.resize(rgb, (W, H))
        vis = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=255 / 4000), cv2.COLORMAP_TURBO)
        vis[depth == 0] = 0
        cv2.imwrite(a.snapshot, cv2.resize(np.hstack([rgb, vis]), None, fx=0.5, fy=0.5))
    fps = n / dt if dt else 0
    inv = 100 * np.mean(invalid) if invalid else 100
    print(f"frames {n} in {dt:.1f} s = {fps:.1f} fps   invalid depth {inv:.1f}% (normal: ~40-50% on a room view, beyond 4 m is cut)")
    ok = speed in ("SUPER", "SUPER_PLUS") and fps > 10
    print("PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
