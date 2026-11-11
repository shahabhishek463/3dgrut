"""OAK-D Pro RGB-D pipeline (depthai v2): RGB 1920x1080 (splat images), depth aligned to RGB at 1280x720 (SLAM, mesh).
Same optical centre, so the depth intrinsics are the RGB's scaled by 2/3."""
import json
from datetime import timedelta
from pathlib import Path

import depthai as dai

W, H = 1280, 720  # depth (and the colour fed to SLAM)
CW, CH = 1920, 1080  # saved colour


def build_pipeline(fps: int = 15, depth_max_mm: int = 4000, lens_pos: int | None = None):
    p = dai.Pipeline()
    rgb = p.create(dai.node.ColorCamera)
    rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    rgb.setFps(fps)
    if lens_pos is not None:  # lock focus at the calibrated lens position: autofocus would move fx/fy
        rgb.initialControl.setManualFocus(lens_pos)

    left, right = p.create(dai.node.MonoCamera), p.create(dai.node.MonoCamera)
    for cam, sock in ((left, dai.CameraBoardSocket.CAM_B), (right, dai.CameraBoardSocket.CAM_C)):
        cam.setBoardSocket(sock)
        cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        cam.setFps(fps)

    st = p.create(dai.node.StereoDepth)
    st.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
    st.setLeftRightCheck(True)
    st.setSubpixel(True)
    st.setExtendedDisparity(False)
    st.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    st.setOutputSize(W, H)
    cfg = st.initialConfig.get()
    cfg.postProcessing.speckleFilter.enable = True
    cfg.postProcessing.speckleFilter.speckleRange = 28  # the most this stereo node's memory allows
    cfg.postProcessing.spatialFilter.enable = True
    cfg.postProcessing.spatialFilter.holeFillingRadius = 2
    cfg.postProcessing.thresholdFilter.minRange = 300
    cfg.postProcessing.thresholdFilter.maxRange = depth_max_mm
    st.initialConfig.set(cfg)
    left.out.link(st.left)
    right.out.link(st.right)

    sync = p.create(dai.node.Sync)
    sync.setSyncThreshold(timedelta(milliseconds=500 / fps))
    rgb.isp.link(sync.inputs["rgb"])
    st.depth.link(sync.inputs["depth"])
    xout = p.create(dai.node.XLinkOut)
    xout.setStreamName("rgbd")
    sync.out.link(xout.input)
    return p


def lens_position(dev: dai.Device) -> int | None:
    try:
        return int(dev.readCalibration().getLensPosition(dai.CameraBoardSocket.CAM_A)) or None
    except Exception:
        return None


def write_intrinsic(dev: dai.Device, zone: Path) -> list[list[float]]:
    """intrinsic.json (depth size, Open3D format) + intrinsic_color.json (saved colour size + distortion)."""
    cal = dev.readCalibration()
    K = cal.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, W, H)
    col_major = [K[0][0], 0, 0, 0, K[1][1], 0, K[0][2], K[1][2], 1]  # Open3D PinholeCameraIntrinsic json
    (zone / "intrinsic.json").write_text(json.dumps({"width": W, "height": H, "intrinsic_matrix": col_major}, indent=1))
    Kc = cal.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, CW, CH)
    dist = list(cal.getDistortionCoefficients(dai.CameraBoardSocket.CAM_A))  # OpenCV order, 14 terms
    (zone / "intrinsic_color.json").write_text(json.dumps({"width": CW, "height": CH, "K": Kc, "dist": dist}, indent=1))
    return K


def ir_on(dev: dai.Device, dots: float = 0.7) -> bool:
    """OAK-D Pro: the IR dot projector gives blank walls texture for stereo (the RGB has an IR-cut filter)."""
    try:
        return bool(dev.getIrDrivers()) and dev.setIrLaserDotProjectorIntensity(dots)
    except Exception:
        return False
