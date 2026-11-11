# mantra_twin: office digital twin from one OAK-D Pro scan (Jira PI-3)

One walk-through with the OAK-D Pro gives you two things in the **same frame** (metres, floor at z = 0, Z up):
- the **collision mesh**, from depth (what robots hit in Isaac Sim);
- the **3DGRUT splat**, from the 1080p RGB, trained on the SLAM poses (what you see).

The SLAM poses replace COLMAP, so the splat comes out at real scale and right side up. Ganesh's notes in
`CLAUDE.md` describe the COLMAP route, which gives an arbitrary scale and an upside-down scene.

```
oak_record.py (per zone) ─► reconstruct.py ─► merge_zones.py ─► finalize.py ─► mesh_to_usd.py ─► office_twin.usd (collider)
   colour 1080p, depth 720p    GPU dense SLAM   zones registered   level, crop,    static triangle-mesh collider
   live map on the GPU                          + re-fused          decimate
                                       └──────────────────────► to_colmap.py ─► train.py (3DGUT) ─► office_splat.usdz
                                         keyframes, undistorted, poses in the twin frame      ply_to_usd
                                                                            compose_twin.py ─► office_twin_scene.usda
```

## 1. Open the devcontainer (once per machine)
You need Docker, the NVIDIA Container Toolkit and an NVIDIA driver of 570 or newer. Nothing else is installed on
the host.
```bash
git clone --recursive -b abhishek/pi3-office-twin https://github.com/shahabhishek463/3dgrut.git
```
- **VS Code / Antigravity:** open the folder, then choose **Reopen in Container**.
  - The first build takes 30–40 minutes, because the CUDA extensions compile. The image is about 25 GB.
- **Terminal only:**
  ```bash
  bash .devcontainer/build-images.sh && docker compose -f .devcontainer/docker-compose.devcontainer.yaml up -d
  docker exec -it 3dgrut-twin bash
  ```
- The host's `~/Robotics/data` appears as **`/data`** in the container. Scans go to `/data/office_scan/<zone>/`.
- Plug the OAK into a **USB 3 port before the container starts**. The container is privileged and sees the whole
  USB tree, so no udev rule is needed.
- The container covers the RTX 3060 (sm_86) up to Blackwell laptops (sm_120). It uses CUDA 12.8 and torch 2.8,
  installed by 3dgrut's own uv installer inside the image.

## 2. Check the camera
```bash
python mantra_twin/oak_check.py --show      # must end with: USB SUPER ... PASS
```
- About 45% of the view having no depth is normal across a room: everything beyond 4 m is cut, and the left edge
  band is seen by only one stereo camera.
- The IR dot projector switches on by itself on the Pro.

## 3. Scan
```bash
python mantra_twin/oak_record.py bench --max-minutes 12      # then desks, corridor, ... one zone per recording
```
- **Keys** (click the window first): **SPACE** pauses or resumes; hold the camera still while paused. **q** stops
  and saves. Never use Ctrl+C, because it skips the save.
- **Prepare the room:** lights on, blinds half closed, nobody walking through, and a **1 m tape** on the floor.
- **One zone = three laps in one recording:**
  1. Level at chest height.
  2. Tilted about 30–45° down, for the floor, desk tops and legs.
  3. Tilted about 30° up, for the upper walls. Don't aim straight at the ceiling.
- Walk slowly (no faster than 0.3 m/s) and turn slowly. **Finish where you started.**
- Each zone must **overlap the previous one by about 1 m**.
- Watch the live map panel. If the dot turns red ("TRACKING LOST"), walk back to an area you've already scanned.
- One minute of recording is about 1 GB.

## 4. Build the collider and the splat dataset
```bash
mantra_twin/build_twin.sh /data/office_scan bench desks corridor          # zones in walking order
python mantra_twin/measure.py /data/office_scan/final/office_visual.ply --expect 1.0   # click the tape's ends: within ±2%
```
- Look at `final/preview.png`, which shows a top view and a front view.
- `merge_zones` prints a `fitness` for each zone. Below 0.15 means too little overlap, so rescan with more.
- To crop the real FR5 bench (the sim models it itself), re-run `finalize.py` with `--exclude-box XMIN YMIN XMAX YMAX`
  in the levelled frame.

## 5. Train the splat and compose the twin
```bash
mantra_twin/train_splat.sh /data/office_scan 30000      # 3DGUT + MCMC; 7000 for a quick look
```
This writes the following into `/data/office_scan/final/`:
- `office_twin.usd`: the collider, plus the coloured mesh;
- `office_splat.usdz`: the NuRec splat;
- **`office_twin_scene.usda`**: both together. This is the file that goes to Isaac Sim.

## 6. Isaac Sim (isaac-cobot, branch `abhishek/pi3-office-twin`)
- The twin goes around Nakul's FR5 bench with `fr5_sim.py --office-twin <office_twin_scene.usda>`.
- The collision gate is `scripts/tools/office_twin_check.py`.
- Both are described in that repo's `docs/office-twin.md`.

## Tested (2026-11-11), on Open3D's Lounge RGB-D sample split into 2 overlapping zones
- SLAM ran at about 85 fps with no lost frames. Zone registration fitness was 0.44 at 21 mm RMSE.
- The live map kept up at 15 fps with nothing lost.
- **Pose check:** the mesh raycast from every 15th exported COLMAP pose matched that frame's measured depth to a
  median of **7–46 mm**.
- 212 keyframes were picked from 1700 frames.
- The crop box removed everything above the floor inside it.
- **Inside the devcontainer** (RTX 3060, driver 580, CUDA 12.8, torch 2.8):
  - `build_twin.sh` took 1 min.
  - `train_splat.sh ... 3000` ran at 23 it/s (128 s) and gave a test PSNR of 19.6 dB. That's only 3k steps on 640×480
    sample images, so 30k steps on the OAK's 1080p should take about 25 min on the 3060.
  - The **splat lands on the collider**: the median distance from a Gaussian to the mesh is 1.4 cm.
  - `office_twin_scene.usda` has identity transforms, Z up and metres.
- Files written from the container are owned by root on the host. To clean up, run `sudo chown -R $USER ~/Robotics/data`.
