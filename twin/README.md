# twin/: office digital twin, the NVIDIA workflow plus the OAK-D Pro (Jira PI-3)

This follows NVIDIA's [*How to Instantly Render Real-World Scenes in Interactive Simulation*](https://developer.nvidia.com/blog/how-to-instantly-render-real-world-scenes-in-interactive-simulation):
**photos → COLMAP → 3DGUT (`apps/colmap_3dgut_mcmc.yaml`) → USD → Isaac Sim**. The OAK-D Pro's depth adds two
things the blog doesn't give:
- **real scale and orientation**: COLMAP's model is snapped onto the measured camera positions, so it's metric, with
  the floor at z = 0 and Z up;
- **real collision geometry**: a mesh of the walls, desks and floor from depth, instead of a single ground plane.

## One repo, both projects' own layouts
```
3dgrut/                         3DGRUT, its layout untouched (threedgrut/, configs/, train.py, ...)
├── thirdparty/colmap/          COLMAP as a git submodule (shahabhishek463/colmap), its layout untouched
├── .devcontainer/              ONE image: COLMAP (built from thirdparty/colmap, CUDA + GUI) + 3DGRUT + the OAK camera
└── twin/                       the workflow, one folder per step
    ├── capture/   oak_check.py, oak_record.py (+ oak_common, live_map)   OAK-D Pro: RGB 1080p, depth 720p
    ├── slam/      reconstruct.py, merge_zones.py                         dense RGB-D SLAM per zone, zones fused
    ├── collision/ finalize.py, mesh_to_usd.py, measure.py, preview.py    levelled collider -> office_twin.usd
    ├── colmap/    keyframes.py, run_colmap.sh, align_metric.py           the blog's COLMAP step + metric alignment
    │              slam_to_colmap.py                                      backup: SLAM poses if COLMAP fails
    ├── splat/     train_splat.sh                                         the blog's 3DGUT step + USD export
    ├── isaac/     compose_twin.py                                        splat + collider -> office_twin_scene.usda
    └── build_twin.sh                                                     slam -> collision -> colmap, in one go
```
**Data layout**: `/data/<scan>/` in the container, `~/Robotics/data/<scan>/` on the host.

| Path | Contents |
|---|---|
| `<zone>/` | `color/`, `depth/`, `intrinsic*.json`, and the SLAM outputs `poses.npy` and `mesh.ply` |
| `merged/` | `fused.ply`, `zones.json` |
| `final/` | `office_twin.usd` (collider), `office_splat.usdz`, `office_twin_scene.usda`, `preview.png` |
| `colmap/` | `images/`, `database.db`, `sparse_raw/` (COLMAP's own result), `sparse/0` (metric), `align_report.json` |

`colmap/` is laid out the way 3DGRUT's colmap dataset reads it: `path/images` + `path/sparse/0`.

## 1. Open the devcontainer (once per machine)
You need Docker, the NVIDIA Container Toolkit and an NVIDIA driver of 570 or newer.
```bash
git clone --recursive -b abhishek/pi3-office-twin https://github.com/shahabhishek463/3dgrut.git ~/Robotics/3dgrut
cd ~/Robotics/3dgrut && bash .devcontainer/build-images.sh        # first time ~50-60 min
```
- **Antigravity / VS Code:** open the folder, then Ctrl+Shift+P → **Dev Containers: Reopen in Container**.
- **Terminal only:** `docker compose -f .devcontainer/docker-compose.devcontainer.yaml up -d && docker exec -it 3dgrut-twin bash`
- Everything below runs **inside** it, in `/workspace/3dgrut`. Nothing is pip-installed on the host.
- Plug the OAK into a USB 3 port before starting the container.

## 2. Capture: check the camera, then record the zones
```bash
python twin/capture/oak_check.py --show                        # ends with: USB SUPER ... PASS
python twin/capture/oak_record.py bench --max-minutes 12        # then desks, corridor, ... one zone per recording
```
- **Keys** (click the window first): **SPACE** pauses or resumes; hold the camera still while paused. **q** stops
  and saves. Never use Ctrl+C, because it skips the save.
- **Prepare the room:** lights on, blinds half closed, nobody walking through, and a **1 m tape** on the floor.
- **One zone = three laps in one recording:**
  1. Level at chest height.
  2. Tilted about 30–45° down, for the floor, desk tops and legs.
  3. Tilted about 30° up, for the upper walls. Not straight at the ceiling.
- Walk slowly (no faster than 0.3 m/s) and finish where you started.
- Each zone must overlap the previous one by about 1 m.
- If the live-map dot turns red, walk back to an area you've already scanned.

## 3. Build: SLAM → collider → COLMAP (the blog's step) → metric alignment
```bash
twin/build_twin.sh /data/office_scan bench desks corridor
python twin/collision/measure.py /data/office_scan/final/office_visual.ply --expect 1.0   # click the tape ends: ±2%
```
- The **COLMAP step** runs the blog's commands, with this COLMAP's option names (the `SiftExtraction.max_image_size`
  and `SiftMatching.use_gpu` names in the blog were renamed). It uses PINHOLE, a single camera with the OAK's
  calibration as the starting intrinsics, DSP-SIFT, `exhaustive_matcher` and `mapper`.
  - To look at the result: `colmap gui --import_path /data/office_scan/colmap/sparse/0 --database_path .../database.db --image_path .../images`
- **`align_metric.py`** runs `colmap model_aligner` against the OAK's camera positions and writes a report:
  - the share of images registered (at least 60%);
  - the median position error (under 10 cm) and the median rotation error (under 5°).
  - If it fails, the build falls back to the SLAM poses and says so. `colmap/poses_source.txt` records which route
    was used.
- To crop the real FR5 bench out of the collider, set `EXCLUDE_BOX="XMIN YMIN XMAX YMAX"` (levelled frame) before
  `build_twin.sh`.

## 4. Splat: 3DGUT (the blog's step) → USDZ → the twin scene
```bash
twin/splat/train_splat.sh /data/office_scan 30000          # apps/colmap_3dgut_mcmc.yaml; 7000 for a quick look
```
This writes `final/office_splat.usdz` and **`final/office_twin_scene.usda`**: the splat to look at plus the collider
for physics, in one frame. That's the file Isaac Sim opens. The USDZ comes from `ply_to_usd`, because in-training USD
export writes a broken reference (`CLAUDE.md` §3).

## 5. Isaac Sim: isaac-cobot, branch `abhishek/pi3-office-twin`
- `fr5_sim.py --office-twin office_twin_scene.usda` puts the twin around Nakul's FR5 bench.
- `scripts/tools/office_twin_check.py` is the collision gate.
- Both are described in `docs/office-twin.md` in that repo.

## Tested (2026-11-11), on Open3D's Lounge RGB-D sample split into 2 overlapping zones
- SLAM ran at about 85 fps with no lost frames. Zone registration fitness was 0.44 at 21 mm RMSE.
- The live map kept up at 15 fps.
- **Pose check:** the mesh raycast from every 15th keyframe pose matched that frame's measured depth to a median of
  **7–46 mm**.
- **Inside the devcontainer** (RTX 3060, driver 580, CUDA 12.8, torch 2.8):
  - the build took 1 min;
  - 3DGUT ran 3k steps at 23 it/s (128 s) and gave a test PSNR of 19.6 dB. That's only 640×480 images, so 30k steps
    at 1080p should take about 25 min;
  - the **splat lands on the collider**: the median distance from a Gaussian to the mesh is 1.4 cm;
  - `office_twin_scene.usda` has identity transforms, Z up and metres.
- Files written from the container are owned by root on the host. To clean up, run `sudo chown -R $USER ~/Robotics/data`.
