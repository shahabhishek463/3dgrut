# CLAUDE.md — 3DGRUT Setup & Workflow Guide

## Environment
- **Cloud GPU**: NVIDIA L4 (Ada Lovelace, sm_89, 24GB VRAM) at 164.52.192.207
- **Local machine**: ASUS ROG Strix G16 with RTX 5070 Ti Laptop (Blackwell, sm_120, 11GB VRAM)
- **Docker image**: `3dgrut:latest` (built with CUDA 12.8.1, 21.1GB)
- **Conda env**: `/root/miniconda3/envs/3dgrut` — installed via `install_env.sh` (alternative to Docker)

---

## 1. COLMAP Reconstruction (Cloud GPU)

### COLMAP v4.0.4 option renames (IMPORTANT)

COLMAP v4.0.4 renamed several CLI options. Old names are silently ignored:

| Old name (broken) | New name (v4.0.4+) |
|---|---|
| `--SiftExtraction.max_image_size` | `--FeatureExtraction.max_image_size` |
| `--SiftExtraction.estimate_affine_shape` | `--FeatureExtraction.estimate_affine_shape` |
| `--SiftExtraction.domain_size_pooling` | `--FeatureExtraction.domain_size_pooling` |
| `--SiftMatching.use_gpu` | `--FeatureMatching.use_gpu` |

### NuRec-recommended COLMAP settings (DSP-SIFT + PINHOLE + exhaustive)

These settings follow the official NuRec documentation for best reconstruction quality.
Run these inside the Docker container with `/root/colmap` mounted at `/working`:

```bash
docker run --rm -it \
    -v /root/colmap:/working \
    --gpus=all --runtime=nvidia \
    colmap/colmap:latest bash

# Inside container:
# Step 1: Feature extraction with DSP-SIFT
colmap feature_extractor \
    --database_path /working/myroom/database.db \
    --image_path /working/myroom/images \
    --ImageReader.camera_model PINHOLE \
    --FeatureExtraction.estimate_affine_shape 1 \
    --FeatureExtraction.domain_size_pooling 1 \
    --FeatureExtraction.max_image_size 1600 \
    --FeatureExtraction.use_gpu 1

# Step 2: Exhaustive matching
colmap exhaustive_matcher \
    --database_path /working/myroom/database.db \
    --FeatureMatching.use_gpu 1

# Step 3: Sparse reconstruction
colmap mapper \
    --database_path /working/myroom/database.db \
    --image_path /working/myroom/images \
    --output_path /working/myroom/sparse
```

Result: 165/196 images registered, 107,600 3D points, reprojection error 1.09px.
Output: `/root/colmap/myroom/sparse/1/` (cameras.bin, images.bin, points3D.bin).

### Dataset: mantra_office_2 (518 images)

```bash
docker run --rm -it \
    -v /root/colmap:/working \
    --gpus=all --runtime=nvidia \
    colmap/colmap:latest \
    colmap automatic_reconstructor \
        --image_path ./mantra_office_2/images \
        --workspace_path ./mantra_office_2
```

**Issues encountered:**
- `automatic_reconstructor` picked `sparse/1` (13 images) instead of `sparse/0` (518 images) for dense — re-ran dense manually on sparse/0
- Dense fusion produced 0 points without `--PatchMatchStereo.filter`

**Fix — manual dense pipeline on sparse/0:**
```bash
colmap image_undistorter \
    --image_path ./mantra_office_2/images \
    --input_path ./mantra_office_2/sparse/0 \
    --output_path ./mantra_office_2/dense/full \
    --output_type COLMAP

colmap patch_match_stereo \
    --workspace_path ./mantra_office_2/dense/full \
    --workspace_format COLMAP \
    --PatchMatchStereo.geom_consistency true \
    --PatchMatchStereo.filter true

colmap stereo_fusion \
    --workspace_path ./mantra_office_2/dense/full \
    --workspace_format COLMAP \
    --input_type geometric \
    --output_path ./mantra_office_2/dense/full/fused.ply
```

**Result**: 9,509 fused points in `fused.ply`.

---

### Dataset: myroom (196 images, extracted from MyroomNurec zip)

Images at `/root/colmap/myroom/images/` — 196 JPGs from `MyroomNurec-20260607T140015Z-3-001.zip`.
Note: only zip part 3 of 3 was available — parts 1 & 2 (~61 additional images) are missing.

```bash
docker run --rm -it \
    -v /root/colmap:/working \
    --gpus=all --runtime=nvidia \
    colmap/colmap:latest \
    colmap automatic_reconstructor \
        --image_path ./myroom/images \
        --workspace_path ./myroom \
        --use_gpu 1
```

**Result (automatic_reconstructor)**: 196/196 images registered, 143,509 3D points, mean reprojection error 1.17px (~42 min).
Output: `/root/colmap/myroom/sparse/0/` (cameras.bin, images.bin, points3D.bin).

**Result (NuRec DSP-SIFT + exhaustive)**: 165/196 images registered, 107,600 3D points, reprojection error 1.09px.
Output: `/root/colmap/myroom/sparse/1/` (cameras.bin, images.bin, points3D.bin).

Note: `sparse/1` registers fewer images but has better reprojection error. Either works for training.
Note: Ceres GPU bundle adjustment falls back to CPU ("compiled without cuDSS support") — non-fatal.

---

## 2. 3DGRUT Training

### Setup: Conda (on host, no Docker)

```bash
# Install miniconda if not present
curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh | bash
source /root/miniconda3/etc/profile.d/conda.sh

# Accept ToS (required before creating envs)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create 3dgrut environment
cd /root/3dgrut
bash install_env.sh
```

### Dependency fixes (required every session)

```bash
conda activate 3dgrut
pip install "numpy<2" "kornia==0.7.0"
```

- `numpy` 2.4.6 installed but PyTorch 2.1.2 compiled against NumPy 1.x → must downgrade
- `kornia` latest uses `any(dim=(-2,-1))` in `@torch.jit.script`, requires PyTorch ≥2.2 → pin to 0.7.0
- `plyfile` 1.1.4 requires numpy≥2.0 — pip conflict warning is non-fatal, training continues

### Config keys (IMPORTANT — common mistakes)

**IMPORTANT:** The correct Hydra key is `export_usd` (NOT `export_usdz`).
Official docs say `export_usdz.enabled=true` — **this is wrong** and crashes with:
`Key 'export_usdz' is not in struct`

**IMPORTANT:** Total training steps is `n_iterations=7000` (top-level key).
`training.max_steps` does NOT exist and crashes with: `Key 'training' is not in struct`

Correct keys (from `configs/base_gs.yaml`):
```yaml
n_iterations: 30000        # top-level — controls total training steps

export_ply:
  enabled: false           # set true to export PLY at end of training

export_usd:
  enabled: false
  apply_normalizing_transform: true   # centers scene near origin using camera poses
  format: standard                    # "nurec" for Omniverse, "standard" for ParticleField3DGaussianSplat

checkpoint:
  iterations: [7000, 30000]           # save checkpoints at these steps
```

### Training command (recommended — with PLY export)

```bash
# From /root/3dgrut with conda env activated:
python train.py --config-name apps/colmap_3dgut_mcmc.yaml \
    path=/root/colmap/myroom \
    out_dir=/root/3dgrut_output/myroom \
    experiment_name=myroom \
    n_iterations=7000 \
    'checkpoint.iterations=[7000]' \
    export_ply.enabled=true
```

- `export_ply.enabled=true` exports `export_last.ply` at end of training
- Use this instead of `export_usd.enabled=true` to avoid the broken USDZ reference bug (see Section 3)
- For 30k steps: `n_iterations=30000 'checkpoint.iterations=[7000,30000]'`
- OOM crash during final LPIPS eval is non-fatal — checkpoint and PLY are saved before that step

### Training results

| Run | Dataset | COLMAP sparse | Steps | PSNR | Output dir |
|-----|---------|---------------|-------|------|------------|
| Run 1 | mantra_office_2 (518 img) | sparse/0 | 30,000 | 21.4 dB | `/root/colmap/3dgrut_output/ours_30000/` |
| Run 2 | myroom (196 img) | sparse/0 | 30,000 | 26.0 dB | `.../myroom-0706_155515/` |
| Run 3 | myroom (196 img) | sparse/1 (NuRec DSP-SIFT) | 7,000 | 25.36 dB | `.../myroom-0806_124825/` |

Run 3: used NuRec COLMAP settings (sparse/1) and PLY export. Produced working USDZ for Isaac Sim.

---

## 3. USDZ Export for Isaac Sim

### Method A: PLY → USDZ (RECOMMENDED — confirmed working in Isaac Sim 5.1)

Export PLY during training, then convert:

```bash
# Step 1: train with PLY export (see training command above)
# PLY will be at: <out_dir>/<experiment>/<run_dir>/export_last.ply

# Step 2: convert PLY to USDZ (from /root/3dgrut with conda activated)
python -m threedgrut.export.scripts.ply_to_usd \
    /root/3dgrut_output/myroom/myroom/myroom-0806_124825/export_last.ply \
    --output_file /root/export_last.usdz
```

**Caveats:**
- NO normalization (dataset=None in ply_to_usd.py) — raw COLMAP coordinates ±167k units
- Overflow warning `RuntimeWarning: overflow encountered in cast` — non-fatal
- Scene is **upside-down** in Isaac Sim — rotate X:180 on the gauss prim, or fix permanently (see below)
- BUT: **room IS visible and recognizable in Isaac Sim 5.1** ✅ (confirmed at 7k steps)

This fix was discovered via GitHub issue comment by user Michelangelo-Y.

### Method B: Training with export_usd (BROKEN — avoid)

```bash
# DO NOT use — produces broken USDZ with missing internal reference
python train.py ... export_usd.enabled=true
# Results in: WARNING: Could not open asset @gaussians.usdc@ → black viewport
```

### Method C: Post-training export_usd.py (normalized, with dataset)

Use for a normalized scene (not upside-down, centered near origin):

```bash
python -m threedgrut.export.scripts.export_usd \
    --checkpoint /root/3dgrut_output/myroom/.../ours_7000/ckpt_7000.pt \
    --output /root/myroom_nurec.usdz \
    --format nurec \
    --dataset /root/colmap/myroom
```

- `--dataset` applies normalizing transform from camera poses (centers/scales scene)
- `--no-transform` skips normalization
- Omit `--format` for standard `ParticleField3DGaussianSplat` format (black in Isaac Sim 5.1)

### Coordinate system fix (upside-down scene)

PLY→USDZ produces an upside-down scene in Isaac Sim.

**Quick fix (Isaac Sim):** Select gauss prim → Properties → Rotate X: 180

**Permanent fix (modify export script):**
- `threedgrut/export/usd/nurec/exporter.py` line 175: `apply_coordinate_transform = kwargs.get("apply_coordinate_transform", False)`
- `threedgrut/export/transforms.py` line 99: `get_3dgrut_to_usdz_coordinate_transform()` — the flip matrix

Modify `ply_to_usd.py` to pass `apply_coordinate_transform=True`:
```python
# In threedgrut/export/scripts/ply_to_usd.py, change line 95:
exporter.export(model, output_path, dataset=None, conf=conf)
# To:
exporter.export(model, output_path, dataset=None, conf=conf, apply_coordinate_transform=True)
```

---

## 4. Isaac Sim Integration

### Rendering results (Isaac Sim 5.1)

| Method | Prim Type | Renders in Isaac Sim 5.1 | Notes |
|--------|-----------|--------------------------|-------|
| PLY→USDZ (Method A) | `OmniNuRecField` (Volume) | ✅ Room visible! | Blurry at 7k steps, upside-down |
| `export_usd.py` nurec format | `OmniNuRecField` (Volume) | ⚠️ Blurry sphere | Broken @gaussians.usdc@ reference |
| `export_usd.py` standard format | `ParticleField3DGaussianSplat` | ❌ Black viewport | No RTX hydra delegate in Isaac Sim 5.1 |

- Use **File → Open** for USDZ (not File → Import — that's for meshes only)
- The USDZ prim is at `/World/gaussians/Gaussians/gaussians` (3 levels deep — expand Stage panel)
- "No lights found" warning is irrelevant — Gaussians are self-emissive via spherical harmonics

### For robot simulation (correct approach)

**Gaussian splats are a rendering primitive, not a physics primitive.**
Robots cannot collide with Gaussians. For simulation you need a mesh.

**Workflow:**
1. Run dense reconstruction to get a mesh from COLMAP output
2. Import mesh into Isaac Sim as collision geometry
3. Optionally load the NuRec USDZ as a visual background overlay

**Dense reconstruction pipeline:**
```bash
docker run --rm -it \
    -v /root/colmap:/working \
    --gpus=all --runtime=nvidia \
    colmap/colmap:latest bash -c "
colmap image_undistorter \
    --image_path ./myroom/images \
    --input_path ./myroom/sparse/0 \
    --output_path ./myroom/dense/full \
    --output_type COLMAP && \
colmap patch_match_stereo \
    --workspace_path ./myroom/dense/full \
    --workspace_format COLMAP \
    --PatchMatchStereo.geom_consistency true \
    --PatchMatchStereo.filter true && \
colmap stereo_fusion \
    --workspace_path ./myroom/dense/full \
    --workspace_format COLMAP \
    --input_type geometric \
    --output_path ./myroom/dense/full/fused.ply && \
colmap poisson_mesher \
    --input_path ./myroom/dense/full/fused.ply \
    --output_path ./myroom/dense/full/meshed.ply"
```

Import `meshed.ply` into Isaac Sim via **File → Import → PLY Files**, then add `PhysicsMeshCollisionAPI` for robot collision.

---

## 4. Local Viewer (Docker on local machine)

**Download checkpoint from cloud GPU:**
```bash
# On cloud GPU — serve files:
python3 -m http.server 8080 --directory /root/3dgrut_output/myroom

# On local machine:
wget http://164.52.192.207:8080/myroom/myroom-0706_155515/ours_30000/ckpt_30000.pt
```

**Launch container with X11 forwarding:**
```bash
xhost +local:root
docker run --rm -it --gpus=all --net=host --ipc=host \
    --runtime=nvidia \
    -e DISPLAY=:1 \
    -v /tmp/.X11-unix/X1:/tmp/.X11-unix/X1 \
    -v $PWD:/workspace \
    --device=/dev/dri \
    3dgrut bash
```

**Run viewer:**
```bash
pip install -q "numpy<2" "kornia==0.7.0"
python playground.py --gs_object ./ckpt_30000.pt --buffer_mode host2device
```

---

## 5. PI-3 office twin from the OAK-D Pro (branch `abhishek/pi3-office-twin`, `mantra_twin/README.md`)

- **Devcontainer** (`.devcontainer/`): CUDA 12.8 + torch 2.8 via `install_env_uv.sh` (no conda, no host env) → runs on
  sm_86 … sm_120 (fixes Known Issue 1 for this path); GPU + OAK-D USB + X11; host `~/Robotics/data` = `/data`.
- **One scan, both outputs, one frame:** RGB-D SLAM poses → COLMAP text dataset (`to_colmap.py`) → 3DGUT; the depth TSDF
  → static triangle-mesh collider (`office_twin.usd`). Metric, floor z=0, Z up: no COLMAP scale ambiguity, not upside-down.
- `compose_twin.py` → `office_twin_scene.usda` = NuRec splat (visual) + collider. Isaac side: isaac-cobot
  `abhishek/pi3-office-twin` (`fr5_sim.py --office-twin`, `scripts/tools/office_twin_check.py`).

## Known Issues

### Issue 1 — Blackwell GPU (RTX 5070 Ti, sm_120) incompatibility
**Symptom:** `UserWarning: RTX 5070 Ti (sm_120) is not compatible with current PyTorch installation`

PyTorch 2.1.2 supports up to sm_90. RTX 5070 Ti is sm_120 (Blackwell).

**Fix:** Rebuild Docker image with PyTorch ≥2.5:
```dockerfile
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### Issue 2 — polyscope `update_data` Fortran array order
**Symptom:** `TypeError: update_data(): incompatible function arguments. Expected: numpy.ndarray[dtype=float32, shape=(*, 4), order='F']`

**Fix applied** in `threedgrut_playground/ps_gui.py` line 172:
```python
# After:
buffer.update_data(np.asfortranarray(tensor_array.detach().cpu().numpy().astype(np.float32)))
```

### Issue 3 — Mesa software rendering (no hardware OpenGL in container)
**Symptom:** `glx: failed to create dri3 screen`

**Fix:** Add `--device=/dev/dri` to docker run.

### Issue 4 — CUDA-GL interop crash with `device2device` mode
**Symptom:** Segfault after polyscope initializes

**Workaround:** Use `--buffer_mode host2device`

---

## Status

| Task | Status | Notes |
|------|--------|-------|
| numpy/kornia compatibility | ✅ Fixed | `pip install "numpy<2" "kornia==0.7.0"` |
| polyscope Fortran array | ✅ Fixed | Patched `ps_gui.py` line 172 |
| myroom training (30k steps) | ✅ Done | PSNR 26.0 dB, checkpoint saved |
| myroom training (7k steps, NuRec COLMAP) | ✅ Done | PSNR 25.36 dB, PLY exported |
| USDZ export — PLY→USDZ method | ✅ Working | Room visible in Isaac Sim 5.1! Upside-down coords. |
| USDZ export — export_usd.py (normalized) | ✅ Working | Use `--dataset` for normalizing transform |
| Isaac Sim rendering (PLY→USDZ) | ✅ Confirmed | NuRec OmniNuRecField renders room at 7k steps |
| Coordinate system (upside-down) | ⚠️ Partial | Quick: rotate X:180 in Isaac Sim; Perm: apply_coordinate_transform=True |
| Dense reconstruction for sim | 🔄 In progress (PI-3) | OAK-D Pro depth → TSDF collider; `mantra_twin/` |
| Mesa → NVIDIA OpenGL | ⚠️ Partial | Add `--device=/dev/dri` |
| PyTorch sm_120 (Blackwell) | 🔄 In progress | `.devcontainer/` image: CUDA 12.8 + torch 2.8 |
| Full interactive GUI (local) | ❌ Pending | Blocked by sm_120 rebuild |
