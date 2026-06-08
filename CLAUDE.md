# CLAUDE.md — 3DGRUT Setup & Workflow Guide

## Environment
- **Cloud GPU**: NVIDIA L4 (Ada Lovelace, sm_89, 24GB VRAM) at 164.52.192.207
- **Local machine**: ASUS ROG Strix G16 with RTX 5070 Ti Laptop (Blackwell, sm_120, 11GB VRAM)
- **Docker image**: `3dgrut:latest` (built with CUDA 12.8.1, 21.1GB)

---

## 1. COLMAP Reconstruction (Cloud GPU)

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

**Result**: 196/196 images registered, 143,509 3D points, mean reprojection error 1.17px (~42 min).
Output: `/root/colmap/myroom/sparse/0/` (cameras.bin, images.bin, points3D.bin).

Note: Ceres GPU bundle adjustment falls back to CPU ("compiled without cuDSS support") — non-fatal.

---

## 2. 3DGRUT Training (Cloud GPU, Docker)

### Dependency fixes (required every container run)

```bash
pip install "numpy<2" "kornia==0.7.0"
```

- `numpy` 2.4.6 installed but PyTorch 2.1.2 compiled against NumPy 1.x → must downgrade
- `kornia` latest uses `any(dim=(-2,-1))` in `@torch.jit.script`, requires PyTorch ≥2.2 → pin to 0.7.0
- `plyfile` 1.1.4 requires numpy≥2.0 — pip conflict warning is non-fatal, training continues

### Config key for USDZ export

**IMPORTANT:** The correct Hydra key is `export_usd` (NOT `export_usdz`).
Official docs say `export_usdz.enabled=true` — this is wrong and crashes with:
`Key 'export_usdz' is not in struct`

Correct keys (from `configs/base_gs.yaml`):
```yaml
export_usd:
  enabled: false
  path: ""
  apply_normalizing_transform: true   # centers scene near origin using camera poses
  format: standard                    # "nurec" for Omniverse, "standard" for ParticleField3DGaussianSplat
```

### Training command (with USDZ export)

```bash
docker run -d --name 3dgrut_train \
    --gpus=all --net=host --ipc=host --runtime=nvidia \
    -v /path/to/colmap/dataset:/data/dataset:ro \
    -v /path/to/output:/output \
    3dgrut:latest \
    bash -c 'source /opt/conda/etc/profile.d/conda.sh && conda activate 3dgrut && \
        pip install -q "numpy<2" "kornia==0.7.0" && \
        python train.py --config-name apps/colmap_3dgut_mcmc.yaml \
            path=/data/dataset \
            out_dir=/output \
            experiment_name=my_experiment \
            export_usd.enabled=true \
            export_usd.apply_normalizing_transform=true \
            export_usd.format=nurec'
```

The normalizing transform is computed from the training camera poses (dataset is available during training), so the exported USDZ will be correctly centered/scaled.

### Run 1 — mantra_office_2 (518 images)

**Result** (30,000 steps, ~20 min on L40S):
- PSNR: 21.4 dB, SSIM: 0.833
- Checkpoint: `/root/colmap/3dgrut_output/ours_30000/ckpt_30000.pt` (675MB)

### Run 2 — myroom (196 images) — without export_usd

```bash
docker run -d --name 3dgrut_myroom_train \
    --gpus=all --net=host --ipc=host --runtime=nvidia \
    -v /root/colmap/myroom:/data/myroom:ro \
    -v /root/3dgrut_output/myroom:/output \
    3dgrut:latest \
    bash -c 'source /opt/conda/etc/profile.d/conda.sh && conda activate 3dgrut && \
        pip install -q "numpy<2" "kornia==0.7.0" && \
        python train.py --config-name apps/colmap_3dgut_mcmc.yaml \
            path=/data/myroom \
            out_dir=/output \
            experiment_name=myroom'
```

**Result** (30,000 steps, ~3.5 hours on L4):
- PSNR: 26.0 dB (Frame 0) — excellent quality
- Checkpoint: `/root/3dgrut_output/myroom/myroom/myroom-0706_155515/ours_30000/ckpt_30000.pt` (676MB)
- Training rate: ~150 steps/min on L4
- OOM crash during final evaluation render (non-fatal — checkpoint is safe)
- USDZ NOT exported during training (export_usd.enabled not set) — exported manually post-training

### Post-training USDZ export (standalone script)

Use when training was done without `export_usd.enabled=true`:

```bash
docker run --rm --gpus=all --runtime=nvidia \
    -v /path/to/output:/output \
    -v /path/to/colmap/dataset:/data/dataset:ro \
    3dgrut:latest \
    bash -c 'source /opt/conda/etc/profile.d/conda.sh && conda activate 3dgrut && \
        pip install -q "numpy<2" "kornia==0.7.0" && \
        python -m threedgrut.export.scripts.export_usd \
            --checkpoint /output/ours_30000/ckpt_30000.pt \
            --output /output/myroom_nurec.usdz \
            --format nurec \
            --dataset /data/dataset'
```

- `--dataset` is required for the normalizing transform to be applied (without it, raw COLMAP coordinates ±1000+ units)
- `--no-transform` skips normalization (default is to apply it when dataset is provided)
- Omit `--format` for standard `ParticleField3DGaussianSplat` format

---

## 3. Isaac Sim Integration

### Findings (Isaac Sim 5.1)

| Format | Prim Type | Renders in Isaac Sim 5.1 | Notes |
|--------|-----------|--------------------------|-------|
| `standard` | `ParticleField3DGaussianSplat` | ❌ Black viewport | No RTX hydra delegate for this type |
| `nurec` | `OmniNuRecField` (Volume) | ⚠️ Blurry sphere | Renders but poor quality; navigate camera inside |

- Use **File → Open** for USDZ (not File → Import — that's for meshes only)
- The USDZ prim is at `/World/gaussians/Gaussians/gaussians` (3 levels deep — expand in Stage panel)
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
| USDZ export (nurec + standard) | ✅ Done | Requires `--dataset` for normalizing transform |
| Isaac Sim rendering | ⚠️ Partial | NuRec renders (blurry); standard format black |
| Dense reconstruction for sim | ❌ Pending | Need for robot collision geometry |
| Mesa → NVIDIA OpenGL | ⚠️ Partial | Add `--device=/dev/dri` |
| PyTorch sm_120 (Blackwell) | ❌ Needs fix | Rebuild with PyTorch ≥2.5 + CUDA 12.8 |
| Full interactive GUI (local) | ❌ Pending | Blocked by sm_120 rebuild |
