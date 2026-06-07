# CLAUDE.md — 3DGRUT Setup & Workflow Guide

## What We Did

### Environment
- **Cloud GPU**: NVIDIA L40S (Ada Lovelace, sm_89, 46GB VRAM) at 164.52.192.207
- **Local machine**: ASUS ROG Strix G16 with RTX 5070 Ti Laptop (Blackwell, sm_120, 11GB VRAM)
- **Docker image**: `3dgrut:latest` (pre-built, 23.7GB)

---

### 1. COLMAP Reconstruction (Cloud GPU)

Ran COLMAP on `mantra_office_2` dataset (518 images) to produce sparse + dense reconstruction.

```bash
# Inside colmap/colmap docker container, mounted at /working
colmap automatic_reconstructor \
    --image_path ./mantra_office_2/images \
    --workspace_path ./mantra_office_2
```

**Issues encountered:**
- `automatic_reconstructor` used `sparse/1` (13 images) instead of `sparse/0` (518 images) for dense reconstruction
- Dense fusion produced 0 points because `--PatchMatchStereo.filter` was not enabled

**Fix — re-ran dense pipeline manually on sparse/0:**
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

### 2. 3DGRUT Training (Cloud GPU, Docker)

**Dependency fixes required inside the container:**
```bash
pip install "numpy<2" "kornia==0.7.0"
```

- `numpy` 2.4.6 was installed but PyTorch 2.1.2 was compiled against NumPy 1.x → downgrade to `numpy<2`
- `kornia` version used `any(dim=(-2,-1))` in a `@torch.jit.script` function, which requires PyTorch ≥2.2 → pin to `kornia==0.7.0`

**Training command:**
```bash
docker run -d --name 3dgrut_train \
    --gpus=all --net=host --ipc=host --runtime=nvidia \
    -v /root/colmap/mantra_office_2:/data/mantra_office_2:ro \
    -v /root/colmap/3dgrut_output:/output \
    3dgrut:latest \
    bash -c 'source /opt/conda/etc/profile.d/conda.sh && conda activate 3dgrut && \
        pip install -q "numpy<2" "kornia==0.7.0" && \
        python train.py --config-name apps/colmap_3dgut_mcmc.yaml \
            "path=/data/mantra_office_2" \
            "out_dir=/output" \
            "experiment_name=mantra_office"'
```

**Result** (30,000 steps, ~20 min on L40S):
- PSNR: 21.4 dB
- SSIM: 0.833
- Checkpoint: `ours_30000/ckpt_30000.pt` (675MB)

---

### 3. Local Viewer (Docker on local machine)

**Download checkpoint from cloud GPU:**
```bash
# Cloud GPU serves file over HTTP (kill after download)
python3 -m http.server 8080 --directory /path/to/ours_30000

# Local machine
wget http://164.52.192.207:8080/ckpt_30000.pt
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

## Known Issues & TODO for GUI

### Issue 1 — Blackwell GPU (RTX 5070 Ti, sm_120) incompatibility
**Symptom:** `UserWarning: RTX 5070 Ti (sm_120) is not compatible with current PyTorch installation`

PyTorch 2.1.2 in the Docker image only supports up to sm_90. The local machine is Blackwell (sm_120).

**Fix needed:** Rebuild the Docker image with PyTorch ≥2.5 (which added sm_120 support):
```dockerfile
# In Dockerfile, update PyTorch installation:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

---

### Issue 2 — polyscope `update_data` Fortran array order
**Symptom:**
```
TypeError: update_data(): incompatible function arguments.
Expected: numpy.ndarray[dtype=float32, shape=(*, 4), order='F']
```

**Fix applied** in `threedgrut_playground/ps_gui.py` line 172:
```python
# Before:
buffer.update_data(tensor_array.detach().cpu().numpy())

# After:
buffer.update_data(np.asfortranarray(tensor_array.detach().cpu().numpy().astype(np.float32)))
```

---

### Issue 3 — Mesa software rendering (no hardware OpenGL in container)
**Symptom:** `glx: failed to create dri3 screen`, falls back to Mesa iris driver

**Fix:** Pass `/dev/dri` to container:
```bash
docker run ... --device=/dev/dri ...
```

---

### Issue 4 — CUDA-GL interop crash with `device2device` mode
**Symptom:** Segfault immediately after polyscope initializes

**Workaround:** Use `--buffer_mode host2device` instead of default `device2device`

---

## Summary — What Still Needs to Be Done for Full GUI

| Task | Status | Fix |
|------|--------|-----|
| numpy/kornia compatibility | ✅ Fixed | `pip install "numpy<2" "kornia==0.7.0"` |
| polyscope Fortran array | ✅ Fixed | Patched `ps_gui.py` line 172 |
| Mesa → NVIDIA OpenGL | ⚠️ Partial | Add `--device=/dev/dri` to docker run |
| PyTorch sm_120 (Blackwell) support | ❌ Needs fix | Rebuild image with PyTorch ≥2.5 + CUDA 12.8 |
| Full interactive GUI working | ❌ Pending | Blocked by sm_120 PyTorch rebuild |
