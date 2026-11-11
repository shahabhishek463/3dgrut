# 3DGRUT + the PI-3 office-twin tools (OAK-D Pro capture, Open3D SLAM, USD) in one image: nothing installed on the host.
# CUDA 12.8 + torch 2.8 cover the RTX 3060 (sm_86) up to Blackwell laptops (sm_120); 3dgrut's uv installer, not conda.
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8
# GL/X11/Qt libs: OpenCV + Open3D windows and the polyscope playground; libusb + udev: the OAK-D over USB.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git git-lfs curl wget ca-certificates build-essential ninja-build pkg-config \
        libgl1 libegl1 libglib2.0-0t64 libgomp1 libx11-6 libxext6 libxrender1 libxi6 libxrandr2 libxinerama1 \
        libxcursor1 libxkbcommon-x11-0 libdbus-1-3 libfontconfig1 libgl1-mesa-dri mesa-utils xauth \
        libusb-1.0-0 usbutils udev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /uvx /usr/local/bin/

# The env is built from a copy at /opt/3dgrut; the live repo (bind-mounted at /workspace/3dgrut) shadows it via PYTHONPATH.
WORKDIR /opt/3dgrut
COPY . .
RUN CUDA_HOME=/usr/local/cuda bash install_env_uv.sh 3dgrut

ENV VIRTUAL_ENV=/opt/3dgrut/.venv CUDA_HOME=/usr/local/cuda
ENV PATH=/opt/3dgrut/.venv/bin:${PATH} PYTHONPATH=/workspace/3dgrut

# PI-3 twin tools (mantra_twin/): OAK-D capture, dense RGB-D SLAM, USD export. usd-core comes with 3dgrut.
RUN uv pip install --python /opt/3dgrut/.venv/bin/python "depthai==2.33.0.0" "open3d==0.19.0"

RUN git config --global --add safe.directory '*'
WORKDIR /workspace/3dgrut
