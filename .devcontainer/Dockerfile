# PI-3 digital twin, one image: COLMAP (thirdparty/colmap) + 3DGRUT (this repo) + the OAK-D Pro camera (twin/capture).
# CUDA 12.8 + torch 2.8 cover the RTX 3060 (sm_86) up to Blackwell laptops (sm_120); nothing is installed on the host.
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8
# GL/X11/Qt libs: OpenCV + Open3D windows, the COLMAP GUI, the polyscope playground; libusb + udev: the OAK-D over USB.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git git-lfs curl wget ca-certificates build-essential ninja-build pkg-config \
        libgl1 libegl1 libglib2.0-0t64 libgomp1 libx11-6 libxext6 libxrender1 libxi6 libxrandr2 libxinerama1 \
        libxcursor1 libxkbcommon-x11-0 libdbus-1-3 libfontconfig1 libgl1-mesa-dri mesa-utils xauth \
        libusb-1.0-0 usbutils udev \
    && rm -rf /var/lib/apt/lists/*

# --- COLMAP, built from the submodule the way its own docker/Dockerfile does (CUDA SIFT + GUI + MVS).
# OpenBLAS instead of Intel MKL (no extra licence, ~2 GB smaller); ONNX features off (the NVIDIA workflow uses SIFT).
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake libboost-program-options-dev libboost-graph-dev libboost-system-dev libeigen3-dev \
        libopenimageio-dev openimageio-tools libmetis-dev libgoogle-glog-dev libsqlite3-dev libglew-dev \
        qt6-base-dev libqt6opengl6-dev libqt6openglwidgets6 libqt6svg6-dev libcgal-dev libceres-dev \
        libcurl4-openssl-dev libssl-dev libopenblas-dev \
    && rm -rf /var/lib/apt/lists/* && mkdir -p /usr/include/opencv4
COPY thirdparty/colmap /opt/colmap-src
ARG COLMAP_CUDA_ARCHS="75;86;89;120"
RUN cmake -S /opt/colmap-src -B /opt/colmap-build -GNinja -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="${COLMAP_CUDA_ARCHS}" -DONNX_ENABLED=OFF -DTESTS_ENABLED=OFF \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
    && ninja -C /opt/colmap-build install && rm -rf /opt/colmap-build
ENV QT_XCB_GL_INTEGRATION=xcb_egl

# --- 3DGRUT: its own uv installer (no conda). Built from a copy at /opt/3dgrut; the live repo (bind-mounted at
# /workspace/3dgrut) shadows it via PYTHONPATH, so code edits need no rebuild.
COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /uvx /usr/local/bin/
WORKDIR /opt/3dgrut
COPY . .
RUN CUDA_HOME=/usr/local/cuda bash install_env_uv.sh 3dgrut

ENV VIRTUAL_ENV=/opt/3dgrut/.venv CUDA_HOME=/usr/local/cuda
ENV PATH=/opt/3dgrut/.venv/bin:${PATH} PYTHONPATH=/workspace/3dgrut

# --- OAK-D Pro capture + dense RGB-D SLAM + USD tools (twin/). usd-core comes with 3DGRUT.
RUN uv pip install --python /opt/3dgrut/.venv/bin/python "depthai==2.33.0.0" "open3d==0.19.0"

RUN git config --global --add safe.directory '*'
WORKDIR /workspace/3dgrut
