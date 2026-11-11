#!/usr/bin/env bash
# HOST side (devcontainer initializeCommand): builds 3dgrut-twin if missing (--force rebuilds), makes ~/Robotics/data, X cookie.
set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FORCE=0
for arg in "$@"; do [[ "$arg" == "--force" ]] && FORCE=1; done

mkdir -p "${HOME}/Robotics/data/office_scan"
if [[ "$FORCE" -eq 0 ]] && docker image inspect 3dgrut-twin > /dev/null 2>&1; then
    echo "[devcontainer] 3dgrut-twin already exists, skipping build. (pass --force to rebuild)"
else
    echo "[devcontainer] Building 3dgrut-twin (first time ~50-60 min: COLMAP + the CUDA extensions compile)..."
    git -C "${REPO_ROOT}" submodule update --init --recursive
    docker build --network host -f "${REPO_ROOT}/.devcontainer/Dockerfile" -t 3dgrut-twin "${REPO_ROOT}"
fi
bash "${REPO_ROOT}/.devcontainer/host-xauth.sh"
