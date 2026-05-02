#!/bin/bash
# run_test_py.sh — 用官方 nvidia/cuopt image + 自訂 libcuopt.so 執行 Python 測試
#
# 工作原理：
#   官方 image 有完整的 Python 環境（pylibraft, rmm, cupy, cuopt...）
#   把 cpp/build/libcuopt.so（已做 ELF patch）掛載覆蓋官方的 pre-built .so
#
# compat 設定（處理 Ubuntu 24.04 build → Ubuntu 22.04 runtime 的 ABI 差異）：
#   - LD_PRELOAD libglibc_compat.so：提供 __isoc23_fscanf/__isoc23_strtol
#   - ELF patch（build 後自動執行）：GLIBC_2.38→2.17, GLIBCXX_3.4.31→3.4.30
#   - LD_LIBRARY_PATH：官方 image 的 CUDA libs（cudss 等）
#
# 注意：TBB ABI 差異（Ubuntu 24.04 TBB 2021.11 vs bundled TBB in 官方 image）
#   若出現 libtbb symbol 錯誤，代表你需要使用 Ubuntu 22.04 dev image 來 build
#   （GitHub Actions 的 workcc/cuopt-dev:26.6.0a 是 Ubuntu 22.04，完全相容）
#
# 用法：
#   bash run_test_py.sh                              # 互動 bash
#   bash run_test_py.sh pytest python/cuopt/cuopt/tests  # Python 測試
#   bash run_test_py.sh ctest --test-dir cpp/build -j4   # C++ 測試

set -eo pipefail

OFFICIAL_IMAGE="nvidia/cuopt:26.6.0a-cuda12.9-py3.14"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIBCUOPT_SO="${REPO_ROOT}/cpp/build/libcuopt.so"
GLIBC_SHIM="${REPO_ROOT}/ci/docker/libglibc_compat.so"

if ! docker info &>/dev/null; then
    echo "[錯誤] Docker daemon 未啟動"
    exit 1
fi

if [ ! -f "${LIBCUOPT_SO}" ]; then
    echo "[錯誤] ${LIBCUOPT_SO} 不存在，請先執行 bash run_dev.sh cuopt-build"
    exit 1
fi

# 官方 image 中 libcuopt.so 的路徑（Python 3.14）
OFFICIAL_SO="/usr/local/lib/python3.14/dist-packages/libcuopt/lib64/libcuopt.so"
# 官方 image 的 CUDA libs（cudss 等）
NVIDIA_LIBS="/usr/local/lib/python3.14/dist-packages/nvidia/cu12/lib"

GPU_FLAG="--gpus all"
if ! docker run --rm --gpus all nvidia/cuda:12.9.0-base-ubuntu24.04 \
        nvidia-smi &>/dev/null 2>&1; then
    echo "[警告] GPU 無法存取，以無 GPU 模式啟動"
    GPU_FLAG=""
fi

echo ""
echo ">>> 啟動 cuOpt Python 測試容器"
echo "    Base    : ${OFFICIAL_IMAGE}"
echo "    so      : ${LIBCUOPT_SO}"
echo "    Source  : ${REPO_ROOT}  → /cuopt"
echo ""

# 設定環境：安裝 Ubuntu 22.04 系統 TBB（apt 安裝需要 update） + GLIBC compat shim + CUDA libs
# libtbb12 提供 libtbb.so.12（與我們 Ubuntu 24.04 build 的 libcuopt.so 相容）
SETUP_CMD="apt-get update -qq 2>/dev/null && apt-get install -y -q libtbb12 2>/dev/null | tail -1 && export LD_LIBRARY_PATH=${NVIDIA_LIBS}"
if [ -f "${GLIBC_SHIM}" ]; then
    SETUP_CMD="${SETUP_CMD} && export LD_PRELOAD=/tmp/libglibc_compat.so"
fi

VOLUMES="-v ${REPO_ROOT}:/cuopt -v ${LIBCUOPT_SO}:${OFFICIAL_SO}"
if [ -f "${GLIBC_SHIM}" ]; then
    VOLUMES="${VOLUMES} -v ${GLIBC_SHIM}:/tmp/libglibc_compat.so"
fi

if [ $# -eq 0 ]; then
    eval "docker run ${GPU_FLAG} --rm -it --shm-size=8g ${VOLUMES} --workdir /cuopt ${OFFICIAL_IMAGE} bash --init-file <(echo '${SETUP_CMD}')"
else
    eval "docker run ${GPU_FLAG} --rm -it --shm-size=8g ${VOLUMES} --workdir /cuopt ${OFFICIAL_IMAGE} bash -c '${SETUP_CMD} && $*'"
fi
