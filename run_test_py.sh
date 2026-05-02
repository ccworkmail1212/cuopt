#!/bin/bash
# run_test_py.sh — 用官方 nvidia/cuopt image + 自訂 libcuopt.so 執行測試
#
# 工作原理：
#   1. 官方 image 已有完整 Python 環境（pylibraft, rmm, cupy, cuopt...）
#   2. 把 cpp/build/libcuopt.so 掛載覆蓋官方的 pre-built .so
#   3. 在官方 Python 環境中跑測試，但使用你改過的 C++ 程式碼
#
# 用法：
#   bash run_test_py.sh                              # 互動 bash
#   bash run_test_py.sh pytest python/cuopt/cuopt/tests  # Python 測試
#   bash run_test_py.sh ctest --test-dir cpp/build -j4   # C++ 測試

set -eo pipefail

OFFICIAL_IMAGE="nvidia/cuopt:26.6.0a-cuda12.9-py3.14"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIBCUOPT_SO="${REPO_ROOT}/cpp/build/libcuopt.so"

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

GPU_FLAG="--gpus all"
if ! docker run --rm --gpus all nvidia/cuda:12.9.0-base-ubuntu24.04 \
        nvidia-smi &>/dev/null 2>&1; then
    echo "[警告] GPU 無法存取，以無 GPU 模式啟動"
    GPU_FLAG=""
fi

echo ""
echo ">>> 啟動 cuOpt Python 測試容器"
echo "    Base  : ${OFFICIAL_IMAGE}"
echo "    so    : ${LIBCUOPT_SO} → ${OFFICIAL_SO}"
echo "    Source: ${REPO_ROOT}  → /cuopt"
echo ""

docker run \
    ${GPU_FLAG} \
    --rm \
    -it \
    --shm-size=8g \
    -v "${REPO_ROOT}:/cuopt" \
    -v "${LIBCUOPT_SO}:${OFFICIAL_SO}" \
    --workdir /cuopt \
    "${OFFICIAL_IMAGE}" \
    "$@"
