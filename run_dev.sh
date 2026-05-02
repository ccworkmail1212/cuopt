#!/bin/bash
# run_dev.sh — 啟動 cuOpt 開發容器
#
# 功能：
#   - 自動 build dev image（第一次執行；之後因 conda layer cache 很快）
#   - 以 --gpus all 啟動容器，source code 從本機 mount 進去
#   - ccache 用 Docker volume 持久化，第二次 build 速度大幅提升
#
# 用法：
#   bash run_dev.sh               # 互動模式（進入 bash）
#   bash run_dev.sh ./build.sh    # 直接執行指令後退出
#   bash run_dev.sh ctest --test-dir cpp/build -j4

set -eo pipefail

DEV_IMAGE="cuopt-dev-official:26.6.0a"
CCACHE_VOLUME="cuopt-ccache"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── 1. 確認 Docker daemon 運行中 ────────────────────────────────────────────
if ! docker info &>/dev/null; then
    echo "[錯誤] Docker daemon 未啟動"
    echo "  請先執行：sudo dockerd &"
    exit 1
fi

# ── 2. 確認 GPU 支援（nvidia-container-toolkit）──────────────────────────────
GPU_FLAG="--gpus all"
if ! docker run --rm --gpus all nvidia/cuda:12.9.0-base-ubuntu24.04 \
        nvidia-smi &>/dev/null 2>&1; then
    echo "[警告] GPU 無法存取（nvidia-container-toolkit 未安裝或驅動未就緒）"
    echo "  將以無 GPU 模式啟動（只能 build，無法跑 solve 測試）"
    GPU_FLAG=""
fi

# ── 3. 建立 dev image（若還沒有）────────────────────────────────────────────
if ! docker image inspect "$DEV_IMAGE" &>/dev/null; then
    echo ">>> 第一次執行，建立 cuopt-dev image..."
    echo "    （conda env 與已有 build cache 共用，通常只需幾分鐘）"
    docker build \
        -f "$REPO_ROOT/Dockerfile.dev" \
        -t "$DEV_IMAGE" \
        --progress=plain \
        "$REPO_ROOT"
    echo ">>> Dev image 建立完成"
fi

# ── 4. 建立 ccache volume（若還沒有）────────────────────────────────────────
if ! docker volume inspect "$CCACHE_VOLUME" &>/dev/null; then
    docker volume create "$CCACHE_VOLUME" >/dev/null
fi

# ── 5. 啟動容器 ──────────────────────────────────────────────────────────────
echo ""
echo ">>> 啟動 cuOpt 開發容器"
echo "    Source  : $REPO_ROOT  →  /cuopt"
echo "    ccache  : $CCACHE_VOLUME  →  /root/.cache/ccache"
echo "    GPU     : ${GPU_FLAG:-無（無 GPU 模式）}"
echo ""

docker run \
    $GPU_FLAG \
    --rm \
    -it \
    --shm-size=8g \
    -v "$REPO_ROOT:/cuopt" \
    -v "$CCACHE_VOLUME:/root/.cache/ccache" \
    -v "$REPO_ROOT/ci/docker/build_cuopt.sh:/usr/local/bin/cuopt-build" \
    --workdir /cuopt \
    -e PARALLEL_LEVEL="${PARALLEL_LEVEL:-$(nproc)}" \
    "$DEV_IMAGE" \
    "$@"
