# cuOpt Docker 開發指南

## 總覽

本指南說明如何使用 cuOpt Docker images 進行開發與部署。

---

## 可用的 Docker Images

### H200 專用（公司 GPU）

| Image | 用途 | 大小 | Chunk 前綴 |
|---|---|---|---|
| `workcc/cuopt-full-h200:26.6` | **只能 run**，直接執行 Python 程式 | ~9.5GB | `cuopt-full-h200-part-XX` |
| `workcc/cuopt-full-h200-sm90a:26.6` | **可以 build**，含 build 環境 + rapids-cmake，可在 H200 跑 `cuopt-build` | ~10.8GB | `cuopt-full-h200-sm90a-part-XX` |

> ⚠️ **兩個 image 都只能在 H200（CUDA SM90a）上使用**
> 無法在其他 GPU（RTX 4060、A100、V100 等）上跑。

**如何選擇：**
- 只需要**執行** cuOpt 程式 → 用 `cuopt-full-h200`
- 需要**修改 C++ 並重新 build** → 用 `cuopt-full-h200-sm90a`

### 開發用（可重編，不限 GPU）

| Image | 用途 | 大小 |
|---|---|---|
| `workcc/cuopt-allinone:26.6` | 完整開發環境（改+build+run），**任何 GPU** | ~8.6GB |

> `cuopt-allinone` 包含 GCC 12 + nvcc + cmake + ccache，可在任何 GPU 上重新編譯。

---

## 離線帶進公司（chunk 方式）

大型 image 透過 Docker Hub chunk images 傳送，詳細說明請見：
https://github.com/ccworkmail1212/cuopt-docker/blob/main/OFFLINE_REASSEMBLE.md

快速組合範例：
```bash
# 只能 run（H200）
bash reassemble.sh cuopt-full-h200 26.6 aa ab ac ad

# 可以 build（H200）
bash reassemble.sh cuopt-full-h200-sm90a 26.6 aa ab ac ad

# 開發 build（任何 GPU）
bash reassemble.sh cuopt-allinone 26.6 aa ab ac ad ae af ag ah ai
```

---

## cuopt-allinone 開發流程

`workcc/cuopt-allinone:26.6` 是完整的開發 image：

```
你的電腦
  ├── ~/cuopt/          ← source code（掛入容器）
  └── docker run ...    ← 啟動容器執行，完成後自動關閉
```

### 步驟 1：取得 source code

```bash
git clone https://github.com/ccworkmail1212/cuopt.git ~/cuopt
cd ~/cuopt
```

### 步驟 2：第一次 Build（約 40 分鐘）

```bash
docker run --gpus all --rm \
    -v $(pwd):/cuopt \
    -v cuopt-ccache:/root/.cache/ccache \
    workcc/cuopt-allinone:26.6 \
    cuopt-build
```

> **第一次 build 需要網路**（下載 CCCL ~200MB）。
> 完成後 `cpp/build/_deps/` 保存在本機，之後離線也能 build。

### 步驟 3：改 code → Build → 驗證

```bash
# 改 code
vim ~/cuopt/cpp/src/routing/local_search/compute_insertions.cu

# Build（只重編有改的檔案）
docker run --gpus all --rm \
    -v $(pwd):/cuopt \
    -v cuopt-ccache:/root/.cache/ccache \
    workcc/cuopt-allinone:26.6 \
    cuopt-build

# 驗證
docker run --gpus all --rm -it \
    -v $(pwd):/cuopt \
    workcc/cuopt-allinone:26.6
# 容器內：
# python3 -c "import libcuopt; print(libcuopt.__version__)"
# ./cpp/build/cuopt_cli datasets/linear_programming/good-mps-some-var-bounds.mps
```

---

## 常用指令速查

```bash
# Build
docker run --gpus all --rm \
    -v $(pwd):/cuopt -v cuopt-ccache:/root/.cache/ccache \
    workcc/cuopt-allinone:26.6 cuopt-build

# 互動進入容器
docker run --gpus all --rm -it \
    -v $(pwd):/cuopt -v cuopt-ccache:/root/.cache/ccache \
    workcc/cuopt-allinone:26.6

# 跑 Python 腳本
docker run --gpus all --rm \
    -v $(pwd):/cuopt \
    workcc/cuopt-allinone:26.6 \
    python3 my_script.py

# 清除 ccache
docker volume rm cuopt-ccache

# 清除 cmake cache（換 GPU 環境後需要）
docker run --rm -v $(pwd):/cuopt workcc/cuopt-allinone:26.6 \
    bash -c "rm -rf /cuopt/cpp/build/CMakeCache.txt /cuopt/cpp/build/CMakeFiles"
```

---

## Image 內容

### cuopt-allinone（完整開發環境）
```
FROM nvidia/cuopt:26.6.0a-cuda12.9-py3.14（官方）
  Python 3.14 + RAPIDS + cuOpt Python packages 26.06.00a150
加裝：
  GCC 12.3 + nvcc 12.9 + cmake 4.3 + ninja + ccache
  cuda-nvcc-12-9 + cuda-cudart-dev-12-9 + cuda-profiler-api-12-9
  pytest 9.0+
  papilo + PSLP v0.0.8 + argparse v3.2 + rapids-cmake（離線 cmake 依賴）
  CUDA library/header symlinks + CUDSS symlinks
  LD_LIBRARY_PATH 預設包含所有 nvidia pip package lib 路徑
```

### cuopt-full-h200（H200 run-only）
```
libcuopt.so cross-compiled for H200（sm_90a）
含 scheduling branch 功能（set_order_weights 等）
無 build 工具
```

### cuopt-full-h200-sm90a（H200 可 build）
```
同上，額外包含 build 環境 + rapids-cmake
可在 H200 上執行 cuopt-build 重新編譯
```

---

## 已知 GCC 12 相容性修復（已寫入 source）

| 問題 | 修復位置 |
|------|----------|
| papilo `-Werror=nonnull` false positive | `cpp/CMakeLists.txt` |
| papilo fmt `-Werror=stringop-overflow=` | `cpp/CMakeLists.txt` |
| `__builtin_cpu_is("graniterapids*")` GCC 12 不支援 | `cpp/src/utilities/version_info.cpp` |
| `__builtin_cpu_is("sierraforest/grandridge")` 同上 | `cpp/src/utilities/version_info.cpp` |

---

## Repositories

- Source code：https://github.com/ccworkmail1212/cuopt
- Docker Hub：https://hub.docker.com/u/workcc
- Docker 腳本 + reassemble 指南：https://github.com/ccworkmail1212/cuopt-docker
