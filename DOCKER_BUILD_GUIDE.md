# cuOpt Docker 開發指南

## 總覽

`workcc/cuopt-allinone:26.6` 是一個完整的 cuOpt 開發 image，包含：
- **改 code**：掛入 source code 用任何編輯器修改
- **Build code**：用 `cuopt-build` 重編 C++ library
- **Run code**：用官方 Python 環境跑測試

```
你的電腦
  ├── ~/cuopt/          ← source code（掛入容器）
  └── docker run ...    ← 啟動容器執行，完成後自動關閉
```

---

## 需要的 Docker Images

| Image | 用途 | 大小 | 類型 |
|---|---|---|---|
| `workcc/cuopt-allinone:26.6` | **完整開發環境**（改+build+run） | 8.58GB | 自訂（基於官方） |
| `nvidia/cuopt:26.6.0a-cuda12.9-py3.14` | allinone 的 base（自動從 Hub 拉取） | 7.74GB | 官方 NVIDIA |

> `workcc/cuopt-allinone:26.6` 基於 `nvidia/cuopt` 官方 image。

---

## 快速開始

### 步驟 0：確認 Docker + GPU 正常

```bash
docker info
nvidia-smi
```

### 步驟 1：取得 source code（只做一次）

```bash
mkdir -p ~/cuopt && cd ~

# 方法 A：從 Git clone（有網路）
git clone https://github.com/ccworkmail1212/cuopt.git ~/cuopt

# 方法 B：從 cuopt-src image 解壓（離線）
docker run --rm workcc/cuopt-src:26.6 tar cf - /cuopt | tar xf - -C ~/
```

### 步驟 2：第一次 Build（約 40 分鐘）

```bash
cd ~/cuopt

docker run --gpus all --rm \
    -v $(pwd):/cuopt \
    -v cuopt-ccache:/root/.cache/ccache \
    workcc/cuopt-allinone:26.6 \
    cuopt-build
```

第一次會下載 CCCL 依賴（~200MB）並完整編譯 175 個 CUDA 檔案。之後改幾個檔案只需幾秒到幾分鐘。

---

## 每次改 code 的完整流程

### 1. 改 code（在你的電腦上直接改）

```bash
# 用任何編輯器修改 C++ 原始碼
vim ~/cuopt/cpp/src/routing/local_search/compute_insertions.cu
```

### 2. Build（只重編有改的檔案）

```bash
cd ~/cuopt
docker run --gpus all --rm \
    -v $(pwd):/cuopt \
    -v cuopt-ccache:/root/.cache/ccache \
    workcc/cuopt-allinone:26.6 \
    cuopt-build
```

成功輸出：
```
[1/4] Building CXX object ...version_info.cpp.o    ← 只編有改的
[2/4] Linking CXX shared library libcuopt.so
=== C++ build complete ===
  libcuopt.so  : cpp/build/libcuopt.so
```

新的 `libcuopt.so` 自動複製進 Python package 路徑，下一步的 Python 測試會直接用新 binary。

### 3. 驗證（跑 Python 或測試）

```bash
cd ~/cuopt

# 互動進入容器（pytest、cuopt_cli 都已內建，不需要額外安裝）
docker run --gpus all --rm -it \
    -v $(pwd):/cuopt \
    workcc/cuopt-allinone:26.6

# 容器內執行：
python3 -c "import libcuopt; print(libcuopt.__version__)"
python3 -m pytest python/cuopt_self_hosted/tests/ -v
python3 your_script.py

# 直接跑 CLI solver（不需設定 LD_LIBRARY_PATH）
./cpp/build/cuopt_cli datasets/linear_programming/good-mps-some-var-bounds.mps
```

### 4. 確認改動確實在 binary 裡

```bash
# 方法：在 C++ 加一個獨特字串，build 後用 strings 確認
strings cpp/build/libcuopt.so | grep "你的字串"
```

---

## 常用指令速查

```bash
# Build
docker run --gpus all --rm \
    -v $(pwd):/cuopt -v cuopt-ccache:/root/.cache/ccache \
    workcc/cuopt-allinone:26.6 cuopt-build

# 互動模式（進入容器）
docker run --gpus all --rm -it \
    -v $(pwd):/cuopt -v cuopt-ccache:/root/.cache/ccache \
    workcc/cuopt-allinone:26.6

# 跑 Python 腳本
docker run --gpus all --rm \
    -v $(pwd):/cuopt \
    workcc/cuopt-allinone:26.6 \
    python3 my_script.py

# 清除 ccache（build 出錯想從頭）
docker volume rm cuopt-ccache
```

---

## 離線環境（公司內網）

所有操作**不需要網路**（第一次 build 除外）：
- Build 工具和 Python 環境在 image 裡
- papilo/pslp/argparse 已預先 clone 在 image 中
- CCCL 第一次 build 後存在 `cpp/build/_deps/`（掛入 volume 保留）
- ccache 存在 Docker volume `cuopt-ccache`，重開機後仍有效

**第一次 build 需要網路**（下載 CCCL）。建議在有網路的環境先 build 一次，讓 `cpp/build/_deps/` 存在，再帶進公司。

---

## Image 內容

```
workcc/cuopt-allinone:26.6
  FROM nvidia/cuopt:26.6.0a-cuda12.9-py3.14（官方）
    Python 3.14 + RAPIDS + cuOpt Python packages 26.06.00a150
  加裝：
    GCC 12.3 + nvcc 12.9 + cmake 4.3 + ninja + ccache
    cuda-nvcc-12-9 + cuda-cudart-dev-12-9 + cuda-profiler-api-12-9
    pytest 9.0+ + pytest-timeout（測試框架，已內建）
    離線依賴：papilo + PSLP v0.0.8 + argparse v3.2
    CUDA symlinks：cublas/cusparse/curand/cusolver headers + libs
    CUDSS symlinks：/opt/cudss → nvidia cu12 package
    LD_LIBRARY_PATH：預設包含所有 nvidia pip package lib 路徑
```

> **完全自給自足**：image 帶進公司後不需要 `pip install` 任何套件，也不需要手動設定環境變數。

---

## 已知 GCC 12 相容性修復（已寫入 source）

| 問題 | 修復位置 |
|------|----------|
| papilo `-Werror=nonnull` false positive | `cpp/CMakeLists.txt:78` |
| papilo fmt `-Werror=stringop-overflow=` | `cpp/CMakeLists.txt:78` |
| `__builtin_cpu_is("graniterapids*")` GCC 12 不支援 | `cpp/src/utilities/version_info.cpp` |
| `__builtin_cpu_is("sierraforest/grandridge")` 同上 | `cpp/src/utilities/version_info.cpp` |

---

## Repositories

- Source code：https://github.com/ccworkmail1212/cuopt
- Docker Hub：https://hub.docker.com/u/workcc
