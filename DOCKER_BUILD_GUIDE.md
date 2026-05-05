# cuOpt Docker 開發指南

## 總覽

`workcc/cuopt-allinone:26.6` 是一個完整的 cuOpt 開發 image，包含：
- **改 code**：掛入 source code 用任何編輯器修改
- **Build code**：用 `cuopt-build` 重編 C++ library
- **Run code**：用官方 Python 環境執行 solver / 測試

```
你的電腦
  ├── ~/cuopt/          ← source code（掛入容器）
  └── docker run ...    ← 啟動容器執行，完成後自動關閉
```

---

## 需要的 Docker Images

| Image | 用途 | 大小 |
|---|---|---|
| `workcc/cuopt-allinone:26.6` | **完整開發環境**（改+build+run） | 8.59GB |

> `docker save workcc/cuopt-allinone:26.6` 會把 base image（`nvidia/cuopt`）的 layers 一起打包，**只需要這一個 image** 就能帶進公司離線使用。

---

## 完整流程（有網路環境先做）

### 步驟 0：確認環境

```bash
docker info      # Docker daemon 正常
nvidia-smi       # GPU 正常
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

**第一次 build 的三件事：**
1. cmake 下載 CCCL 依賴（~200MB）→ 存到 `cpp/build/_deps/cccl-src/`
2. 完整編譯 175 個 CUDA 檔案（ccache 加速後續 build）
3. `libcuopt.so` 自動複製進 Python package，Python 可直接用

> **重要**：`cpp/build/_deps/` 目錄會存在你的電腦上（透過 mount），帶進公司後就不需要網路了。

---

## 每次改 code 的流程

### 1. 改 code

```bash
# 在你的電腦上直接改，不需要進容器
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
[1/4] Building CUDA object ...compute_insertions.cu.o   ← 只編有改的
[2/4] Linking CXX shared library libcuopt.so
=== C++ build complete ===
```

### 3. 執行驗證

```bash
cd ~/cuopt

# 互動進入容器
docker run --gpus all --rm -it \
    -v $(pwd):/cuopt \
    workcc/cuopt-allinone:26.6

# 容器內——跑 LP solver
./cpp/build/cuopt_cli datasets/linear_programming/good-mps-some-var-bounds.mps

# 容器內——Python 驗證
python3 -c "import libcuopt; print(libcuopt.__version__)"
python3 your_script.py

# 容器內——確認改動真的在 binary 裡（改動前後比較）
strings cpp/build/libcuopt.so | grep "你加的字串"
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

# 跑 Python 腳本（不進容器）
docker run --gpus all --rm \
    -v $(pwd):/cuopt \
    workcc/cuopt-allinone:26.6 \
    python3 my_script.py

# 跑 LP solver CLI
docker run --gpus all --rm \
    -v $(pwd):/cuopt \
    workcc/cuopt-allinone:26.6 \
    ./cpp/build/cuopt_cli datasets/linear_programming/good-mps-some-var-bounds.mps

# 清除 ccache（build 出錯想從頭）
docker volume rm cuopt-ccache

# 清除 cmake cache（換環境後需要）
docker run --rm -v $(pwd):/cuopt workcc/cuopt-allinone:26.6 \
    bash -c "rm -rf /cuopt/cpp/build/CMakeCache.txt /cuopt/cpp/build/CMakeFiles"
```

---

## 離線環境（公司內網）

### 需要帶進公司的東西

| 項目 | 位置 | 說明 |
|------|------|------|
| `workcc/cuopt-allinone:26.6` | Docker Hub → 匯出為 tar | 開發環境 image |
| `nvidia/cuopt:26.6.0a-cuda12.9-py3.14` | Docker Hub → 匯出為 tar | allinone 的 base |
| Source code | `~/cuopt/` 目錄 | C++ 原始碼 |
| `cpp/build/_deps/` | `~/cuopt/cpp/build/_deps/` | CCCL 等 cmake 依賴（第一次 build 後產生） |
| ccache volume | `docker volume export cuopt-ccache` | 編譯快取（可選但有大幅加速） |

### 匯出 / 匯入 image

```bash
# 匯出（有網路的電腦）——只需要一個 tar，base image 的 layers 已包含在內
docker save workcc/cuopt-allinone:26.6 | gzip > cuopt-allinone.tar.gz

# 匯入（公司電腦）——完全自給自足，不需要另外 load nvidia/cuopt
docker load < cuopt-allinone.tar.gz
```

> `docker save` 會把所有 layers（包含 `nvidia/cuopt` 的 base layers）一起打包，所以只需要一個 tar 檔案。

### 公司內 Build

帶進公司後，`cpp/build/_deps/` 已存在，cmake 不需要下載，直接 build：

```bash
cd ~/cuopt
docker run --gpus all --rm \
    -v $(pwd):/cuopt \
    -v cuopt-ccache:/root/.cache/ccache \
    workcc/cuopt-allinone:26.6 \
    cuopt-build
# 完全離線，不需要網路
```

---

## Image 內容（完全自給自足）

```
workcc/cuopt-allinone:26.6
  FROM nvidia/cuopt:26.6.0a-cuda12.9-py3.14（官方）
    Python 3.14 + RAPIDS + cuOpt Python packages 26.06.00a150
  加裝：
    GCC 12.3 + nvcc 12.9 + cmake 4.3 + ninja + ccache
    cuda-nvcc-12-9 + cuda-cudart-dev-12-9 + cuda-profiler-api-12-9
    pytest 9.0+（測試框架，已內建）
    papilo + PSLP v0.0.8 + argparse v3.2（cmake 離線依賴）
    CUDA library symlinks（cublas/cusparse/curand/cusolver）
    CUDSS symlinks（/opt/cudss）
    LD_LIBRARY_PATH 預設包含所有 nvidia pip package lib 路徑
```

進到容器後**不需要任何 `pip install`**，也**不需要設定環境變數**。

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
