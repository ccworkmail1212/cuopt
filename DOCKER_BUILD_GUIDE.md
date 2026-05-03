# cuOpt Docker 開發指南

## 需要的三個 Docker Image

| Image | 用途 | 大小 |
|---|---|---|
| `workcc/cuopt-dev:26.6.0a` | Build 環境（Ubuntu 22.04 + GCC 12 + nvcc） | ~11.7GB |
| `nvidia/cuopt:26.6.0a-cuda12.9-py3.14` | Python 測試環境（官方 image） | ~7.7GB |
| `workcc/cuopt-src:26.6` | Source code 封裝 | ~336MB |

---

## 快速上手

### 步驟 1：準備 Images

```bash
docker pull workcc/cuopt-dev:26.6.0a
docker pull nvidia/cuopt:26.6.0a-cuda12.9-py3.14
docker pull workcc/cuopt-src:26.6
```

### 步驟 2：取得 Source Code

```bash
# 方法 A：從 cuopt-src image 取出（無網路環境用）
docker run --rm workcc/cuopt-src:26.6 \
    tar cf - /cuopt | tar xf - -C ~/
cd ~/cuopt

# 方法 B：git clone（需要網路）
git clone https://github.com/ccworkmail1212/cuopt.git
cd cuopt
```

### 步驟 3：Build C++ Library

```bash
bash run_dev.sh cuopt-build
# 第一次：~40 分鐘（ccache 暖機）
# 之後增量 build：秒級（只重編有改動的檔案）
# 輸出：cpp/build/libcuopt.so（~93MB）
#        cpp/build/cuopt_cli
```

### 步驟 4：驗證 Build 成功

```bash
# 查看 artifacts
ls -lh cpp/build/libcuopt.so cpp/build/cuopt_cli

# C++ 測試（需 GPU）
bash run_dev.sh cuopt-build --with-tests
bash run_dev.sh ctest --test-dir cpp/build -j4
```

### 步驟 5：Python 測試

```bash
bash run_test_py.sh pytest python/cuopt/cuopt/tests/

# 或互動進入 Python 環境
bash run_test_py.sh
```

---

## 日常開發流程（改 code → 測試）

```bash
# 1. 直接編輯 host 上的 C++ 原始碼
vim cpp/src/routing/local_search/compute_insertions.cu

# 2. Build（只重編有改動的檔案，通常 < 1 分鐘）
bash run_dev.sh cuopt-build

# 3. 測試
bash run_test_py.sh pytest python/cuopt/cuopt/tests/lp/
```

---

## `run_dev.sh` 常用指令

```bash
bash run_dev.sh                            # 互動 bash（進入 dev container）
bash run_dev.sh cuopt-build                # 一般 build（跳過 test binary）
bash run_dev.sh cuopt-build --with-tests   # Build + 啟用 ctest
bash run_dev.sh ctest --test-dir cpp/build -j4  # 執行 C++ 測試
```

## `run_test_py.sh` 常用指令

```bash
bash run_test_py.sh                                          # 互動 bash
bash run_test_py.sh pytest python/cuopt/cuopt/tests/         # 全部 Python 測試
bash run_test_py.sh pytest python/cuopt/cuopt/tests/lp/ -v   # 只跑 LP 測試
bash run_test_py.sh python3 my_script.py                     # 自訂腳本
```

---

## 公司（離線）環境

### 需攜帶的三個 Images

```bash
# 事先在有網路的機器上 pull，然後 save 或透過公司 registry 傳輸
docker pull workcc/cuopt-dev:26.6.0a
docker pull nvidia/cuopt:26.6.0a-cuda12.9-py3.14
docker pull workcc/cuopt-src:26.6
```

### 完整離線流程

```bash
# 取出 source code（不需要 git clone）
docker run --rm workcc/cuopt-src:26.6 \
    tar cf - /cuopt | tar xf - -C ~/
cd ~/cuopt

# 改 code
vim cpp/src/...

# Build（在 workcc/cuopt-dev 容器裡，不需要網路）
bash run_dev.sh cuopt-build

# Python 測試（在 nvidia/cuopt 容器裡，不需要網路）
bash run_test_py.sh pytest python/cuopt/cuopt/tests/
```

---

## 架構說明

```
workcc/cuopt-dev:26.6.0a          ← Build 容器
  ├── Ubuntu 22.04 + GCC 12
  ├── CUDA 12.9 (nvcc)
  ├── cmake 3.30+, ninja, ccache
  ├── RAPIDS cmake configs (from 官方 image)
  └── 離線 deps: papilo, pslp, argparse

nvidia/cuopt:26.6.0a-cuda12.9-py3.14  ← Python 測試容器
  ├── 完整 Python 環境（pylibraft, rmm, cupy...）
  └── 官方 pre-built libcuopt.so（被 custom 版本替換）

workcc/cuopt-src:26.6             ← Source code 封裝
  └── /cuopt/ 目錄（僅保留編譯所需部分）
```

---

## Repositories

- Source code：https://github.com/ccworkmail1212/cuopt
- Docker 腳本：https://github.com/ccworkmail1212/cuopt-docker
- Docker Hub：https://hub.docker.com/u/workcc
