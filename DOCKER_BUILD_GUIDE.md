# cuOpt Docker 開發指南

## 概念說明

你只需要在**自己的電腦（host）**上操作。`bash run_dev.sh` 會自動啟動和關閉 Docker 容器，你不需要手動管理容器。

```
你的電腦（host）
  ├── ~/cuopt/            ← source code 在這裡，直接用編輯器改
  └── bash run_dev.sh ... ← 自動啟動容器執行，完成後自動關閉
```

---

## 需要的 Docker Images（帶進公司）

| Image | 用途 | 類型 | 大小 |
|---|---|---|---|
| `nvidia/cuopt:26.6.0a-cuda12.9-py3.14` | cuopt-dev 的基底（Ubuntu 22.04 + Python + RAPIDS） | **官方 NVIDIA** | 7.7GB |
| `workcc/cuopt-src:26.6` | 裝著 source code 的「快遞盒」 | 自訂 | 336MB |
| `workcc/cuopt-dev:26.6.0a` | 開發容器（GCC 12 + cmake + build 工具） | 自訂 delta | ~500MB |

> `workcc/cuopt-dev` 以官方 `nvidia/cuopt` 為基底，自訂部分只有 ~500MB，符合公司 < 2GB 限制。

---

## 完整操作步驟

### 步驟 0：確認 Docker 有在執行

```bash
docker info
# 看到 "Server Version: ..." 代表正常
```

### 步驟 1：取出 source code（只做一次）

```bash
mkdir -p ~/cuopt && cd ~

# 從 cuopt-src image 解壓 source code 到 ~/cuopt/
docker run --rm workcc/cuopt-src:26.6 \
    tar cf - /cuopt | tar xf - -C ~/

cd ~/cuopt
ls cpp/src/    # 應該看到 routing、math_optimization 等目錄
```

> 之後就不再需要 `workcc/cuopt-src`，它只是個「快遞盒」。

---

### 每次改 code 的流程

#### 第 1 步：改 code（在你的電腦上直接改，不需要進容器）

```bash
# 用任何編輯器修改 C++ 原始碼
vim ~/cuopt/cpp/src/routing/local_search/compute_insertions.cu

# 或用 VSCode、gedit 等，只要改 ~/cuopt/cpp/src/ 下的檔案
```

#### 第 2 步：Build（自動啟動容器編譯，完成後容器自動關閉）

```bash
cd ~/cuopt
bash run_dev.sh cuopt-build

# 第一次：約 40 分鐘（編譯 141 個 CUDA 檔案）
# 之後改幾個檔案：幾秒到幾分鐘（ccache 快取加速）
# 成功時最後顯示：=== C++ build complete ===
```

#### 第 3 步：驗證（自動啟動容器跑測試，完成後容器自動關閉）

```bash
# Python 測試
bash run_dev.sh pytest python/cuopt/cuopt/tests/ -v

# 只跑特定測試
bash run_dev.sh pytest python/cuopt/cuopt/tests/lp/ -v

# C++ 測試（需要 GPU）
bash run_dev.sh cuopt-build --with-tests
bash run_dev.sh ctest --test-dir cpp/build -j4
```

#### 互動模式（進入容器自己操作）

```bash
bash run_dev.sh
# 現在你在容器內，可以：
python3 -c "import libcuopt; print(libcuopt.__version__)"
python3 my_script.py
exit   # 離開容器，回到你的電腦
```

---

### 完整範例：改 routing 邏輯並驗證

```bash
cd ~/cuopt

# 1. 改 code
vim cpp/src/routing/local_search/compute_insertions.cu

# 2. Build（只重編有改的檔案，通常 < 2 分鐘）
bash run_dev.sh cuopt-build

# 3. 跑 routing 測試確認改動有效
bash run_dev.sh pytest python/cuopt/cuopt/tests/routing/ -v
```

---

## 常用指令速查

```bash
# Build
bash run_dev.sh cuopt-build

# Python 測試（全部）
bash run_dev.sh pytest python/cuopt/cuopt/tests/

# Python 測試（指定目錄）
bash run_dev.sh pytest python/cuopt/cuopt/tests/lp/ -v

# C++ 測試
bash run_dev.sh ctest --test-dir cpp/build -j4

# 互動進入容器
bash run_dev.sh

# Build + C++ 測試（一起）
bash run_dev.sh cuopt-build --with-tests
```

---

## 三個 Image 的分工

```
workcc/cuopt-src:26.6
  → 只用一次：解壓 source code 到 ~/cuopt/
  → 之後不需要

workcc/cuopt-dev:26.6.0a
  → 所有開發工作（build + test）
  → 基底：nvidia/cuopt（官方，Ubuntu 22.04 + Python 3.14 + RAPIDS）
  → 自訂：GCC 12、cmake、papilo/pslp/argparse 等 build 工具

nvidia/cuopt:26.6.0a-cuda12.9-py3.14
  → cuopt-dev 的基底（你不會直接用它）
  → 官方 NVIDIA，公司已核准
```

---

## 離線環境注意事項

所有操作**不需要網路**：
- Build 工具和依賴都在 `workcc/cuopt-dev` image 裡
- papilo、pslp、argparse 已預先 clone 在 image 中
- ccache 快取存在 Docker volume `cuopt-ccache`，重開機後仍有效

---

## Repositories

- Source code：https://github.com/ccworkmail1212/cuopt
- Docker 腳本：https://github.com/ccworkmail1212/cuopt-docker
- Docker Hub：https://hub.docker.com/u/workcc
