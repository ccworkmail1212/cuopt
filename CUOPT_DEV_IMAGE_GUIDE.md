# cuOpt Dev Image 使用指南

**Image**: `workcc/cuopt-full:26.6-sm90a-dev`  
**大小**: ~12.4 GB  
**GPU 架構**: SM90a（H200 / H100）  
**基底**: cuOpt 26.06 scheduling branch，含完整 source code

---

## 包含工具

### cuOpt 核心
| 項目 | 說明 |
|------|------|
| cuOpt source code | `/cuopt/`（可直接修改） |
| `libcuopt.so` | 已編譯的 SM90a binary |
| `cuopt_cli` | C++ 命令列工具 |
| Python bindings | `from cuopt import routing` |

### 開發工具
| 工具 | 版本 | 用途 |
|------|------|------|
| Claude Code | 2.1.131 | AI coding assistant |
| Claude Code Router | latest | 公司內網 proxy（`claude-code-router`） |
| code-server | 4.117.0 | VSCode（瀏覽器版） |
| JupyterLab | 4.5.7 | 互動式 notebook |
| Python | 3.14.4 | |
| Node.js | 22.22.2 | |
| CUDA | 12.9 | |
| nvcc | 12.9.86 | CUDA compiler |
| cuda-gdb | 12.9 | CUDA debugger |
| gdb | — | C++ debugger |
| openssh-server | 8.9p1 | SSH 連線（VS Code Remote SSH） |
| python3-venv | — | Python 虛擬環境 |
| gcc-12 / g++-12 | 12 | C++ compiler |
| cmake | 4.3 | Build system |

### CLI 工具
| 工具 | 用途 |
|------|------|
| `bat` | 語法高亮的 cat |
| `fzf` | 模糊搜尋 |
| `ripgrep (rg)` | 快速 code 搜尋 |
| `nvtop` | GPU 使用率監控 |
| `strace` | syscall 追蹤 |
| `screen` | 多視窗終端 |
| `ipython3` | 互動式 Python |
| `black` | Python formatter |
| `tree` | 目錄結構顯示 |
| `clang-format` | C++ formatter |

### 內建 Alias
```bash
cuopt-build     # 增量 build（只重編有改動的檔案）
cuopt-rebuild   # 完整重新 build（清掉 cmake cache）
ll              # ls -lah
```

---

## 啟動方式

### 基本啟動（一行）
```bash
docker run --gpus all -it \
    -p 8080:8080 \
    -p 8888:8888 \
    -e ANTHROPIC_API_KEY=你的key \
    --name cuopt-dev \
    workcc/cuopt-full:26.6-sm90a-dev bash
```

### 一次啟動所有服務
```bash
docker run --gpus all -it -p 8080:8080 -p 8888:8888 -e ANTHROPIC_API_KEY=你的key --name cuopt-dev workcc/cuopt-full:26.6-sm90a-dev bash -c "code-server --bind-addr 0.0.0.0:8080 --auth none /cuopt & jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' & exec bash"
```

### 加入 Claude Code Router（公司內網）
```bash
docker run --gpus all -it -p 8080:8080 -p 8888:8888 \
    -e ANTHROPIC_API_KEY=你的key \
    -e ANTHROPIC_BASE_URL=http://router-ip:port \
    --name cuopt-dev \
    workcc/cuopt-full:26.6-sm90a-dev bash
```

---

## 各工具啟動方式

### code-server（VSCode）
```bash
# 在容器內執行
code-server --bind-addr 0.0.0.0:8080 --auth none /cuopt
```

### JupyterLab
```bash
# 在容器內執行
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=''
```

### Claude Code
```bash
# 直接連外網（有網路時）
cd /cuopt
claude

# 透過公司內網 Router
ANTHROPIC_BASE_URL=http://router-ip:port claude
```

### Claude Code Router（公司內網）
```bash
# 啟動 router（需先設定 config）
claude-code-router

# config 路徑
~/.config/claude-code-router/config.json
```

---

## 連線網址

| 環境 | VSCode | JupyterLab |
|------|--------|-----------|
| Linux 本機 | `http://localhost:8080` | `http://localhost:8888` |
| WSL2 內的瀏覽器 | `http://localhost:8080` | `http://localhost:8888` |
| WSL2 外的 Windows | `http://WSL2-IP:8080` | `http://WSL2-IP:8888` |
| 遠端 SSH | SSH tunnel 後用 `localhost` | SSH tunnel 後用 `localhost` |
| VS Code Remote SSH | 直接連容器 port 22 | — |

**查詢 WSL2 IP：**
```bash
hostname -I | awk '{print $1}'
```

**SSH Tunnel（遠端連線）：**
```bash
ssh -L 8080:localhost:8080 -L 8888:localhost:8888 user@server-ip
# 然後瀏覽器開 http://localhost:8080
```

**VS Code Remote SSH 直連容器：**
```bash
# 啟動容器時開 port 22
docker run --gpus all -it -p 22:22 -p 8080:8080 -p 8888:8888 ... workcc/cuopt-full:26.6-sm90a-dev bash

# 容器內啟動 SSH daemon
/usr/sbin/sshd

# VS Code：Remote SSH → ssh root@localhost（密碼：cuopt）
```

---

## cuOpt Build

### 增量 build（修改 code 後）
```bash
cuopt-build
# 等同於：
cd /cuopt && RAPIDS_DIST=/usr/local/lib/python3.14/dist-packages PARALLEL_LEVEL=4 bash ci/docker/build_cuopt.sh
```

### 完整重新 build（清掉 cmake cache）
```bash
cuopt-rebuild
# 等同於：
cd /cuopt && rm -f cpp/build/CMakeCache.txt && rm -rf cpp/build/CMakeFiles cpp/build/*.so cpp/build/cuopt* && RAPIDS_DIST=/usr/local/lib/python3.14/dist-packages PARALLEL_LEVEL=4 bash ci/docker/build_cuopt.sh
```

> **注意**：重新 build 會保留 `_deps/`（rapids-cmake、papilo、pslp 等），不需要網路即可完成。

### 確認 build 成功
```bash
ls -lh /cuopt/cpp/build/libcuopt.so
strings /cuopt/cpp/build/libcuopt.so | grep 'sm_90' | head -3
```

---

## 常用開發流程

```bash
# 1. 啟動容器
docker run --gpus all -it -p 8080:8080 -p 8888:8888 -e ANTHROPIC_API_KEY=key --name cuopt-dev workcc/cuopt-full:26.6-sm90a-dev bash

# 2. 啟動服務（容器內）
code-server --bind-addr 0.0.0.0:8080 --auth none /cuopt &
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' &

# 3. 用瀏覽器開 VSCode 修改 code
# http://localhost:8080

# 4. 重新 build
cuopt-rebuild   # 完整 build
cuopt-build     # 增量 build

# 5. 執行測試
python3 /demo/lot_scheduling_demo.py

# 6. 問 Claude Code
claude
```

---

## 停止與清理

```bash
# 停止容器
docker stop cuopt-dev

# 停止並刪除
docker rm -f cuopt-dev

# 停止所有 running 容器
docker stop $(docker ps -q)
```

---

## 離線環境說明

此 image 設計為**完全離線可用**：

- rapids-cmake、papilo、pslp 依賴均已預載於 `_deps/`
- cmake reconfigure 時自動使用本地目錄，不下載
- 唯一需要網路的是 Claude Code（需連 Anthropic API 或公司 router）
