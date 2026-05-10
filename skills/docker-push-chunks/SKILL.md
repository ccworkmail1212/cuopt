---
name: docker-push-chunks
version: "1.0.0"
description: Split a large Docker image into small chunk images and push each chunk to Docker Hub. Use when a Docker image exceeds upload bandwidth limits or Docker Hub layer size limits (~1GB), or when you need to transport a large image through a constrained channel (e.g., a company with per-image size limits).
origin: skill-evolution
---

# Docker Image Chunked Push Skill

**Context (cuOpt project):** The company only allows NVIDIA official images at any size; all custom images must be < 2GB. cuOpt images are 9–11GB, so they cannot be pushed as-is. Solution: split into chunks (each < 1.5GB), wrap each in an alpine image, push separately, reassemble on target with `docker load`.

Split a large Docker image into multiple small chunk images, push each chunk to Docker Hub, then reassemble on the target machine.

**When to use:**
- Company policy limits each custom image to < 2GB (primary reason for cuOpt)
- Local upload bandwidth is too low to push a large image directly (< 5 Mbps upload)
- Docker Hub rejects large layers (`unexpected EOF` during push)
- Need to transport a large image to an air-gapped or offline network

---

## Core Concept

```
docker save IMAGE | gzip > image.tar.gz   # ~50% compression
split -b 1400m image.tar.gz part_          # chunks ≤ 1.4GB
→ wrap each chunk in an alpine image       # CMD ["cat", "/chunk"]
→ push each chunk image to Docker Hub
→ target: docker run image > part_XX for each, then cat parts | docker load
```

Each chunk image uses `CMD ["cat", "/chunk"]` so `docker run --rm IMAGE > file` extracts the chunk without entering the container.

---

## Naming Convention

Use descriptive chunk repo names that encode purpose and GPU target:

| Image purpose | Chunk prefix |
|---|---|
| H200 run-only (pre-compiled sm_90a) | `myimage-h200` |
| H200 + build tools | `myimage-h200-sm90a` |
| General dev / any GPU | `myimage-allinone` |
| Specific release | `myimage-v1-2` |

**Avoid generic names** like `myimage-part` — if you push multiple versions, the chunks collide and overwrite each other.

---

## Chunk Size Selection

| Upload speed | Max chunk size | Reason |
|---|---|---|
| < 5 Mbps | 500MB | Docker Hub drops connections on large layers |
| 5–50 Mbps | 1.4GB | Safe margin under 1.5GB per-image policy |
| > 50 Mbps | 1.9GB | Faster reassembly, fewer chunks |

**Do not use 1.9GB chunks if upload is slow** — Docker Hub returns `unexpected EOF` for large layers on slow connections regardless of chunk size.

---

## Method 1: GitHub Actions (recommended when local bandwidth is low)

Create `.github/workflows/push-chunks.yml`:

```yaml
name: Push Image Chunks to Docker Hub

on:
  workflow_dispatch:
    inputs:
      image:
        description: 'Source image (e.g. myorg/myimage:1.0)'
        required: true
      chunk_name:
        description: 'Chunk repo prefix (e.g. myimage-h200)'
        required: true
      tag:
        description: 'Tag'
        required: true
        default: '1.0'
      chunk_size_mb:
        description: 'Chunk size in MB'
        required: true
        default: '1400'

jobs:
  split-and-push:
    runs-on: ubuntu-latest
    steps:
      - name: Free disk space
        run: |
          sudo rm -rf /usr/share/dotnet /usr/local/lib/android /opt/ghc || true

      - name: Login to Docker Hub
        uses: docker/login-action@v3
        with:
          username: ${{ vars.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}

      - name: Pull source image
        run: docker pull ${{ inputs.image }}

      - name: Save, compress, split
        run: |
          mkdir -p /tmp/chunks
          docker save ${{ inputs.image }} | gzip > /tmp/chunks/image.tar.gz
          split -b ${{ inputs.chunk_size_mb }}m /tmp/chunks/image.tar.gz /tmp/chunks/part_
          echo "Total chunks: $(ls /tmp/chunks/part_* | wc -l)"
          ls -lh /tmp/chunks/part_*

      - name: Build and push each chunk
        run: |
          for part in $(ls /tmp/chunks/part_* | sort); do
            suffix=$(basename $part | sed 's/part_//')
            tag="${{ vars.DOCKERHUB_USERNAME }}/${{ inputs.chunk_name }}-part-${suffix}:${{ inputs.tag }}"
            mkdir -p /tmp/build_${suffix}
            cp $part /tmp/build_${suffix}/chunk
            cat > /tmp/build_${suffix}/Dockerfile << 'DOCKERFILE'
          FROM alpine
          COPY chunk /chunk
          CMD ["cat", "/chunk"]
          DOCKERFILE
            docker build -t ${tag} /tmp/build_${suffix}/
            docker push ${tag}
            echo "✅ ${tag} pushed"
          done

      - name: Summary
        run: |
          echo "## Pushed chunks" >> $GITHUB_STEP_SUMMARY
          echo '```bash' >> $GITHUB_STEP_SUMMARY
          for part in $(ls /tmp/chunks/part_* | sort); do
            suffix=$(basename $part | sed 's/part_//')
            echo "docker pull ${{ vars.DOCKERHUB_USERNAME }}/${{ inputs.chunk_name }}-part-${suffix}:${{ inputs.tag }}" >> $GITHUB_STEP_SUMMARY
          done
          echo '```' >> $GITHUB_STEP_SUMMARY
```

Trigger:
```bash
gh workflow run push-chunks.yml \
    --repo myorg/myrepo \
    -f image="myorg/myimage:1.0" \
    -f chunk_name="myimage-h200" \
    -f tag="1.0" \
    -f chunk_size_mb="1400"
```

**GitHub Actions runners have ~1–5 Gbps bandwidth** — a 10GB image compresses and pushes in ~15–20 minutes.

---

## Method 2: Local push (only when upload ≥ 50 Mbps)

```bash
# 1. Save and compress
docker save myorg/myimage:1.0 | gzip > /tmp/image.tar.gz

# 2. Split
split -b 1400m /tmp/image.tar.gz /tmp/part_
ls -lh /tmp/part_*

# 3. Build chunk images
for part in /tmp/part_*; do
    suffix=$(basename $part | sed 's/part_//')
    mkdir -p /tmp/build_${suffix}
    cp $part /tmp/build_${suffix}/chunk
    printf "FROM alpine\nCOPY chunk /chunk\nCMD [\"cat\", \"/chunk\"]\n" \
        > /tmp/build_${suffix}/Dockerfile
    docker build -t myorg/myimage-h200-part-${suffix}:1.0 /tmp/build_${suffix}/
done

# 4. Push one by one (NOT parallel — parallel kills connections)
for suffix in aa ab ac ad; do
    docker push myorg/myimage-h200-part-${suffix}:1.0
    # verify
    docker manifest inspect myorg/myimage-h200-part-${suffix}:1.0 >/dev/null \
        && echo "✅ ${suffix}" || echo "❌ ${suffix} FAILED"
done
```

**Do NOT push in parallel (`&`)** — concurrent pushes share bandwidth and cause `unexpected EOF` on all of them.

---

## Reassembly Script (target machine)

Save as `reassemble.sh` and ship with the instructions:

```bash
#!/bin/bash
# Usage: bash reassemble.sh <chunk-prefix> <tag> <suffix1> <suffix2> ...
# Example: bash reassemble.sh myimage-h200 1.0 aa ab ac ad

set -e
NAME=$1; TAG=$2; shift 2; SUFFIXES=("$@")

echo "=== Pull chunks ==="
for s in "${SUFFIXES[@]}"; do
    docker pull "myorg/${NAME}-part-${s}:${TAG}"
done

echo "=== Extract chunks ==="
PARTS=()
for s in "${SUFFIXES[@]}"; do
    docker run --rm "myorg/${NAME}-part-${s}:${TAG}" > "part_${s}"
    PARTS+=("part_${s}")
done

echo "=== Load image ==="
cat "${PARTS[@]}" | docker load

echo "=== Cleanup ==="
rm -f "${PARTS[@]}"
docker images | grep "${NAME%-part*}" | grep -v "\-part\-"
```

---

## After Reassembly: Using the cuOpt Image

After `docker load` completes, the full image is available locally. Choose the right commands depending on which image you loaded:

### cuopt-allinone（任何 GPU，可 build + run）

```bash
# Build cuOpt from source（需先有 source code）
docker run --gpus all --rm \
    -v ~/cuopt:/cuopt \
    -v cuopt-ccache:/root/.cache/ccache \
    workcc/cuopt-allinone:26.6 \
    cuopt-build

# 互動模式（改 code、執行測試）
docker run --gpus all --rm -it \
    -v ~/cuopt:/cuopt \
    -v cuopt-ccache:/root/.cache/ccache \
    workcc/cuopt-allinone:26.6

# 跑 Python 腳本
docker run --gpus all --rm \
    -v ~/cuopt:/cuopt \
    workcc/cuopt-allinone:26.6 \
    python3 my_script.py
```

### cuopt-full-h200（H200 only，只能 run）

```bash
# 互動模式
docker run --gpus all --rm -it workcc/cuopt-full-h200:26.6 bash

# 執行 Python 程式
docker run --gpus all --rm \
    -v ~/scripts:/workspace \
    --workdir /workspace \
    workcc/cuopt-full-h200:26.6 \
    python3 my_script.py
```

### cuopt-full-h200-sm90a（H200 only，可 build）

```bash
# Build（在 H200 機器上）
docker run --gpus all --rm \
    -v ~/cuopt:/cuopt \
    -v cuopt-ccache:/root/.cache/ccache \
    workcc/cuopt-full-h200-sm90a:26.6 \
    cuopt-build
```

載入後 image 名稱就是**原始 image 名**（`workcc/cuopt-allinone:26.6` 等），chunk images（`workcc/cuopt-allinone-part-aa` 等）是搬運用的，載完可以刪掉節省磁碟空間：

```bash
for s in aa ab ac ad ae af ag ah ai; do
    docker rmi workcc/cuopt-allinone-part-${s}:26.6
done
```

---

## Gotchas

### `unexpected EOF` during push
- **Cause**: Docker Hub drops the TCP connection when uploading a layer > ~1GB on slow connections
- **Fix**: Reduce chunk size to 500MB, or switch to GitHub Actions

### Chunks push but Docker Hub web UI doesn't show them
- **Cause**: Docker Hub API (`/v2/repositories/`) lags behind; `docker manifest inspect` is authoritative
- **Fix**: Use `docker manifest inspect myorg/myimage-part-aa:1.0` to verify, not the web UI

### Push process shows 0% CPU for 10+ minutes
- **Cause**: Docker daemon is stuck in layer preparation (hash computation blocked on slow I/O or network)
- **Fix**: Kill the stuck process (`kill $(pgrep -f "docker push")`) and retry

### `docker load` produces wrong image after reassembly
- **Cause**: `cat` order was wrong — chunks must be concatenated in alphabetical order (aa, ab, ac…)
- **Fix**: Always use `sort` when iterating: `for part in $(ls part_* | sort)`

### Parallel pushes all fail with EOF
- **Cause**: Parallel pushes saturate upload bandwidth, all connections time out
- **Fix**: Push sequentially; verify each before starting next

### Tag already exists with different content
- **Cause**: Re-using chunk names after changing chunk size (e.g. 500MB → 1.4GB)
- **Fix**: Use descriptive, version-specific chunk names; never reuse the same name for different-sized chunks

---

## Disk Space Requirements

On the machine doing the splitting:
- **Compressed tar**: ~50% of original image size
- **Split chunks**: ~50% of original image size (same content, just split)
- **Chunk Docker images**: same as chunks (Docker stores layers)
- **Total**: ~2× compressed size = ~1× original image size

Example: 10GB image → ~5GB compressed → need ~10GB free disk on splitting machine.

On GitHub Actions (`ubuntu-latest`):
- Default free disk: ~29GB
- After `rm -rf /usr/share/dotnet /usr/local/lib/android /opt/ghc`: ~40GB
- Sufficient for images up to ~20GB

---

## Quick Reference

```bash
# Check upload speed before deciding method
dd if=/dev/urandom bs=1M count=10 2>/dev/null | \
    curl -s -X POST https://httpbin.org/post \
    -H "Content-Type: application/octet-stream" \
    --data-binary @- \
    -w "upload: %{speed_upload} bytes/sec\n" -o /dev/null

# < 50,000 bytes/sec (< 0.4 Mbps) → use GitHub Actions
# > 50,000 bytes/sec → local push may work with 500MB chunks

# Verify all chunks are on Docker Hub
for s in aa ab ac ad; do
    docker manifest inspect myorg/myimage-part-${s}:1.0 2>/dev/null \
        && echo "✅ ${s}" || echo "❌ ${s}"
done
```
