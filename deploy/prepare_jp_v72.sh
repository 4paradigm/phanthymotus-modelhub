#!/usr/bin/env bash
# prepare_jp_v72.sh — ModelHub 基础镜像，JetPack 7.2 线（L4T R39.2，驱动 CUDA 13.2）。
#
# 上游取驱动 CUDA 以内最新的 NGC vLLM：26.04（CUDA 13.2.1 / vLLM 0.19.0）。之前用过
# 26.10（CUDA 13.4 / vLLM 0.31）：它高于驱动，Qwen3-0.6B 能起来只是没碰到要 JIT 的
# 内核，换了模型就会和 7.1 线一样报 PTX 不支持。
#
#   sudo ./deploy/prepare_jp_v72.sh      # 在 arm64 主机上，见 prepare_vllm_base.sh
set -euo pipefail
export JP_LINE=7.2 L4T=R39.2 DRIVER_CUDA=13.2
export NGC_TAG=26.04-py3
export NGC_DIGEST=sha256:ebd2b86dd262729d44df230961bd42d623b39490adef4bf16e7f11dda5a6098d
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/prepare_vllm_base.sh"
