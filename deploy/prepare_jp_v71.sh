#!/usr/bin/env bash
# prepare_jp_v71.sh — ModelHub 基础镜像，JetPack 7.1 线（L4T R38.4，驱动 CUDA 13.0）。
#
# 上游取驱动 CUDA 以内最新的 NGC vLLM：25.11（CUDA 13.0.2 / vLLM 0.11.0）。26.x 全是
# CUDA 13.1+，在这条线上 vLLM 起不来。vLLM 0.11 不认识 Qwen3.5 架构，所以
# PhanthyMotus 0.1 在这条线上不可用（注册表的 min_vllm 会在运维页标出来）。
#
#   sudo ./deploy/prepare_jp_v71.sh      # 在 arm64 主机上，见 prepare_vllm_base.sh
set -euo pipefail
export JP_LINE=7.1 L4T=R38.4 DRIVER_CUDA=13.0
export NGC_TAG=25.11-py3
export NGC_DIGEST=sha256:dfaa939c57e98f9db401e2ed66f247de938b2d2cf802072a90f5b6f3fd6b2a53
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/prepare_vllm_base.sh"
