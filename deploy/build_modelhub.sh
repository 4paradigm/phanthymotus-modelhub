#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# build_modelhub.sh — 构建 ModelHub 镜像（Jetson Thor）
#
# 沿用 phanthymotus 的构建基础设施：直接 source 兄弟检出里的
# deploy/build_common.sh（镜像源选择、x86 交叉编译 / ARM64 原生、jetpack_vars），
# 不拷一份 —— 两份必然漂移。registry 凭证同理，默认读 phanthymotus/deploy/.env。
#
#   ./deploy/build_modelhub.sh --mirror tuna                 # JetPack 7.2（默认）
#   PHANTHYMOTUS_DIR=/path/to/phanthymotus ./deploy/build_modelhub.sh
#
# phanthymotus 检出默认在本仓库旁边（../phanthymotus），本地和 VPC 构建机
# （/root/embodied-ai/）都是这个布局。
#
# 镜像 tag：release.<YYMMDD>.<本仓库 short-sha>-jetson-jp<版本>
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PHANTHYMOTUS_DIR="${PHANTHYMOTUS_DIR:-$(cd "${REPO_ROOT}/.." && pwd)/phanthymotus}"

if [ ! -f "${PHANTHYMOTUS_DIR}/deploy/build_common.sh" ]; then
    echo "找不到 ${PHANTHYMOTUS_DIR}/deploy/build_common.sh" >&2
    echo "把 phanthymotus 检出放在本仓库旁边，或设 PHANTHYMOTUS_DIR。" >&2
    exit 1
fi
source "${PHANTHYMOTUS_DIR}/deploy/build_common.sh"

ENV_FILE="${ENV_FILE:-${SCRIPT_DIR}/.env}"
[ -f "${ENV_FILE}" ] || ENV_FILE="${PHANTHYMOTUS_DIR}/deploy/.env"
if [ -f "${ENV_FILE}" ]; then
    source "${ENV_FILE}"
fi

eval "$(parse_mirror_arg "$@")"
JP_VERSION="7.2"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --jp-version) JP_VERSION="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

RESOURCE_CENTER_URL="${RESOURCE_CENTER_URL:-https://motus.phanthy.com}"
PUSH_ENABLED=true
if [ -z "${REGISTRY:-}" ] || [ -z "${REGISTRY_USER:-}" ] || [ -z "${REGISTRY_PASSWORD:-}" ] || [ -z "${IMAGE_NAMESPACE:-}" ]; then
    echo "[info] Registry not configured — building locally only (no push)."
    PUSH_ENABLED=false
    REGISTRY="${REGISTRY:-local}"
    IMAGE_NAMESPACE="${IMAGE_NAMESPACE:-phanthy-motus}"
fi

DATE="$(date +%y%m%d)"
# 从 `git archive` 解出来的源码（拷到 rig 上原生构建时）没有 .git：那时显式传
# BUILD_COMMIT / PHANTHYMOTUS_COMMIT，tag 仍然对应一个真实提交。
COMMIT="${BUILD_COMMIT:-$(git -C "${REPO_ROOT}" rev-parse --short=7 HEAD)}"
PHANTHYMOTUS_COMMIT="${PHANTHYMOTUS_COMMIT:-$(git -C "${PHANTHYMOTUS_DIR}" rev-parse --short=7 HEAD 2>/dev/null || echo unknown)}"
DOCKERFILE="${REPO_ROOT}/modelhub/Dockerfile.jetson"
BUILD_CONTEXT="${REPO_ROOT}"
TAG="release.${DATE}.${COMMIT}-jetson-jp${JP_VERSION}"

jetpack_vars "${JP_VERSION}" || exit 1
if [ "${JP_ARG}" != "72" ]; then
    echo "ModelHub 目前只有 JetPack 7.2（Thor）的基础镜像，收到 ${JP_VERSION}" >&2
    exit 1
fi
BUILD_ARGS=("JP_VERSION=${JP_ARG}")
# build_common.sh 的 do_build 会把这个数组原样加到 docker build 上
EXTRA_BUILD_FLAGS=(--build-context "phanthymotus=${PHANTHYMOTUS_DIR}")

CPU_ARCH="arm64"
FULL_IMAGE="${REGISTRY}/${IMAGE_NAMESPACE}/modelhub:${TAG}"

echo "============================================"
echo "Building modelhub image (Jetson only)"
echo "JetPack: ${JP_VERSION}"
echo "Image  : ${FULL_IMAGE}"
echo "Shared : phanthymotus@${PHANTHYMOTUS_COMMIT} (${PHANTHYMOTUS_DIR})"
echo "Arch   : ${ARCH} (native=${IS_ARM64})"
echo "Runs on: ${ACC_ARCH} / ${CPU_ARCH}"
echo "Push   : ${PUSH_ENABLED}"
echo "============================================"

if ${PUSH_ENABLED}; then
    echo "${REGISTRY_PASSWORD}" | docker login "${REGISTRY}" -u "${REGISTRY_USER}" --password-stdin
fi

select_mirror
do_build "${DOCKERFILE}" "${BUILD_CONTEXT}" "${FULL_IMAGE}" "${BUILD_ARGS[@]}"

# 下面两行的措辞被 phanthymotus 的 PR 审查 agent 按正则解析，别改
if ${PUSH_ENABLED}; then
    do_push "${FULL_IMAGE}"
    echo ""
    echo "Done. Image pushed: ${FULL_IMAGE}"
else
    echo ""
    echo "Done. Image built locally: ${FULL_IMAGE}"
fi

if ${PUSH_ENABLED} && [ -n "${RESOURCE_CENTER_API_KEY:-}" ]; then
    SYNC_CONFIRM="y"
    if { : >/dev/tty; } 2>/dev/null; then
        printf "Sync to resource-center (%s)? [Y/n]: " "${RESOURCE_CENTER_URL}" >/dev/tty
        read -r SYNC_CONFIRM </dev/tty || SYNC_CONFIRM="y"
    fi
    if [[ ! "${SYNC_CONFIRM}" =~ ^[Nn] ]]; then
        echo "Registering image to resource-center (${RESOURCE_CENTER_URL})..."
        HTTP_STATUS=$(curl -s -o /tmp/rc_register_resp.json -w "%{http_code}" \
            -X POST "${RESOURCE_CENTER_URL}/api/admin/register" \
            -H "Content-Type: application/json" \
            -H "x-api-key: ${RESOURCE_CENTER_API_KEY}" \
            -d "{
                \"imageRef\": \"${FULL_IMAGE}\",
                \"registryImage\": \"modelhub\",
                \"tag\": \"${TAG}\",
                \"category\": \"modelhub\",
                \"acc_arch\": \"${ACC_ARCH}\",
                \"cpu_arch\": \"${CPU_ARCH}\",
                \"name\": \"ModelHub\",
                \"port\": 15740,
                \"description\": \"本机大模型服务 — LLM（vLLM），OpenAI 兼容接口 :15741\",
                \"cards\": [{\"name\": \"llm\", \"type\": \"actuator\"}]
            }")
        if [ "${HTTP_STATUS}" = "200" ] || [ "${HTTP_STATUS}" = "201" ]; then
            echo "Registered: $(cat /tmp/rc_register_resp.json)"
        else
            echo "Warning: registration failed (HTTP ${HTTP_STATUS}): $(cat /tmp/rc_register_resp.json)"
        fi
    else
        echo "跳过同步。"
    fi
fi
