#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# prepare_vllm_base.sh — ModelHub 基础镜像的共用构建逻辑。别直接调它，用每条
# JetPack 线自己的入口：prepare_jp_v71.sh / prepare_jp_v72.sh（同 phanthymotus 的
# prepare_jp_v511.sh / prepare_jp_v61.sh：一条线一个脚本，写死那条线的事实）。
#
# 入口脚本先设好：
#   JP_LINE       7.1 / 7.2
#   L4T           这条线的 L4T 版本（只用于记录）
#   DRIVER_CUDA   这条线宿主机驱动的 CUDA 版本，镜像的 CUDA 不能高于它
#   NGC_TAG       nvcr.io/nvidia/vllm 的版本（记录用）
#   NGC_DIGEST    该版本 arm64 manifest 的 digest（FROM 按它固定，tag 会被覆盖）
#
# 为什么镜像 CUDA 不能高于驱动：驱动（libcuda）只能用宿主机的，它的 PTX JIT 不认识
# 更新工具链生成的 PTX。星源智 N4（R38.4，驱动 13.0）上实测，CUDA 13.4 的 vLLM 在
# 引擎初始化时报 `the provided PTX was compiled with an unsupported toolchain`。
#
# 必须在 arm64 主机上跑（Thor 本机即可，直连 nvcr.io；N4 上实测 ~9 MB/s）。
# 配了 registry（.env）就推到 TCR；没配就只留本地 tag，供本机 build_modelhub.sh 用：
#   BASE_IMAGE=local/phanthy-motus/jetson-base:jp71-vllm ./deploy/build_modelhub.sh --jp-version 7.1
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

: "${JP_LINE:?} ${L4T:?} ${DRIVER_CUDA:?} ${NGC_TAG:?} ${NGC_DIGEST:?}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PHANTHYMOTUS_DIR="${PHANTHYMOTUS_DIR:-$(cd "${REPO_ROOT}/.." && pwd)/phanthymotus}"

ENV_FILE="${ENV_FILE:-${SCRIPT_DIR}/.env}"
[ -f "${ENV_FILE}" ] || ENV_FILE="${PHANTHYMOTUS_DIR}/deploy/.env"
[ -f "${ENV_FILE}" ] && source "${ENV_FILE}"

case "$(uname -m)" in
    aarch64|arm64) ;;
    *) echo "[error] 要在 arm64 主机上构建（上游 manifest 按 arm64 固定）。" >&2; exit 1 ;;
esac

JP_ARG="${JP_LINE//./}"
PUSH=true
if [ -z "${REGISTRY:-}" ] || [ -z "${REGISTRY_USER:-}" ] || [ -z "${REGISTRY_PASSWORD:-}" ] || [ -z "${IMAGE_NAMESPACE:-}" ]; then
    echo "[info] Registry not configured — building locally only (no push)."
    PUSH=false
    REGISTRY="local"
    IMAGE_NAMESPACE="${IMAGE_NAMESPACE:-phanthy-motus}"
fi
UPSTREAM="nvcr.io/nvidia/vllm@${NGC_DIGEST}"
TARGET="${REGISTRY}/${IMAGE_NAMESPACE}/jetson-base:jp${JP_ARG}-vllm"

echo "============================================"
echo "Building ModelHub base image (JetPack ${JP_LINE})"
echo "L4T     : ${L4T} (driver CUDA ${DRIVER_CUDA})"
echo "Upstream: nvcr.io/nvidia/vllm:${NGC_TAG}"
echo "          ${UPSTREAM}"
echo "Target  : ${TARGET}"
echo "Push    : ${PUSH}"
echo "============================================"

TMPDIR_BUILD="$(mktemp -d)"
trap 'rm -rf "${TMPDIR_BUILD}"' EXIT
cat > "${TMPDIR_BUILD}/Dockerfile" <<DOCKERFILE
FROM ${UPSTREAM}

# 这条线的事实，镜像里一处可查（同 perception 的 /etc/jetpack.env）
RUN printf '%s\n' \
      "JP_LINE=${JP_LINE}" "L4T=${L4T}" "DRIVER_CUDA=${DRIVER_CUDA}" \
      "NGC_VLLM_TAG=${NGC_TAG}" "NGC_VLLM_DIGEST=${NGC_DIGEST}" \
      "CUDA_VERSION=\${CUDA_VERSION}" \
      "VLLM_VERSION=\$(python3 -c 'import vllm; print(vllm.__version__)')" \
      > /etc/jetpack.env && cat /etc/jetpack.env

# 守门：镜像 CUDA 高于驱动就别产出这个镜像 —— 否则要到机器上 vLLM 初始化才炸
RUN python3 - <<'PY'
import os
cuda = tuple(int(x) for x in os.environ["CUDA_VERSION"].split(".")[:2])
drv = tuple(int(x) for x in "${DRIVER_CUDA}".split(".")[:2])
assert cuda <= drv, f"base CUDA {cuda} is newer than the JetPack ${JP_LINE} driver's {drv}"
print("CUDA", cuda, "<= driver", drv, "ok")
PY

# 冒烟：ModelHub 运行时要用的东西都在
RUN python3 -c "import vllm, torch, yaml, requests; print('vllm', vllm.__version__, 'torch', torch.__version__)"
DOCKERFILE

docker build -t "${TARGET}" "${TMPDIR_BUILD}"

if ${PUSH}; then
    echo "${REGISTRY_PASSWORD}" | docker login "${REGISTRY}" -u "${REGISTRY_USER}" --password-stdin
    echo "Pushing → ${TARGET}"
    docker push "${TARGET}"
fi
echo ""
echo "Done. Image available at:"
echo "  ${TARGET}"
