"""
serving/cuda_compat.py — 这个镜像里的 CUDA 能不能在本机的显卡驱动上跑。

Jetson 的 CUDA 没有前向兼容：驱动只认识它那一版及更早的 CUDA。给 JetPack 7.2 构建的
ModelHub（CUDA 13.2）装到 JetPack 7.1 的机器上（驱动只到 13.0），vLLM 启动时报
CUDA_ERROR_SYSTEM_DRIVER_MISMATCH，而运维页上看到的是 vLLM 退出前最后几行堆栈 ——
2026-10-31 thor_pack 从部署列表默认装上 jp7.2 版时就是这样，根因埋在日志前面。

所以在启动模型之前先比一次：驱动版本取自 libcuda 的 cuDriverGetVersion（容器里调到的
是宿主机挂进来的驱动），镜像的 CUDA 取自 NGC 镜像都设的 CUDA_VERSION。比不出来（不在
Jetson 上、变量没设）就不拦。
"""

from __future__ import annotations

import ctypes
import os

# 驱动支持的 CUDA → JetPack 线，用来把「换哪个版本」说具体
DRIVER_JETPACK = {(13, 0): "7.1", (13, 2): "7.2"}


def driver_cuda() -> tuple[int, int] | None:
    try:
        lib = ctypes.CDLL("libcuda.so.1")
        v = ctypes.c_int()
        if lib.cuDriverGetVersion(ctypes.byref(v)) != 0 or v.value <= 0:
            return None
        return v.value // 1000, (v.value % 1000) // 10
    except (OSError, AttributeError):
        return None


def image_cuda(env: dict | None = None) -> tuple[int, int] | None:
    raw = (env if env is not None else os.environ).get("CUDA_VERSION", "")
    parts = raw.split(".")
    try:
        return int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        return None


def mismatch(driver: tuple[int, int] | None, image: tuple[int, int] | None) -> str:
    """镜像比驱动新时给一句说清楚怎么办的话；能跑或比不出来时返回空串。"""
    if not driver or not image or image <= driver:
        return ""
    jp = DRIVER_JETPACK.get(driver)
    host = f"（JetPack {jp}）" if jp else ""
    target = f" jetson-jp{jp} 版本" if jp else "和本机 JetPack 对应的版本"
    return (f"这个 ModelHub 镜像带的是 CUDA {image[0]}.{image[1]}，这台机器的显卡驱动只支持到 "
            f"CUDA {driver[0]}.{driver[1]}{host}，模型跑不起来。"
            f"请到「设置 → 部署」把 ModelHub 换成{target}。")


def check() -> str:
    return mismatch(driver_cuda(), image_cuda())
