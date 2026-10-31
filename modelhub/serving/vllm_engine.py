"""
serving/vllm_engine.py — 一个 vLLM 进程的生命周期：起、探活、收日志、停。

vLLM 是现成的推理引擎，这里不碰它的内部，只管它作为子进程：
  - 起：`vllm serve <dir>`，只绑 127.0.0.1 的内部端口。对外统一走 gateway。
  - 探活：加载期间每秒问 /health；就绪后低频问；进程退出立即判 error。
  - 日志：stdout/stderr 合流进一个环形缓冲，运维页看最后几百行。
  - 停：对整个进程组 SIGTERM，超时再 SIGKILL —— vLLM 会派生 worker 进程，
    只杀父进程会留下占着几十 GB 显存的孤儿。

线程模型（和 perception 的插件并发规则同一套）：状态读写全在 self._lock 下，
但不在锁里等进程、等网络。stop 可以在 start 还在加载时到达，并能取消它。
"""

from __future__ import annotations

import collections
import json
import logging
import re
import os
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request

log = logging.getLogger(__name__)

IDLE, LOADING, READY, STOPPING, ERROR = "idle", "loading", "ready", "stopping", "error"


def total_memory_gb(meminfo: str = "/proc/meminfo") -> float:
    """整机内存。Jetson 是统一内存，CPU 和 GPU 共用这一份。"""
    try:
        with open(meminfo) as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except OSError:
        pass
    return 0.0


# 预算之外、vLLM 进程在整机内存里的额外占用（GB）。
#
# 预算只约束 vLLM 自己记账的 GPU 侧（权重 + KV cache + CUDA graph）。在统一内存上，
# CPU 侧的 Python/torch/CUDA 上下文/编译 worker 也吃同一份内存，而且 CUDA graph 实际
# 比 vLLM 的估算大。Thor 实测（Qwen3-0.6B，预算 6 GB）：停掉 vLLM 前后 MemAvailable
# 差 9.8 GB —— 多出 3.8 GB；CUDA graph 实际 0.64 GB、估算 0.08 GB，又多 0.56 GB。
DEFAULT_HOST_OVERHEAD_GB = 4.4


def footprint_gb(budget_gb: float, overhead_gb: float = DEFAULT_HOST_OVERHEAD_GB) -> float:
    """这个模型跑起来在整机内存里大约占多少 —— 运维页显示的就是这个数。"""
    return round(budget_gb + overhead_gb, 1)


def memory_fraction(budget_gb: float, total_gb: float,
                    overhead_gb: float = DEFAULT_HOST_OVERHEAD_GB) -> float:
    """把「给这个模型多少 GB」换成 vLLM 的 --gpu-memory-utilization。

    vLLM 的这个参数是占「设备总显存」的比例。在独立显卡上那是显卡自己的显存；
    在 Jetson 上设备总显存就是整机内存，默认 0.9 意味着要走整机 90% —— perception、
    agent-core 和系统都在这里面。所以必须让人填绝对值，由这里换算。

    上限检查按实际占用（预算 + overhead_gb）算，不按预算算：只看预算会放行一个
    实际要吃掉整机 97% 的配置。
    """
    if total_gb <= 0:
        raise ValueError("读不到整机内存，无法换算显存比例")
    if budget_gb <= 0:
        raise ValueError(f"内存预算必须大于 0，收到 {budget_gb}")
    actual = footprint_gb(budget_gb, overhead_gb)
    if actual > 0.9 * total_gb:
        raise ValueError(f"内存预算 {budget_gb:.1f} GB 加上进程开销约 {overhead_gb:.1f} GB，"
                         f"实际约占 {actual:.1f} GB，超过整机 {total_gb:.1f} GB 的 90%，"
                         f"会把系统和其他服务挤死")
    return round(max(budget_gb / total_gb, 0.02), 4)


# 预算里除权重和 KV cache 之外要留的：CUDA 上下文等非 torch 部分 ~2.1 GB、激活峰值 0.4–0.7 GB、
# CUDA graph ~0.6 GB（Thor 上 Qwen3-0.6B 实测）。
NON_KV_MARGIN_GB = 2.5


def kv_cache_bytes(budget_gb: float, weights_bytes: int) -> int:
    """预算 - 权重 - 余量 = 给 KV cache 的字节数。

    为什么不让 vLLM 自己算：它的算法是「profile 前后 CUDA 空闲内存的差」，而 Thor 上
    CUDA 的空闲内存就是内核的 MemFree，不算 page cache。profile 期间内核回收 page
    cache，「空闲」反而变多，vLLM 直接断言失败 —— N4 上刚做完一次镜像构建（几十 GB
    的磁盘读写）就启动 Qwen3 4B，实测必现。下完模型紧接着加载也是同样的处境。
    """
    kv = budget_gb - weights_bytes / 2**30 - NON_KV_MARGIN_GB
    if kv < 0.5:
        raise ValueError(f"内存预算 {budget_gb:.1f} GB 放不下：权重约 {weights_bytes / 2**30:.1f} GB，"
                         f"另需约 {NON_KV_MARGIN_GB} GB 运行开销，KV cache 至少 0.5 GB")
    return int(kv * 2**30)


_DTYPE_BYTES = {"bfloat16": 2, "float16": 2, "half": 2, "float32": 4, "float": 4,
                "fp8": 1, "fp8_e4m3": 1, "fp8_e5m2": 1}


def kv_bytes_per_token(config: dict, kv_cache_dtype: str | None = None) -> int | None:
    """一个 token 在 KV cache 里占多少字节：2（K 和 V）× 全注意力层数 × KV 头数 × 头维度 × 精度。

    读模型自己的 config.json。多模态模型的语言部分在 text_config 里；混合注意力的模型
    （Qwen3.5：每 4 层只有 1 层是全注意力，其余是线性注意力，状态大小与长度无关）只算
    full_attention 那几层。算不出来就返回 None，页面上不给估计。
    """
    c = config.get("text_config") or config
    try:
        layers = int(c["num_hidden_layers"])
        types = c.get("layer_types")
        if isinstance(types, list) and types:
            layers = sum(1 for t in types if t == "full_attention")
        heads = int(c.get("num_key_value_heads") or c["num_attention_heads"])
        head_dim = int(c.get("head_dim") or int(c["hidden_size"]) // int(c["num_attention_heads"]))
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
    dtype = (kv_cache_dtype if kv_cache_dtype not in (None, "", "auto")
             else c.get("torch_dtype") or c.get("dtype") or config.get("torch_dtype") or "bfloat16")
    return 2 * layers * heads * head_dim * _DTYPE_BYTES.get(str(dtype).lower(), 2)


def build_command(model_path: str, served_name: str, port: int, serve: dict,
                  budget_gb: float, total_gb: float, executable: str = "vllm",
                  overhead_gb: float = DEFAULT_HOST_OVERHEAD_GB,
                  weights_bytes: int | None = None) -> list[str]:
    frac = memory_fraction(budget_gb, total_gb, overhead_gb)      # 也做「不超过整机 90%」的检查
    cmd = [executable, "serve", model_path,
           "--served-model-name", served_name,
           "--host", "127.0.0.1", "--port", str(port)]
    if weights_bytes:
        # KV cache 直接给定（跳过 vLLM 的显存 profiling，见 kv_cache_bytes）。此时
        # gpu-memory-utilization 只剩启动时那道「比例 × 总量 ≤ CUDA 空闲」的检查在用，
        # 而 CUDA 空闲 = MemFree，page cache 多时只有 1 GB 上下 —— 压到最低让它别挡路。
        cmd += ["--kv-cache-memory-bytes", str(kv_cache_bytes(budget_gb, weights_bytes)),
                "--gpu-memory-utilization", "0.01"]
    else:
        cmd += ["--gpu-memory-utilization", str(frac)]
    if serve.get("max_model_len"):
        cmd += ["--max-model-len", str(int(serve["max_model_len"]))]
    if serve.get("tool_call_parser"):
        # agent-core 每一轮都带 tools 调用，不开这两个参数模型只会把工具调用写成正文
        cmd += ["--enable-auto-tool-choice", "--tool-call-parser", str(serve["tool_call_parser"])]
    if serve.get("reasoning_parser"):
        cmd += ["--reasoning-parser", str(serve["reasoning_parser"])]
    for key, flag in (("quantization", "--quantization"), ("dtype", "--dtype"),
                      ("kv_cache_dtype", "--kv-cache-dtype"),
                      ("max_num_seqs", "--max-num-seqs")):
        if serve.get(key) not in (None, ""):
            cmd += [flag, str(serve[key])]
    if serve.get("enforce_eager"):
        cmd.append("--enforce-eager")
    sizes = serve.get("cudagraph_capture_sizes")
    if sizes:
        # vLLM 默认按高并发服务捕获 51 + 35 种批大小的 CUDA graph，Thor 上光这一步
        # 就 2×62 s，而且每次启动都重做（编译缓存只省编译，省不了捕获）。机器人上
        # 决策核心同时只有一两路请求，只捕获小批量。
        if not isinstance(sizes, list) or not all(isinstance(n, int) and n > 0 for n in sizes):
            raise ValueError("serve.cudagraph_capture_sizes 必须是正整数列表")
        cmd += ["--compilation-config",
                json.dumps({"cudagraph_capture_sizes": sorted(set(sizes))})]
    extra = serve.get("extra_args") or []
    if not isinstance(extra, list) or not all(isinstance(a, str) for a in extra):
        raise ValueError("serve.extra_args 必须是字符串列表")
    return cmd + extra


def cache_env(cache_dir: str) -> dict:
    """vLLM 及其依赖会写缓存的每个位置，全部指到持久卷上。

    默认它们都在容器内的 ~/.cache 下，容器一重建就丢：Thor 实测 Qwen3-0.6B 冷启动的
    编译要 18 s，缓存命中后 0.4 s。HOME/XDG 也指过来，兜住没有专门环境变量的库。
    """
    return {
        "VLLM_CACHE_ROOT": f"{cache_dir}/vllm",
        "TORCHINDUCTOR_CACHE_DIR": f"{cache_dir}/inductor",
        "TRITON_CACHE_DIR": f"{cache_dir}/triton",
        "XDG_CACHE_HOME": f"{cache_dir}/xdg",
        "HF_HOME": f"{cache_dir}/hf",
    }


# vLLM 的日志带终端颜色码（`\x1b[1;36m(APIServer pid=139)\x1b[0;0m INFO ...`），原样显示在
# 运维页上就是一串方框；进程前缀 `(APIServer pid=139)` 对看日志的人也没有用。
_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_PROC_PREFIX = re.compile(r"^\((?:APIServer|EngineCore[^)]*) pid=\d+\)\s*")
LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")
_KV_SIZE = re.compile(r"GPU KV cache size: ([\d,]+) tokens")
_OWN_POLLS = re.compile(r'"GET /(metrics|health) HTTP/1\.[01]" 200')
_LEVEL = re.compile(r"\b(DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL)\b")


def clean_log_line(line: str) -> str:
    return _PROC_PREFIX.sub("", _ANSI.sub("", line)).rstrip()


def log_level(line: str) -> str:
    """一行日志的级别；Traceback 和异常行按 ERROR 算，认不出的按 INFO。"""
    m = _LEVEL.search(line[:80])
    if m:
        lv = m.group(1)
        return {"WARN": "WARNING", "CRITICAL": "ERROR"}.get(lv, lv)
    if line.startswith(("Traceback", "  File ")) or "Error:" in line[:120]:
        return "ERROR"
    return "INFO"


class VLLMEngine:
    """一个模型、一个进程、一个内部端口。换模型 = 停掉这个、起一个新的。"""

    def __init__(self, *, model_id: str, model_path: str, port: int, serve: dict,
                 budget_gb: float, total_gb: float, load_timeout_s: float = 1800.0,
                 executable: str = "vllm", log_lines: int = 500,
                 overhead_gb: float = DEFAULT_HOST_OVERHEAD_GB, cache_dir: str | None = None,
                 weights_bytes: int | None = None,
                 popen=subprocess.Popen, env: dict | None = None):
        self.model_id = model_id
        self.serve = dict(serve)
        self.port = port
        self.base_url = f"http://127.0.0.1:{port}/v1"
        self.budget_gb = budget_gb
        self.footprint_gb = footprint_gb(budget_gb, overhead_gb)
        self.command = build_command(model_path, model_id, port, serve, budget_gb,
                                     total_gb, executable, overhead_gb, weights_bytes)
        self._load_timeout_s = load_timeout_s
        self._popen = popen
        self._env = {**cache_env(cache_dir), **(env or {})} if cache_dir else dict(env or {})
        self._lock = threading.RLock()
        self._state = IDLE
        self._error = ""
        self._proc = None
        self._started_at = 0.0
        self._ready_at = 0.0
        self._stop_requested = threading.Event()
        self._logs: collections.deque[str] = collections.deque(maxlen=log_lines)
        self.kv_tokens: int | None = None       # 引擎启动后自己报告的 KV cache 容量

    # ── 对外 ────────────────────────────────────────────────────────────────

    def start(self) -> None:
        with self._lock:
            if self._state in (LOADING, READY):
                return
            self._state, self._error = LOADING, ""
            self._started_at, self._ready_at = time.monotonic(), 0.0
            self._stop_requested.clear()
        log.info("[vllm] %s: %s", self.model_id, " ".join(self.command))
        try:
            proc = self._popen(self.command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL, start_new_session=True,
                               env={**os.environ, **self._env})
        except OSError as error:
            self._fail(f"起不来 vLLM：{error}")
            return
        with self._lock:
            self._proc = proc
        threading.Thread(target=self._pump_logs, args=(proc,), daemon=True,
                         name=f"vllm-log-{self.model_id}").start()
        threading.Thread(target=self._watch, args=(proc,), daemon=True,
                         name=f"vllm-watch-{self.model_id}").start()
        if self._stop_requested.is_set():   # stop 在 Popen 期间到达
            self.stop()

    def stop(self, timeout_s: float = 30.0) -> None:
        self._stop_requested.set()            # 先发信号，再拿锁：加载中也能被取消
        with self._lock:
            proc = self._proc
            if proc is None:
                if self._state != ERROR:
                    self._state = IDLE
                return
            self._state = STOPPING
        _kill_group(proc, signal.SIGTERM)
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            log.warning("[vllm] %s: %ss 内没退，SIGKILL", self.model_id, timeout_s)
            _kill_group(proc, signal.SIGKILL)
            proc.wait(timeout=10)
        with self._lock:
            self._proc = None
            self._state = IDLE

    def status(self) -> dict:
        with self._lock:
            now = time.monotonic()
            return {
                "model": self.model_id, "state": self._state, "error": self._error,
                "base_url": self.base_url, "budget_gb": self.budget_gb,
                "footprint_gb": self.footprint_gb, "kv_tokens": self.kv_tokens,
                "pid": self._proc.pid if self._proc else None,
                "uptime_s": round(now - self._ready_at, 1) if self._state == READY else 0,
                "load_s": round((self._ready_at or now) - self._started_at, 1)
                if self._started_at else 0,
                "last_log": self._logs[-1] if self._logs else "",
            }

    def pgid(self) -> int | None:
        """推理进程组号（start_new_session，所以等于 vLLM 主进程 pid）；监控按它汇总内存。"""
        with self._lock:
            return self._proc.pid if self._proc else None

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    def logs(self, lines: int = 200, level: str = "", query: str = "") -> list[str]:
        """最近的日志。level 只保留该级别及以上；query 按子串过滤（不区分大小写）。"""
        with self._lock:
            out = list(self._logs)
        if level:
            floor = LEVELS.index(level.upper()) if level.upper() in LEVELS else 0
            out = [l for l in out if LEVELS.index(log_level(l)) >= floor]
        if query:
            q = query.lower()
            out = [l for l in out if q in l.lower()]
        return out[-max(1, min(int(lines), self._logs.maxlen)):]

    # ── 内部 ────────────────────────────────────────────────────────────────

    def _fail(self, message: str) -> None:
        log.error("[vllm] %s: %s", self.model_id, message)
        with self._lock:
            self._state, self._error = ERROR, message

    def _pump_logs(self, proc) -> None:
        for raw in iter(proc.stdout.readline, b""):
            line = clean_log_line(raw.decode("utf-8", "replace"))
            if _OWN_POLLS.search(line):
                continue        # 监控每 5 秒抓 /metrics、加载时每秒探 /health —— 不留，否则日志全是它
            m = _KV_SIZE.search(line)
            if m:
                self.kv_tokens = int(m.group(1).replace(",", ""))
            if line:
                with self._lock:
                    self._logs.append(line)

    def _healthy(self) -> bool:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as r:
                return r.status == 200
        except (urllib.error.URLError, OSError, ValueError):
            return False

    def _watch(self, proc) -> None:
        deadline = time.monotonic() + self._load_timeout_s
        while not self._stop_requested.is_set():
            code = proc.poll()
            if code is not None:
                if not self._stop_requested.is_set():
                    tail = " | ".join(self.logs(5))
                    self._fail(f"vLLM 退出（code {code}）：{tail}")
                    with self._lock:
                        self._proc = None
                return
            state = self.state
            if state == LOADING:
                if self._healthy():
                    with self._lock:
                        if self._state == LOADING:
                            self._state, self._ready_at = READY, time.monotonic()
                    log.info("[vllm] %s: ready after %.0fs", self.model_id,
                             time.monotonic() - self._started_at)
                elif time.monotonic() > deadline:
                    self._fail(f"{self._load_timeout_s:.0f}s 内没加载完")
                    _kill_group(proc, signal.SIGKILL)
                    return
                self._stop_requested.wait(1.0)
            else:
                self._stop_requested.wait(5.0)


def _kill_group(proc, sig) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), sig)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.send_signal(sig)
        except (ProcessLookupError, OSError):
            pass
