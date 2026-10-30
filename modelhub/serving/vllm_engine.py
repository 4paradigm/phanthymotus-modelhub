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
import logging
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


def memory_fraction(budget_gb: float, total_gb: float) -> float:
    """把「给这个模型多少 GB」换成 vLLM 的 --gpu-memory-utilization。

    vLLM 的这个参数是占「设备总显存」的比例。在独立显卡上那是显卡自己的显存；
    在 Jetson 上设备总显存就是整机内存，默认 0.9 意味着要走整机 90% —— perception、
    agent-core 和系统都在这里面。所以必须让人填绝对值，由这里换算。
    """
    if total_gb <= 0:
        raise ValueError("读不到整机内存，无法换算显存比例")
    if budget_gb <= 0:
        raise ValueError(f"内存预算必须大于 0，收到 {budget_gb}")
    frac = budget_gb / total_gb
    if frac > 0.9:
        raise ValueError(f"内存预算 {budget_gb:.1f} GB 超过整机 {total_gb:.1f} GB 的 90%，"
                         f"会把系统和其他服务挤死")
    return round(max(frac, 0.02), 4)


def build_command(model_path: str, served_name: str, port: int, serve: dict,
                  budget_gb: float, total_gb: float, executable: str = "vllm") -> list[str]:
    cmd = [executable, "serve", model_path,
           "--served-model-name", served_name,
           "--host", "127.0.0.1", "--port", str(port),
           "--gpu-memory-utilization", str(memory_fraction(budget_gb, total_gb))]
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
    extra = serve.get("extra_args") or []
    if not isinstance(extra, list) or not all(isinstance(a, str) for a in extra):
        raise ValueError("serve.extra_args 必须是字符串列表")
    return cmd + extra


class VLLMEngine:
    """一个模型、一个进程、一个内部端口。换模型 = 停掉这个、起一个新的。"""

    def __init__(self, *, model_id: str, model_path: str, port: int, serve: dict,
                 budget_gb: float, total_gb: float, load_timeout_s: float = 1800.0,
                 executable: str = "vllm", log_lines: int = 500,
                 popen=subprocess.Popen, env: dict | None = None):
        self.model_id = model_id
        self.port = port
        self.base_url = f"http://127.0.0.1:{port}/v1"
        self.budget_gb = budget_gb
        self.command = build_command(model_path, model_id, port, serve, budget_gb,
                                     total_gb, executable)
        self._load_timeout_s = load_timeout_s
        self._popen = popen
        self._env = env
        self._lock = threading.RLock()
        self._state = IDLE
        self._error = ""
        self._proc = None
        self._started_at = 0.0
        self._ready_at = 0.0
        self._stop_requested = threading.Event()
        self._logs: collections.deque[str] = collections.deque(maxlen=log_lines)

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
                               env={**os.environ, **(self._env or {})})
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
                "pid": self._proc.pid if self._proc else None,
                "uptime_s": round(now - self._ready_at, 1) if self._state == READY else 0,
                "load_s": round((self._ready_at or now) - self._started_at, 1)
                if self._started_at else 0,
                "last_log": self._logs[-1] if self._logs else "",
            }

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    def logs(self, lines: int = 200) -> list[str]:
        with self._lock:
            return list(self._logs)[-max(1, min(int(lines), self._logs.maxlen)):]

    # ── 内部 ────────────────────────────────────────────────────────────────

    def _fail(self, message: str) -> None:
        log.error("[vllm] %s: %s", self.model_id, message)
        with self._lock:
            self._state, self._error = ERROR, message

    def _pump_logs(self, proc) -> None:
        for raw in iter(proc.stdout.readline, b""):
            line = raw.decode("utf-8", "replace").rstrip()
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
