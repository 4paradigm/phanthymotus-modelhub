"""
serving/metrics.py — 运维统计：每 5 秒采一次，存 SQLite，按时间段查。

采什么（Thor 上能拿到的，2026-10-31 在 N4 上逐项确认过）：
  - vLLM 的 Prometheus `/metrics`：生成 / 输入 token 计数（差分成 tok/s）、在跑和
    排队的请求数、KV cache 占用、按结束原因分的完成请求数
  - 网关自己的计数：请求数、4xx / 5xx、客户端中途断开
  - nvidia-smi：GPU 利用率、功耗（Thor 是统一内存，它报不出「显存」）
  - /sys/class/thermal：GPU / CPU / 结温（tj）
  - /proc：整机可用内存、推理进程（整个进程组）的常驻内存

存储：5 秒粒度保留 24 小时，每分钟汇总一次保留 30 天。查询时按时间段选表，再在
服务端分桶到 ~240 个点 —— 运维页走的是 agent-core 的 MCP 代理，一次别回太多。

事件（错误监控）单独一张表：引擎出错、下载失败、网关 5xx，保留 30 天。
"""

from __future__ import annotations

import glob
import logging
import os
import re
import sqlite3
import subprocess
import threading
import time
import urllib.request

log = logging.getLogger(__name__)

SAMPLE_S = 5
RAW_KEEP_S = 24 * 3600
MINUTE_KEEP_S = 30 * 24 * 3600
MAX_POINTS = 240

FIELDS = ("gen_tps", "prompt_tps", "running", "waiting", "kv_pct", "req_rate", "err_rate",
          "mem_used_gb", "proc_gb", "gpu_util", "gpu_power_w", "temp_gpu", "temp_cpu", "temp_tj")
RANGES = {"15m": 900, "1h": 3600, "6h": 6 * 3600, "24h": 86400, "7d": 7 * 86400, "30d": 30 * 86400}

_PROM = re.compile(r"^(vllm:[a-z_]+)\{([^}]*)\}\s+([0-9.eE+-]+)$")


def parse_prometheus(text: str) -> dict:
    """{metric: 求和后的值}。按标签分的计数（如 finished_reason）合在一起。"""
    out: dict[str, float] = {}
    for line in text.splitlines():
        m = _PROM.match(line.strip())
        if m:
            out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(3))
    return out


def read_thermal(base: str = "/sys/class/thermal") -> dict:
    temps = {}
    for zone in glob.glob(f"{base}/thermal_zone*"):
        try:
            kind = open(f"{zone}/type").read().strip()
            temps[kind] = int(open(f"{zone}/temp").read().strip()) / 1000.0
        except (OSError, ValueError):
            continue
    return {"temp_gpu": temps.get("gpu-thermal"), "temp_cpu": temps.get("cpu-thermal"),
            "temp_tj": temps.get("tj-thermal")}


def read_gpu() -> dict:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,power.draw",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             timeout=3).stdout.strip().splitlines()[0]
        util, power = [x.strip() for x in out.split(",")]
        return {"gpu_util": float(util), "gpu_power_w": float(power)}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return {"gpu_util": None, "gpu_power_w": None}


def read_memory(meminfo: str = "/proc/meminfo") -> dict:
    vals = {}
    try:
        for line in open(meminfo):
            k, v = line.split(":", 1)
            vals[k] = int(v.split()[0]) / (1024 * 1024)
    except (OSError, ValueError):
        return {"mem_used_gb": None}
    return {"mem_used_gb": round(vals.get("MemTotal", 0) - vals.get("MemAvailable", 0), 2)}


def group_rss_gb(pgid: int | None) -> float | None:
    """进程组（vLLM 的 API server + EngineCore + worker）的常驻内存之和。"""
    if not pgid:
        return None
    total = 0
    for stat in glob.glob("/proc/[0-9]*/stat"):
        try:
            fields = open(stat).read().rsplit(")", 1)[1].split()
            if int(fields[2]) != pgid:          # fields[2] 是 pgrp
                continue
            for line in open(stat[:-4] + "status"):
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1])
        except (OSError, ValueError, IndexError):
            continue
    return round(total / (1024 * 1024), 2)


class MetricsStore:
    def __init__(self, path: str):
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        cols = ", ".join(f"{f} REAL" for f in FIELDS)
        with self._lock, self._db:
            self._db.execute(f"CREATE TABLE IF NOT EXISTS raw (ts INTEGER PRIMARY KEY, {cols})")
            self._db.execute(f"CREATE TABLE IF NOT EXISTS minute (ts INTEGER PRIMARY KEY, {cols})")
            self._db.execute("CREATE TABLE IF NOT EXISTS events (ts INTEGER, level TEXT, "
                             "source TEXT, message TEXT)")
            self._db.execute("CREATE INDEX IF NOT EXISTS events_ts ON events(ts)")

    def add(self, ts: int, sample: dict) -> None:
        with self._lock, self._db:
            self._db.execute(f"INSERT OR REPLACE INTO raw (ts, {', '.join(FIELDS)}) VALUES "
                             f"(?{', ?' * len(FIELDS)})", [ts] + [sample.get(f) for f in FIELDS])

    def event(self, level: str, source: str, message: str, ts: int | None = None) -> None:
        with self._lock, self._db:
            self._db.execute("INSERT INTO events VALUES (?, ?, ?, ?)",
                             (int(ts or time.time()), level, source, message[:500]))

    def rollup(self, now: int | None = None) -> None:
        """把已经过完的分钟汇总进 minute 表，并按保留期删旧数据。"""
        now = int(now or time.time())
        upto = now - now % 60
        avg = ", ".join(f"AVG({f})" for f in FIELDS)
        with self._lock, self._db:
            last = self._db.execute("SELECT MAX(ts) FROM minute").fetchone()[0] or 0
            self._db.execute(f"INSERT OR REPLACE INTO minute (ts, {', '.join(FIELDS)}) "
                             f"SELECT (ts / 60) * 60 AS m, {avg} FROM raw "
                             f"WHERE ts >= ? AND ts < ? GROUP BY m", (last + 60 if last else 0, upto))
            self._db.execute("DELETE FROM raw WHERE ts < ?", (now - RAW_KEEP_S,))
            self._db.execute("DELETE FROM minute WHERE ts < ?", (now - MINUTE_KEEP_S,))
            self._db.execute("DELETE FROM events WHERE ts < ?", (now - MINUTE_KEEP_S,))

    def query(self, range_key: str, now: int | None = None) -> dict:
        span = RANGES.get(range_key, 3600)
        now = int(now or time.time())
        start = now - span
        table = "raw" if span <= 6 * 3600 else "minute"
        bucket = max(SAMPLE_S if table == "raw" else 60, span // MAX_POINTS)
        avg = ", ".join(f"AVG({f})" for f in FIELDS)
        with self._lock:
            rows = self._db.execute(
                f"SELECT (ts / ?) * ? AS b, {avg} FROM {table} WHERE ts >= ? GROUP BY b ORDER BY b",
                (bucket, bucket, start)).fetchall()
        series = {f: [None if r[i + 1] is None else round(r[i + 1], 2) for r in rows]
                  for i, f in enumerate(FIELDS)}
        return {"range": range_key, "start": start, "end": now, "bucket_s": bucket,
                "ts": [r[0] for r in rows], "series": series}

    def events(self, range_key: str, limit: int = 100, now: int | None = None) -> list[dict]:
        span = RANGES.get(range_key, 86400)
        now = int(now or time.time())
        with self._lock:
            rows = self._db.execute("SELECT ts, level, source, message FROM events WHERE ts >= ? "
                                    "ORDER BY ts DESC LIMIT ?", (now - span, limit)).fetchall()
        return [{"ts": r[0], "level": r[1], "source": r[2], "message": r[3]} for r in rows]


class Sampler:
    """每 SAMPLE_S 秒采一次。`engine()` 返回当前引擎（或 None），`gateway_stats()` 返回
    网关累计计数。差分需要上一次的值，所以计数器类指标从第二次采样开始才有。"""

    def __init__(self, store: MetricsStore, engine, gateway_stats, interval_s: float = SAMPLE_S,
                 fetch=None):
        self.store = store
        self._engine = engine
        self._gateway_stats = gateway_stats
        self._interval = interval_s
        self._fetch = fetch or self._fetch_metrics
        self._prev: dict = {}
        self._stop = threading.Event()
        self.watchers: list = []      # 每次采样后调用，用来记事件

    def start(self) -> "Sampler":
        threading.Thread(target=self._run, daemon=True, name="metrics").start()
        return self

    def stop(self) -> None:
        self._stop.set()

    @staticmethod
    def _fetch_metrics(base_url: str) -> str:
        url = base_url.rstrip("/").removesuffix("/v1") + "/metrics"
        with urllib.request.urlopen(url, timeout=2) as r:
            return r.read().decode("utf-8", "replace")

    def sample(self, now: float | None = None) -> dict:
        now = now or time.time()
        s: dict = {}
        engine = self._engine()
        counters: dict = {}
        if engine is not None and engine.state == "ready":
            try:
                prom = parse_prometheus(self._fetch(engine.base_url))
                s["running"] = prom.get("vllm:num_requests_running")
                s["waiting"] = prom.get("vllm:num_requests_waiting")
                kv = prom.get("vllm:kv_cache_usage_perc", prom.get("vllm:gpu_cache_usage_perc"))
                s["kv_pct"] = None if kv is None else round(kv * 100, 2)
                counters["gen"] = prom.get("vllm:generation_tokens_total")
                counters["prompt"] = prom.get("vllm:prompt_tokens_total")
            except Exception:                    # noqa: BLE001 — 采样失败只是少一个点
                pass
            s["proc_gb"] = group_rss_gb(engine.pgid())
        gw = self._gateway_stats() or {}
        counters["req"] = gw.get("requests")
        counters["err"] = gw.get("errors_5xx")
        prev, dt = self._prev, now - self._prev.get("_t", now)
        if dt > 0:
            for key, field in (("gen", "gen_tps"), ("prompt", "prompt_tps"),
                               ("req", "req_rate"), ("err", "err_rate")):
                a, b = prev.get(key), counters.get(key)
                if a is not None and b is not None and b >= a:   # 引擎重启计数会归零
                    s[field] = round((b - a) / dt, 2)
        a, b = prev.get("err"), counters.get("err")
        if a is not None and b is not None and b > a:
            self.store.event("error", "gateway", f"{int(b - a)} 个请求返回 5xx", int(now))
        self._prev = {**counters, "_t": now}
        s.update(read_gpu())
        s.update(read_thermal())
        s.update(read_memory())
        return s

    def _run(self) -> None:
        last_rollup = 0.0
        while not self._stop.wait(self._interval):
            try:
                now = time.time()
                self.store.add(int(now), self.sample(now))
                for watch in self.watchers:
                    watch()
                if now - last_rollup >= 60:
                    self.store.rollup(int(now))
                    last_rollup = now
            except Exception:                    # noqa: BLE001
                log.exception("[metrics] sample failed")
