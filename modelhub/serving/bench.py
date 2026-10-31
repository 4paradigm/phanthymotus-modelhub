"""
serving/bench.py — 标准性能测试：同一套请求，测出一个模型 + 一套配置的基础服务性能，
结果存档，以后可以对比着选模型、调配置。

走的是网关（和 agent-core 同一条路），所以测到的就是决策核心实际会拿到的延迟。
每个请求的提示开头带随机串，防止 vLLM 的前缀缓存命中把首 token 测得偏快；输出长度
用 ignore_eos 固定，几次测试之间才可比。

三组：
  首 token 延迟   提示 ~128 / ~1k / ~4k token（按上下文上限截断），各 3 次取中位数，
                  顺带算出预填充速度
  生成速度        单路，提示 ~128，输出 256，3 次中位数；token 间隔 p50 / p99
  并发吞吐        1 / 4 / 8 路同时发（不超过最大并发），提示 ~512，各输出 128
  前缀缓存        agent 的请求形态：同一段 ~3k token 的前缀（系统提示 + 历史），只换结尾。
                  第一次是冷的，之后 3 次命中缓存，比两者的首 token 延迟

测试期间是真流量，服务会变慢。
"""

from __future__ import annotations

import json
import os
import random
import socket
import statistics
import string
import threading
import time
import urllib.request
import uuid
from pathlib import Path

FILLER = "机器人在办公室里巡逻，记录每个房间的温度、灯光和人员情况，并在发现异常时报告。"
TOKENS_PER_FILLER = 28          # 上面这句在 Qwen 系 tokenizer 下约 28 token
PROMPT_LENS = (128, 1024, 4096)
DECODE_TOKENS = 256
CONCURRENCY = (1, 4, 8)


def _shared_prefix(approx_tokens: int) -> str:
    """固定前缀（每次测试开头一个随机串，测试内部各请求共用）。"""
    nonce = "".join(random.choices(string.ascii_letters, k=16))
    return f"[{nonce}] " + FILLER * max(1, approx_tokens // TOKENS_PER_FILLER)


def _prompt(approx_tokens: int) -> str:
    nonce = "".join(random.choices(string.ascii_letters, k=16))
    return f"[{nonce}] " + FILLER * max(1, approx_tokens // TOKENS_PER_FILLER) + "\n请用一句话总结。"


def _one(url: str, model: str, prompt: str, max_tokens: int, timeout: float = 300) -> dict:
    body = {"model": model, "stream": True, "max_tokens": max_tokens, "ignore_eos": True,
            "temperature": 0, "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": prompt}],
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    stamps, usage = [], {}
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            ch = d.get("choices") or []
            if ch and ((ch[0].get("delta") or {}).get("content") or (ch[0].get("delta") or {}).get("reasoning_content")):
                stamps.append(time.perf_counter())
    if not stamps:
        raise RuntimeError("模型没有返回任何内容")
    gen = usage.get("completion_tokens") or len(stamps)
    span = stamps[-1] - stamps[0]
    itl = [b - a for a, b in zip(stamps, stamps[1:])]
    return {"ttft_ms": (stamps[0] - t0) * 1000, "total_s": stamps[-1] - t0,
            "prompt_tokens": usage.get("prompt_tokens"), "gen_tokens": gen,
            "decode_tps": (gen - 1) / span if span > 0 else None, "itl": itl}


def _med(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 1) if xs else None


def _pct(xs, p):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(len(xs) * p))] * 1000, 1) if xs else None


class BenchRunner:
    """一次只跑一个。结果写 <dir>/<id>.json，进度给运维页轮询。"""

    def __init__(self, results_dir: str | Path, gateway_url: str, one=_one):
        self.dir = Path(results_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._url = f"{gateway_url.rstrip('/')}/v1/chat/completions"
        self._one = one
        self._lock = threading.Lock()
        self._run: dict | None = None
        self._cancel = threading.Event()

    def status(self) -> dict | None:
        with self._lock:
            return dict(self._run) if self._run else None

    def start(self, model: str, title: str, config: dict, engine_version: str,
              total_gb: float) -> dict:
        with self._lock:
            if self._run and self._run["state"] == "running":
                raise RuntimeError("已经有一个测试在跑")
            max_len = int(config.get("max_model_len") or 8192)
            seqs = int(config.get("max_num_seqs") or 8)
            lens = [n for n in PROMPT_LENS if n + 64 < max_len] or [min(128, max_len // 2)]
            conc = [c for c in CONCURRENCY if c <= seqs] or [1]
            steps = len(lens) + 1 + len(conc) + 1
            self._cancel.clear()
            self._run = {"id": time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4],
                         "state": "running", "model": model, "title": title, "step": 0,
                         "steps": steps, "label": "预热", "error": "", "started": time.time()}
            plan = {"lens": lens, "conc": conc}
            meta = {"model": model, "title": title, "engine_version": engine_version,
                    "host": socket.gethostname(), "total_gb": round(total_gb, 1),
                    "config": {k: config.get(k) for k in ("memory_gb", "max_model_len", "max_num_seqs",
                                                          "cudagraph_capture_sizes", "enforce_eager")}}
        threading.Thread(target=self._go, args=(plan, meta), daemon=True, name="bench").start()
        return self.status()

    def stop(self) -> None:
        self._cancel.set()

    def _step(self, label: str) -> None:
        if self._cancel.is_set():
            raise InterruptedError
        with self._lock:
            self._run["step"] += 1
            self._run["label"] = label

    def _go(self, plan: dict, meta: dict) -> None:
        model, res = meta["model"], {}
        try:
            self._one(self._url, model, _prompt(64), 16)                      # 预热
            res["ttft"] = []
            for n in plan["lens"]:
                self._step(f"首 token 延迟：提示约 {n} token")
                runs = [self._one(self._url, model, _prompt(n), 16) for _ in range(3)]
                ttft = _med([r["ttft_ms"] for r in runs])
                ptok = _med([r["prompt_tokens"] for r in runs])
                res["ttft"].append({"prompt_tokens": ptok, "ttft_ms": ttft,
                                    "prefill_tps": round(ptok / (ttft / 1000)) if ptok and ttft else None})
            self._step("生成速度：单路")
            runs = [self._one(self._url, model, _prompt(128), DECODE_TOKENS) for _ in range(3)]
            itl = [x for r in runs for x in r["itl"]]
            res["decode"] = {"tps": _med([r["decode_tps"] for r in runs]),
                             "itl_p50_ms": _pct(itl, .5), "itl_p99_ms": _pct(itl, .99)}
            res["concurrency"] = []
            for c in plan["conc"]:
                self._step(f"并发吞吐：{c} 路")
                out, threads = [], []
                t0 = time.perf_counter()
                for _ in range(c):
                    th = threading.Thread(target=lambda: out.append(self._one(self._url, model, _prompt(512), 128)))
                    th.start()
                    threads.append(th)
                for th in threads:
                    th.join()
                wall = time.perf_counter() - t0
                if len(out) < c:
                    raise RuntimeError(f"{c} 路里有 {c - len(out)} 个请求失败")
                res["concurrency"].append({"n": c, "total_tps": round(sum(r["gen_tokens"] for r in out) / wall, 1),
                                           "per_request_tps": _med([r["decode_tps"] for r in out]),
                                           "ttft_ms": _med([r["ttft_ms"] for r in out])})
            self._step("前缀缓存：同一前缀，冷 / 热")
            n = min(3072, max(256, plan["lens"][-1] - 512))
            prefix = _shared_prefix(n)
            cold = self._one(self._url, model, prefix + "\n第 0 轮：请用一句话总结。", 16)
            warm = [self._one(self._url, model, prefix + f"\n第 {i} 轮：请换个说法总结。", 16)
                    for i in range(1, 4)]
            warm_ms = _med([r["ttft_ms"] for r in warm])
            res["prefix_cache"] = {"prompt_tokens": cold["prompt_tokens"],
                                   "cold_ttft_ms": round(cold["ttft_ms"], 1), "warm_ttft_ms": warm_ms,
                                   "speedup": round(cold["ttft_ms"] / warm_ms, 1) if warm_ms else None}
            state, error = "done", ""
        except InterruptedError:
            state, error = "stopped", ""
        except Exception as e:                                                # noqa: BLE001
            state, error = "error", f"{type(e).__name__}: {e}"
        with self._lock:
            run = self._run
            run.update(state=state, error=error, label="")
        if state == "done":
            doc = {**meta, "id": run["id"], "ts": run["started"],
                   "duration_s": round(time.time() - run["started"], 1), "results": res}
            tmp = self.dir / f"{run['id']}.tmp"
            tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=1))
            os.replace(tmp, self.dir / f"{run['id']}.json")

    def list(self) -> list[dict]:
        out = []
        for p in sorted(self.dir.glob("*.json"), reverse=True):
            try:
                out.append(json.loads(p.read_text()))
            except (OSError, ValueError):
                continue
        return out

    def delete(self, run_id: str) -> None:
        if not run_id or "/" in run_id or run_id.startswith("."):
            raise ValueError("测试编号不合法")
        p = self.dir / f"{run_id}.json"
        if not p.is_file():
            raise KeyError(f"没有这次测试：{run_id}")
        p.unlink()
