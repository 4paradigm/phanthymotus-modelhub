"""
serving/downloads.py — 模型权重的下载队列：续传、暂停、继续、取消，重启后接着下。

为什么不再用 perception 的 model_downloader：那套是给几百 MB 的感知模型写的，失败了
整个文件重下、不能取消、没有队列。LLM 权重是几 GB 到几十 GB，在机器人的上行带宽上
一下就是几十分钟 —— 断一次重来、或者下错了停不下来，都不可接受。

做法：
  - 一次只下一个，其余排队（几个大下载并行只会让每个都更慢）。
  - 每个文件先写 `<模型目录>/.partial/<文件>.part`，断了用 HTTP Range 从断点接着下。
  - SHA256 边下边算；续传时先把已有的部分补算一遍。下完立刻比对，不对就删掉重下。
  - 所有文件都对上之后才整体移进模型目录、写安装标记 —— 半截的模型永远不会被加载。
  - 队列状态存在 state/downloads.json：容器重启后，下到一半的、排队的接着下，暂停的
    保持暂停。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import quote

from serving.registry import ModelSpec, is_installed, mark_installed, model_dir

log = logging.getLogger(__name__)

CHUNK = 1 << 20
RETRIES = 6
QUEUED, DOWNLOADING, VERIFYING, PAUSED, DONE, ERROR, CANCELLED = (
    "queued", "downloading", "verifying", "paused", "done", "error", "cancelled")
_ACTIVE_STATES = (QUEUED, DOWNLOADING, VERIFYING)


class _Stop(Exception):
    """暂停或取消：由控制标志触发，不算失败。"""


def _source_url(source: str, filename: str) -> str:
    quoted = "/".join(quote(part) for part in filename.split("/"))
    return source.replace("{file}", quoted) if "{file}" in source \
        else source.rstrip("/") + "/" + quoted


def dir_bytes(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def disk_usage(path: str) -> dict:
    try:
        u = shutil.disk_usage(path)
        return {"total_gb": round(u.total / 1e9, 1), "used_gb": round(u.used / 1e9, 1),
                "free_gb": round(u.free / 1e9, 1)}
    except OSError:
        return {"total_gb": 0, "used_gb": 0, "free_gb": 0}


class DownloadManager:
    def __init__(self, models_root: str, state_path: str | Path | None = None,
                 opener=urllib.request.urlopen, retry_wait_s: float = 2.0):
        self.models_root = models_root
        self._state_path = Path(state_path) if state_path else None
        self._open = opener
        self._retry_wait = retry_wait_s
        self._cv = threading.Condition()
        self._jobs: dict[str, dict] = {}
        self._specs: dict[str, ModelSpec] = {}
        self._order: list[str] = []          # 排队顺序
        self._active: str | None = None
        self._control: dict[str, str] = {}   # model -> "pause" / "cancel"
        self._closing = threading.Event()
        threading.Thread(target=self._worker, daemon=True, name="downloads").start()

    def close(self) -> None:
        """停下工作线程，**不改**队列状态文件 —— 和进程被杀时留下的一样，重启后接着下。"""
        self._closing.set()
        with self._cv:
            self._cv.notify_all()

    # ── 对外 ────────────────────────────────────────────────────────────────

    def start(self, spec: ModelSpec) -> dict:
        """加入队列（已在队列/下载中则原样返回；暂停的会被继续）。"""
        if is_installed(spec, self.models_root):
            return {"model": spec.id, "state": DONE, "pct": 100}
        with self._cv:
            job = self._jobs.get(spec.id)
            if job and job["state"] in _ACTIVE_STATES:
                return dict(job)
            self._specs[spec.id] = spec
            done = self._partial_bytes(spec)
            self._jobs[spec.id] = self._new_job(spec, done)
            if spec.id not in self._order:
                self._order.append(spec.id)
            self._save()
            self._cv.notify_all()
            return dict(self._jobs[spec.id])

    def pause(self, model_id: str) -> dict:
        with self._cv:
            job = self._require(model_id)
            if job["state"] == QUEUED:
                job["state"] = PAUSED
                self._order.remove(model_id)
            elif job["state"] in (DOWNLOADING, VERIFYING):
                self._control[model_id] = "pause"
            self._save()
            return dict(job)

    def resume(self, spec: ModelSpec) -> dict:
        with self._cv:
            job = self._jobs.get(spec.id)
            if job and job["state"] in _ACTIVE_STATES:
                return dict(job)
        return self.start(spec)

    def cancel(self, model_id: str) -> dict:
        with self._cv:
            job = self._require(model_id)
            if self._active == model_id:
                self._control[model_id] = "cancel"
                return dict(job)
            if model_id in self._order:
                self._order.remove(model_id)
            job["state"] = CANCELLED
            self._save()
        self._drop_partial(model_id)
        return self.job(model_id)

    def remove(self, spec: ModelSpec) -> None:
        with self._cv:
            job = self._jobs.get(spec.id)
            if job and job["state"] in _ACTIVE_STATES:
                raise RuntimeError(f"{spec.id} 正在下载，先取消下载")
            self._jobs.pop(spec.id, None)
            self._save()
        target = model_dir(self.models_root, spec.id)
        if target.exists():
            shutil.rmtree(target)

    def job(self, model_id: str | None) -> dict | None:
        with self._cv:
            job = self._jobs.get(model_id) if model_id else None
            return dict(job) if job else None

    def jobs(self) -> dict[str, dict]:
        with self._cv:
            return {k: dict(v) for k, v in self._jobs.items()}

    @property
    def active(self) -> str | None:
        with self._cv:
            return self._active

    def queue(self) -> list[str]:
        with self._cv:
            return ([self._active] if self._active else []) + list(self._order)

    def restore(self, specs: dict[str, ModelSpec]) -> None:
        """容器重启后：下到一半的和排队的接着下，暂停的保持暂停。"""
        if not self._state_path or not self._state_path.is_file():
            return
        try:
            saved = json.loads(self._state_path.read_text())
        except (OSError, ValueError):
            return
        with self._cv:
            for model_id in saved.get("order", []) + list(saved.get("jobs", {})):
                if model_id in self._jobs or model_id not in specs:
                    continue
                state = saved["jobs"].get(model_id, {}).get("state")
                spec = specs[model_id]
                if is_installed(spec, self.models_root) or state in (DONE, CANCELLED, None):
                    continue
                self._specs[model_id] = spec
                job = self._new_job(spec, self._partial_bytes(spec))
                if state == PAUSED:
                    job["state"] = PAUSED
                elif state == ERROR:
                    job.update(state=ERROR, error=saved["jobs"][model_id].get("error", ""))
                else:
                    self._order.append(model_id)
                self._jobs[model_id] = job
            self._cv.notify_all()

    # ── 内部 ────────────────────────────────────────────────────────────────

    def _require(self, model_id: str) -> dict:
        job = self._jobs.get(model_id)
        if not job:
            raise KeyError(f"{model_id} 没有下载任务")
        return job

    def _new_job(self, spec: ModelSpec, done: int) -> dict:
        total = spec.size_bytes
        return {"model": spec.id, "state": QUEUED, "error": "", "file": "",
                "bytes_done": done, "bytes_total": total,
                "pct": int(done * 100 / total) if total else 0,
                "mb_done": done // 2**20, "mb_total": total // 2**20,
                "speed_bps": 0, "eta_s": None, "updated_at": time.time()}

    def _partial_dir(self, model_id: str) -> Path:
        return model_dir(self.models_root, model_id) / ".partial"

    def _partial_bytes(self, spec: ModelSpec) -> int:
        d = self._partial_dir(spec.id)
        return sum(min((d / f"{name}.part").stat().st_size, meta["size"])
                   for name, meta in spec.files.items() if (d / f"{name}.part").is_file())

    def _drop_partial(self, model_id: str) -> None:
        shutil.rmtree(self._partial_dir(model_id), ignore_errors=True)
        target = model_dir(self.models_root, model_id)
        try:
            if target.exists() and not any(target.iterdir()):
                target.rmdir()
        except OSError:
            pass

    def _save(self) -> None:
        if not self._state_path:
            return
        doc = {"order": list(self._order) + ([self._active] if self._active else []),
               "jobs": {k: {"state": v["state"], "error": v.get("error", "")}
                        for k, v in self._jobs.items()}}
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(doc, ensure_ascii=False))
            os.replace(tmp, self._state_path)
        except OSError:
            log.warning("[download] could not persist queue state", exc_info=True)

    def _worker(self) -> None:
        while not self._closing.is_set():
            with self._cv:
                while not self._order and not self._closing.is_set():
                    self._cv.wait()
                if self._closing.is_set():
                    return
                model_id = self._order.pop(0)
                spec = self._specs[model_id]
                self._active = model_id
                self._control.pop(model_id, None)
                self._jobs[model_id]["state"] = DOWNLOADING
                self._save()
            try:
                self._download(spec)
                final = {"state": DONE, "pct": 100, "eta_s": 0, "speed_bps": 0}
                log.info("[download] %s ready", spec.id)
            except _Stop as stop:
                if str(stop) == "shutdown":
                    return                            # 状态文件保持「下载中」，重启后续传
                final = {"state": PAUSED if str(stop) == "pause" else CANCELLED,
                         "speed_bps": 0, "eta_s": None}
                if str(stop) == "cancel":
                    self._drop_partial(spec.id)
                log.info("[download] %s %s", spec.id, final["state"])
            except Exception as error:          # noqa: BLE001 — 落到运维页上
                log.exception("[download] %s failed", spec.id)
                final = {"state": ERROR, "error": f"{type(error).__name__}: {error}",
                         "speed_bps": 0, "eta_s": None}
            with self._cv:
                self._jobs[model_id].update(final, updated_at=time.time())
                self._active = None
                self._control.pop(model_id, None)
                self._save()

    def _check(self, model_id: str) -> None:
        if self._closing.is_set():
            raise _Stop("shutdown")
        with self._cv:
            ctl = self._control.get(model_id)
        if ctl:
            raise _Stop(ctl)

    def _progress(self, model_id: str, base: int, speed: list) -> callable:
        """返回 (已下字节增量) -> None 的回调，顺带算速度（指数平均）和剩余时间。"""
        state = {"t": time.monotonic(), "b": 0}

        def cb(delta_total: int, file_name: str):
            now = time.monotonic()
            with self._cv:
                job = self._jobs[model_id]
                job["bytes_done"] = base + delta_total
                job["file"] = file_name
                dt = now - state["t"]
                if dt >= 1.0:
                    inst = (delta_total - state["b"]) / dt
                    speed[0] = inst if speed[0] == 0 else 0.7 * speed[0] + 0.3 * inst
                    state.update(t=now, b=delta_total)
                    job["speed_bps"] = int(speed[0])
                    left = job["bytes_total"] - job["bytes_done"]
                    job["eta_s"] = int(left / speed[0]) if speed[0] > 0 else None
                job["pct"] = int(job["bytes_done"] * 100 / job["bytes_total"]) if job["bytes_total"] else 0
                job["mb_done"], job["mb_total"] = job["bytes_done"] // 2**20, job["bytes_total"] // 2**20
                job["updated_at"] = time.time()
        return cb

    def _download(self, spec: ModelSpec) -> None:
        partial = self._partial_dir(spec.id)
        partial.mkdir(parents=True, exist_ok=True)
        base = 0
        speed = [0.0]
        for name, meta in sorted(spec.files.items(), key=lambda kv: kv[1]["size"]):
            part = partial / f"{name}.part"
            part.parent.mkdir(parents=True, exist_ok=True)
            cb = self._progress(spec.id, base, speed)
            self._fetch_file(spec, name, meta, part, lambda n, _cb=cb, _f=name: _cb(n, _f))
            base += meta["size"]
        self._check(spec.id)
        with self._cv:
            self._jobs[spec.id].update(state=VERIFYING, file="")
        target = model_dir(self.models_root, spec.id)
        for name in spec.files:
            dest = target / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.replace(partial / f"{name}.part", dest)
        shutil.rmtree(partial, ignore_errors=True)
        mark_installed(spec, self.models_root)

    def _fetch_file(self, spec: ModelSpec, name: str, meta: dict, part: Path, on_bytes) -> None:
        size, want = int(meta["size"]), meta["sha256"]
        digest = hashlib.sha256()
        have = part.stat().st_size if part.exists() else 0
        if have > size:
            part.unlink()
            have = 0
        if have:                                   # 续传：先补算已有部分的哈希
            with open(part, "rb") as f:
                for chunk in iter(lambda: f.read(CHUNK), b""):
                    digest.update(chunk)
                    self._check(spec.id)
        on_bytes(have)
        attempts, sources = 0, spec.sources()
        while have < size:
            self._check(spec.id)
            url = _source_url(sources[attempts % len(sources)], name)
            try:
                req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
                with self._open(req, timeout=60) as resp:
                    if have and getattr(resp, "status", 200) == 200:   # 不认 Range：从头来
                        digest, have = hashlib.sha256(), 0
                        part.unlink(missing_ok=True)
                    with open(part, "ab") as out:
                        while True:
                            self._check(spec.id)
                            # read1：有多少给多少。read(1 MB) 会阻塞到凑满 1 MB，慢链路上
                            # 暂停/取消和进度都要等好几秒才有反应
                            chunk = resp.read1(CHUNK) if hasattr(resp, "read1") else resp.read(CHUNK)
                            if not chunk:
                                break
                            out.write(chunk)
                            digest.update(chunk)
                            have += len(chunk)
                            on_bytes(have)
                            if have > size:
                                break
                attempts = 0
                if have < size:
                    raise OSError(f"连接提前结束（{have}/{size}）")
            except _Stop:
                raise
            except (urllib.error.URLError, OSError) as error:
                attempts += 1
                if attempts > RETRIES:
                    raise RuntimeError(f"{name}: 重试 {RETRIES} 次仍失败：{error}") from error
                log.warning("[download] %s %s: %s, retry %d at %d bytes", spec.id, name, error,
                            attempts, have)
                time.sleep(self._retry_wait * attempts)
        if have != size or digest.hexdigest() != want:
            part.unlink(missing_ok=True)
            raise ValueError(f"{name} 校验失败（大小 {have}/{size}，SHA256 不一致）")
