"""
serving/downloads.py — 后台下载权重，运维页轮询进度。

真正的下载、校验、多源择优用 perception 的 `model_downloader.ensure_verified_bundle`
（镜像构建时从 phanthymotus 拷进来，不另写一份）。这里只做三件事：放进后台线程、
记录进度、下完写安装标记。

一次只下一个模型：几个十几 GB 的下载并行，只会让每一个都更慢，而且会一起占满
机器人仅有的上行带宽。
"""

from __future__ import annotations

import logging
import shutil
import threading
import time

from serving.registry import ModelSpec, is_installed, mark_installed, model_dir

log = logging.getLogger(__name__)


def _default_ensure_bundle(name, target_dir, sources, files, progress_cb):
    from model_downloader import ensure_verified_bundle   # 镜像里在 /work 下
    return ensure_verified_bundle(name, target_dir, sources, files, progress_cb=progress_cb)


class DownloadManager:
    def __init__(self, models_root: str, ensure_bundle=_default_ensure_bundle):
        self.models_root = models_root
        self._ensure_bundle = ensure_bundle
        self._lock = threading.Lock()
        self._jobs: dict[str, dict] = {}
        self._active: str | None = None

    def start(self, spec: ModelSpec) -> dict:
        if is_installed(spec, self.models_root):
            return {"model": spec.id, "state": "done", "pct": 100}
        with self._lock:
            if self._active and self._active != spec.id:
                raise RuntimeError(f"正在下载 {self._active}，等它完成再下一个")
            if self._active == spec.id:
                return dict(self._jobs[spec.id])
            self._active = spec.id
            self._jobs[spec.id] = {"model": spec.id, "state": "downloading", "pct": 0,
                                   "mb_done": 0, "mb_total": round(spec.size_bytes / 2**20),
                                   "error": "", "started_at": time.time()}
        threading.Thread(target=self._run, args=(spec,), daemon=True,
                         name=f"download-{spec.id}").start()
        return self.job(spec.id)

    def job(self, model_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(model_id)
            return dict(job) if job else None

    def jobs(self) -> dict[str, dict]:
        with self._lock:
            return {k: dict(v) for k, v in self._jobs.items()}

    @property
    def active(self) -> str | None:
        with self._lock:
            return self._active

    def remove(self, spec: ModelSpec) -> None:
        with self._lock:
            if self._active == spec.id:
                raise RuntimeError(f"{spec.id} 正在下载，不能删")
            self._jobs.pop(spec.id, None)
        target = model_dir(self.models_root, spec.id)
        if target.exists():
            shutil.rmtree(target)

    def _progress(self, model_id: str):
        def cb(pct, mb_done, mb_total):
            with self._lock:
                job = self._jobs.get(model_id)
                if job:
                    job.update(pct=int(pct), mb_done=int(mb_done), mb_total=int(mb_total))
        return cb

    def _run(self, spec: ModelSpec) -> None:
        target = str(model_dir(self.models_root, spec.id))
        try:
            self._ensure_bundle(spec.id, target, spec.sources(), spec.files,
                                self._progress(spec.id))
            mark_installed(spec, self.models_root)
            update = {"state": "done", "pct": 100}
            log.info("[download] %s ready at %s", spec.id, target)
        except Exception as error:      # noqa: BLE001 — 任何失败都要落到运维页上
            log.exception("[download] %s failed", spec.id)
            update = {"state": "error", "error": f"{type(error).__name__}: {error}"}
        with self._lock:
            self._jobs[spec.id].update(update, finished_at=time.time())
            self._active = None
