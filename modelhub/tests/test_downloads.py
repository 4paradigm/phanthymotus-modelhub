"""下载队列：续传、暂停/继续、取消、校验、排队、重启恢复 —— 对一个真的 HTTP 服务器跑。"""

import hashlib
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

from serving.downloads import (CANCELLED, DONE, ERROR, PAUSED, DownloadManager, dir_bytes,
                               disk_usage)
from serving.registry import ModelSpec, is_installed, model_dir


class _Files:
    """按 /<repo>/<file> 提供文件；认 Range；可限速；可在第 N 个请求时断开连接。"""

    def __init__(self, files: dict[str, bytes], delay=0.0):
        self.files, self.delay = files, delay
        self.requests = []
        self.drop_after_bytes = None     # 第一次请求只发这么多就断开
        srv = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                name = self.path.split("/", 2)[-1]
                data = srv.files[name]
                rng = self.headers.get("Range")
                start = int(rng.split("=")[1].split("-")[0]) if rng else 0
                srv.requests.append((name, start))
                body = data[start:]
                self.send_response(206 if rng else 200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                sent = 0
                try:
                    for i in range(0, len(body), 4096):
                        piece = body[i:i + 4096]
                        if srv.drop_after_bytes is not None and sent + len(piece) > srv.drop_after_bytes:
                            srv.drop_after_bytes = None
                            self.connection.shutdown(2)
                            return
                        self.wfile.write(piece)
                        sent += len(piece)
                        if srv.delay:
                            time.sleep(srv.delay)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass

        class S(ThreadingMixIn, HTTPServer):
            daemon_threads = True

        self.server = S(("127.0.0.1", 0), H)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}/repo"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


def _spec(srv, files, model_id="m", bad_sha=False):
    meta = {n: {"size": len(b), "sha256": ("0" * 64 if bad_sha else hashlib.sha256(b).hexdigest())}
            for n, b in files.items()}
    spec = ModelSpec(id=model_id, title=model_id, kind="llm", repo="o/r", revision="abc", files=meta)
    object.__setattr__(spec, "sources", lambda: [srv.base + "/{file}"])
    return spec


def _wait(pred, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end and not pred():
        time.sleep(0.02)
    return pred()


FILES = {"config.json": b'{"a": 1}', "model.safetensors": os.urandom(300_000)}


def test_downloads_verifies_and_installs(tmp_path):
    srv = _Files(FILES)
    dm = DownloadManager(str(tmp_path), tmp_path / "state.json", retry_wait_s=0.01)
    spec = _spec(srv, FILES)
    dm.start(spec)
    assert _wait(lambda: dm.job("m")["state"] == DONE)
    assert is_installed(spec, tmp_path)
    assert (model_dir(tmp_path, "m") / "model.safetensors").read_bytes() == FILES["model.safetensors"]
    assert not (model_dir(tmp_path, "m") / ".partial").exists()
    assert dm.job("m")["pct"] == 100


def test_a_dropped_connection_resumes_from_where_it_stopped(tmp_path):
    srv = _Files(FILES)
    srv.drop_after_bytes = 100_000
    dm = DownloadManager(str(tmp_path), retry_wait_s=0.01)
    dm.start(_spec(srv, FILES))
    assert _wait(lambda: dm.job("m")["state"] == DONE)
    starts = [s for n, s in srv.requests if n == "model.safetensors"]
    assert starts[0] == 0 and starts[1] > 0          # 第二次是 Range 续传，不是从头


def test_pause_keeps_the_partial_and_resume_continues(tmp_path):
    srv = _Files(FILES, delay=0.003)
    dm = DownloadManager(str(tmp_path), retry_wait_s=0.01)
    spec = _spec(srv, FILES)
    dm.start(spec)
    assert _wait(lambda: dm.job("m")["bytes_done"] > 50_000)
    dm.pause("m")
    assert _wait(lambda: dm.job("m")["state"] == PAUSED)
    kept = dm.job("m")["bytes_done"]
    assert kept > 0 and (model_dir(tmp_path, "m") / ".partial").exists()
    srv.delay = 0
    dm.resume(spec)
    assert _wait(lambda: dm.job("m")["state"] == DONE)
    starts = [s for n, s in srv.requests if n == "model.safetensors"]
    assert starts[-1] > 0                            # 继续时从断点开始
    assert is_installed(spec, tmp_path)


def test_cancel_removes_the_partial(tmp_path):
    srv = _Files(FILES, delay=0.003)
    dm = DownloadManager(str(tmp_path), retry_wait_s=0.01)
    dm.start(_spec(srv, FILES))
    assert _wait(lambda: dm.job("m")["bytes_done"] > 20_000)
    dm.cancel("m")
    assert _wait(lambda: dm.job("m")["state"] == CANCELLED)
    assert not model_dir(tmp_path, "m").exists()


def test_a_corrupt_file_fails_verification_and_is_not_installed(tmp_path):
    srv = _Files(FILES)
    dm = DownloadManager(str(tmp_path), retry_wait_s=0.01)
    spec = _spec(srv, FILES, bad_sha=True)
    dm.start(spec)
    assert _wait(lambda: dm.job("m")["state"] == ERROR)
    assert "校验失败" in dm.job("m")["error"]
    assert not is_installed(spec, tmp_path)


def test_one_at_a_time_the_rest_queue(tmp_path):
    srv = _Files(FILES, delay=0.002)
    dm = DownloadManager(str(tmp_path), retry_wait_s=0.01)
    dm.start(_spec(srv, FILES, "a"))
    dm.start(_spec(srv, FILES, "b"))
    assert _wait(lambda: dm.active == "a")
    assert dm.job("b")["state"] == "queued" and dm.queue() == ["a", "b"]
    assert _wait(lambda: dm.job("b")["state"] == DONE, 20)


def test_restore_after_restart_continues_the_queue_and_keeps_paused(tmp_path):
    srv = _Files(FILES, delay=0.003)
    state = tmp_path / "state.json"
    dm = DownloadManager(str(tmp_path), state, retry_wait_s=0.01)
    a, b = _spec(srv, FILES, "a"), _spec(srv, FILES, "b")
    dm.start(a)
    dm.start(b)
    dm.pause("b")
    assert _wait(lambda: dm.job("a")["bytes_done"] > 20_000)
    saved = json.loads(state.read_text())
    assert saved["jobs"]["b"]["state"] == PAUSED

    dm.close()                                       # 「进程被杀」：状态文件停在下载中
    assert _wait(lambda: dm.active is None or dm._closing.is_set())
    time.sleep(0.1)
    srv.delay = 0
    dm2 = DownloadManager(str(tmp_path), state, retry_wait_s=0.01)   # 「容器重启」
    dm2.restore({"a": a, "b": b})
    assert _wait(lambda: dm2.job("a")["state"] == DONE)
    assert dm2.job("b")["state"] == PAUSED


def test_disk_helpers(tmp_path):
    (tmp_path / "x").write_bytes(b"1" * 1000)
    assert dir_bytes(tmp_path) == 1000
    assert disk_usage(str(tmp_path))["total_gb"] > 0
