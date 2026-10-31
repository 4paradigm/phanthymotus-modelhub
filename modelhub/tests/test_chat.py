"""运维页对话：后台流式 → 轮询增量；停止会断开到网关的连接。"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

import pytest

from serving.chat import ChatJob, validate_messages


def _sse(obj) -> bytes:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode()


class _Gateway:
    """假网关：先吐两段思考、再逐字吐正文，每块间隔 GAP；最后一帧带 usage。"""

    GAP = 0.05

    def __init__(self, n=6, status=200):
        self.aborted = threading.Event()
        self.bodies = []
        gw = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def do_POST(self):
                gw.bodies.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                if status != 200:
                    body = json.dumps({"error": {"message": "模型 x 当前状态 loading"}}).encode()
                    self.send_response(status)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                frames = [{"choices": [{"delta": {"reasoning_content": "想"}}]},
                          {"choices": [{"delta": {"reasoning_content": "一下"}}]}]
                frames += [{"choices": [{"delta": {"content": c}}]} for c in "你好世界！！"[:n]]
                frames += [{"choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": n + 2}}]
                try:
                    for f in frames:
                        data = _sse(f)
                        self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
                        self.wfile.flush()
                        time.sleep(gw.GAP)
                    self.wfile.write(b"0\r\n\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    gw.aborted.set()

        class S(ThreadingMixIn, HTTPServer):
            daemon_threads = True

        self.server = S(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


def _wait(pred, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end and not pred():
        time.sleep(0.02)
    return pred()


MSGS = [{"role": "user", "content": "你好"}]


def test_poll_returns_increments_and_stats():
    gw = _Gateway()
    job = ChatJob(gw.url, "m", MSGS, think=True).start()
    seen, pos, rpos = "", 0, 0
    while True:
        p = job.poll(pos, rpos)
        seen += p["text"]
        pos, rpos = p["text_len"], p["reasoning_len"]
        if p["state"] != "running":
            break
        time.sleep(0.03)
    assert seen == "你好世界！！"
    assert job.reasoning == "想一下"
    assert p["state"] == "done"
    assert p["stats"]["tokens"] == 8 and p["stats"]["prompt_tokens"] == 9
    assert p["stats"]["ttft_ms"] is not None and p["stats"]["tok_per_s"] > 0
    body = gw.bodies[0]
    assert body["stream"] is True and body["chat_template_kwargs"] == {"enable_thinking": True}


def test_stop_aborts_the_upstream_generation():
    """停止要断开到网关的连接，否则 vLLM 会在没人看的时候把整段生成完。"""
    gw = _Gateway()
    gw.GAP = 0.3
    job = ChatJob(gw.url, "m", MSGS).start()
    assert _wait(lambda: job.poll()["text_len"] or job.poll()["reasoning_len"])
    job.stop()
    assert _wait(lambda: job.poll()["state"] == "stopped")
    assert gw.aborted.wait(3)


def test_gateway_error_reaches_the_page_in_plain_words():
    gw = _Gateway(status=503)
    job = ChatJob(gw.url, "m", MSGS).start()
    assert _wait(lambda: job.poll()["state"] == "error")
    assert "503" in job.error and "loading" in job.error


def test_messages_are_validated():
    with pytest.raises(ValueError):
        validate_messages([])
    with pytest.raises(ValueError):
        validate_messages([{"role": "tool", "content": "x"}])
    with pytest.raises(ValueError, match="最后一条"):
        validate_messages([{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}])
    with pytest.raises(ValueError, match="太长"):
        validate_messages([{"role": "user", "content": "字" * 30_001}])
