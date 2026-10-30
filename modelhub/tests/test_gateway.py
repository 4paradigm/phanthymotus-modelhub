"""统一出口网关：路由、未就绪的说法、以及流式逐块透传。"""

import http.client
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

import pytest

from serving.gateway import Router, start_gateway


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Backend:
    """假的 vLLM：/v1/chat/completions 流式吐 3 个 SSE 块，块间隔 GAP 秒。"""

    GAP = 0.4

    def __init__(self, chunks=3):
        self.port = _free_port()
        self.requests = []
        self.aborted = threading.Event()
        backend = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                backend.requests.append((self.path, body))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                try:
                    for i in range(chunks):
                        data = f'data: {{"n": {i}}}\n\n'.encode()
                        self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
                        self.wfile.flush()
                        time.sleep(backend.GAP)
                    self.wfile.write(b"0\r\n\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    backend.aborted.set()

        class S(ThreadingMixIn, HTTPServer):
            daemon_threads = True

        self.server = S(("127.0.0.1", self.port), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


@pytest.fixture()
def setup(request):
    backend = _Backend(getattr(request, "param", 3))
    routes = [{"id": "qwen", "state": "ready", "base_url": f"http://127.0.0.1:{backend.port}/v1"},
              {"id": "big", "state": "loading", "base_url": "http://127.0.0.1:1/v1"}]
    router = Router()
    router.add_provider(lambda: routes)
    port = _free_port()
    gw = start_gateway(router, "127.0.0.1", port)
    yield backend, port, routes
    gw.shutdown()
    backend.server.shutdown()


def _post(port, body, path="/v1/chat/completions"):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request("POST", path, body=json.dumps(body), headers={"Content-Type": "application/json"})
    return conn, conn.getresponse()


def test_stream_is_relayed_chunk_by_chunk_not_buffered(setup):
    """agent-core 靠首 token 期限判卡死；网关攒一下，健康请求就会被判死。"""
    backend, port, _ = setup
    t0 = time.monotonic()
    conn, resp = _post(port, {"model": "qwen", "stream": True, "messages": []})
    assert resp.status == 200
    first = resp.read1(65536)
    t_first = time.monotonic() - t0
    rest = resp.read()
    t_all = time.monotonic() - t0
    conn.close()
    assert b'"n": 0' in first
    assert t_first < backend.GAP, f"首块 {t_first:.2f}s 才到，网关在攒"
    assert t_all >= 2 * backend.GAP
    assert b'"n": 2' in rest
    assert backend.requests[0][0] == "/v1/chat/completions"    # 路径没被拼成 /v1/v1


def test_loading_model_gets_503_with_reason(setup):
    _, port, _ = setup
    conn, resp = _post(port, {"model": "big", "messages": []})
    body = json.loads(resp.read())
    assert resp.status == 503
    assert resp.getheader("Retry-After")
    assert body["error"]["code"] == "model_not_ready"
    assert "loading" in body["error"]["message"]


def test_unknown_model_gets_404_naming_what_exists(setup):
    _, port, _ = setup
    conn, resp = _post(port, {"model": "nope", "messages": []})
    body = json.loads(resp.read())
    assert resp.status == 404 and body["error"]["code"] == "model_not_found"
    assert "qwen" in body["error"]["message"]


def test_models_lists_only_ready_ones(setup):
    _, port, _ = setup
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("GET", "/v1/models")
    data = json.loads(conn.getresponse().read())["data"]
    assert [m["id"] for m in data] == ["qwen"]


def test_missing_model_falls_back_to_the_only_ready_one(setup):
    backend, port, _ = setup
    conn, resp = _post(port, {"messages": []})
    resp.read()
    assert resp.status == 200
    assert backend.requests[-1][1]["model"] == "qwen"


@pytest.mark.parametrize("setup", [12], indirect=True)
def test_client_disconnect_aborts_the_backend_request(setup):
    """被用户新消息抢占时 agent-core 会断开；后端必须跟着断，否则 GPU 白白生成完整段。"""
    backend, port, _ = setup
    conn, resp = _post(port, {"model": "qwen", "stream": True, "messages": []})
    resp.read1(65536)
    resp.close()          # Connection: close 时 socket 归 response 管
    conn.close()
    # 12 块要 4.8s 才发完；断开后几块之内后端就该写失败
    assert backend.aborted.wait(6 * backend.GAP)


def test_api_key_is_enforced_when_set():
    router = Router()
    router.add_provider(lambda: [])
    port = _free_port()
    gw = start_gateway(router, "127.0.0.1", port, api_key="k")
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/v1/models")
        assert conn.getresponse().status == 401
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/v1/models", headers={"Authorization": "Bearer k"})
        assert conn.getresponse().status == 200
    finally:
        gw.shutdown()


def test_a_broken_provider_does_not_take_down_routing():
    router = Router()
    router.add_provider(lambda: 1 / 0)
    router.add_provider(lambda: [{"id": "a", "state": "ready", "base_url": "http://x/v1"}])
    route, status, _ = router.resolve("a")
    assert status == 200 and route["id"] == "a"
