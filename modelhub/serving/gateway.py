"""
serving/gateway.py — 统一出口：一个端口、OpenAI 兼容、按请求里的 `model` 转发。

为什么不让 agent-core 直连各个 vLLM：vLLM 一个进程只跑一个模型，多个模型就是多个
进程多个端口；换模型、加卡片时 agent-core 的 url 都得跟着变。经过这里，url 永远是
`http://<host>:15741/v1`，变的只是 model 名。

它还负责把「还没好」说清楚：模型加载要几分钟，直连的话这段时间是 connection
refused；这里回 503 + 原因，`/v1/models` 只列已就绪的模型。

**流式必须逐块透传，不能攒。** agent-core 用「首 token 期限」判断连接卡死（见
agent-core/src/client/llm.py 的流式部分），网关只要攒一下，健康的请求就会被它判死
重试。客户端中途断开（被用户新消息抢占）时，同时断掉到后端的连接 —— vLLM 据此
abort 那个请求，不然它会把整段回答生成完，白占 GPU。
"""

from __future__ import annotations

import http.client
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from urllib.parse import urlsplit

log = logging.getLogger(__name__)

# 不往下游转的逐跳头
_HOP_HEADERS = {"connection", "keep-alive", "transfer-encoding", "content-length",
                "proxy-authenticate", "proxy-authorization", "te", "trailers", "upgrade"}
_MAX_BODY = 64 * 1024 * 1024
_BACKEND_TIMEOUT_S = 600.0


class Router:
    """model id → 后端。后端由各张卡片以 provider 函数的形式登记。

    provider() 返回 [{"id", "state", "base_url", "kind"}]；网关每个请求现问一次，
    不缓存 —— 加载/卸载随时发生，缓存只会制造「刚卸载的模型还能路由到」这种窗口。
    """

    def __init__(self):
        self._providers: list = []
        self._lock = threading.Lock()
        # 累计计数，监控页按差分画请求率 / 错误率
        self.stats = {"requests": 0, "errors_4xx": 0, "errors_5xx": 0, "aborted": 0}

    def count(self, key: str) -> None:
        with self._lock:
            self.stats[key] = self.stats.get(key, 0) + 1

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self.stats)

    def add_provider(self, provider) -> None:
        with self._lock:
            self._providers.append(provider)

    def routes(self) -> list[dict]:
        with self._lock:
            providers = list(self._providers)
        out = []
        for provider in providers:
            try:
                out.extend(provider())
            except Exception:       # noqa: BLE001 — 一张卡片坏了不能让网关全挂
                log.exception("[gateway] route provider failed")
        return out

    def resolve(self, model: str | None) -> tuple[dict | None, int, str]:
        """返回 (route, http_status, message)。route 为 None 时后两项是错误。"""
        routes = self.routes()
        if not model:
            ready = [r for r in routes if r.get("state") == "ready"]
            if len(ready) == 1:
                return ready[0], 200, ""
            return None, 400, "请求里没有 model，且就绪模型不止一个（或没有）"
        for route in routes:
            if route.get("id") == model:
                if route.get("state") == "ready":
                    return route, 200, ""
                return None, 503, f"模型 {model} 当前状态 {route.get('state')}，还不能服务"
        known = ", ".join(r["id"] for r in routes) or "无"
        return None, 404, f"本机没有模型 {model}（已登记：{known}）"


def _error_body(status: int, message: str) -> bytes:
    kind = {400: "invalid_request_error", 401: "authentication_error",
            404: "invalid_request_error", 503: "service_unavailable"}.get(status, "api_error")
    code = {404: "model_not_found", 503: "model_not_ready"}.get(status)
    return json.dumps({"error": {"message": message, "type": kind, "code": code}},
                      ensure_ascii=False).encode()


def make_handler(router: Router, api_key: str = ""):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            log.debug("[gateway] %s %s", self.address_string(), fmt % args)

        def _reply(self, status: int, body: bytes, extra: dict | None = None):
            if self.path.startswith("/v1/") and self.command == "POST":
                router.count("errors_5xx" if status >= 500 else "errors_4xx" if status >= 400 else "requests")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self) -> bool:
            if not api_key:
                return True
            return self.headers.get("Authorization", "") == f"Bearer {api_key}"

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/health":
                self._reply(200, json.dumps({"ok": True, "models": router.routes()}).encode())
            elif path == "/v1/models":
                if not self._authorized():
                    return self._reply(401, _error_body(401, "API key 不对"))
                data = [{"id": r["id"], "object": "model", "owned_by": "modelhub",
                         "kind": r.get("kind", "llm")}
                        for r in router.routes() if r.get("state") == "ready"]
                self._reply(200, json.dumps({"object": "list", "data": data}).encode())
            else:
                self._reply(404, _error_body(404, f"没有 {path}"))

        def do_POST(self):
            if not self.path.startswith("/v1/"):
                return self._reply(404, _error_body(404, f"没有 {self.path}"))
            if not self._authorized():
                return self._reply(401, _error_body(401, "API key 不对"))
            length = int(self.headers.get("Content-Length") or 0)
            if length > _MAX_BODY:
                return self._reply(413, _error_body(413, "请求体太大"))
            raw = self.rfile.read(length)
            try:
                body = json.loads(raw or b"{}")
            except ValueError:
                return self._reply(400, _error_body(400, "请求体不是 JSON"))
            route, status, message = router.resolve(body.get("model"))
            if route is None:
                extra = {"Retry-After": "10"} if status == 503 else None
                return self._reply(status, _error_body(status, message), extra)
            if not body.get("model"):
                body["model"] = route["id"]
                raw = json.dumps(body).encode()
            self._proxy(route, raw)

        def _proxy(self, route: dict, raw: bytes):
            target = urlsplit(route["base_url"])
            # base_url 以 /v1 结尾，请求路径也以 /v1/ 开头：只拼一次
            path = target.path.rstrip("/").removesuffix("/v1") + self.path
            conn = http.client.HTTPConnection(target.hostname, target.port,
                                              timeout=_BACKEND_TIMEOUT_S)
            try:
                conn.request("POST", path, body=raw, headers={
                    "Content-Type": "application/json",
                    "Accept": self.headers.get("Accept", "*/*"),
                })
                resp = conn.getresponse()
            except OSError as error:
                conn.close()
                return self._reply(502, _error_body(502, f"后端 {route['id']} 连不上：{error}"))
            router.count("requests")
            if resp.status >= 500:
                router.count("errors_5xx")
            elif resp.status >= 400:
                router.count("errors_4xx")
            try:
                self.send_response(resp.status)
                for key, value in resp.getheaders():
                    if key.lower() not in _HOP_HEADERS:
                        self.send_header(key, value)
                self.send_header("Transfer-Encoding", "chunked")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                while True:
                    chunk = resp.read1(65536)
                    if not chunk:
                        break
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                router.count("aborted")
                log.info("[gateway] client went away mid-response; aborting %s", route["id"])
            except OSError as error:
                log.warning("[gateway] backend %s broke mid-response: %s", route["id"], error)
            finally:
                conn.close()        # 关掉后端连接 = 让 vLLM abort 这个请求

    return Handler


class _Server(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def start_gateway(router: Router, host: str, port: int, api_key: str = "") -> _Server:
    server = _Server((host, port), make_handler(router, api_key))
    threading.Thread(target=server.serve_forever, daemon=True, name="gateway").start()
    log.info("[gateway] OpenAI-compatible endpoint → http://%s:%d/v1", host, port)
    return server
