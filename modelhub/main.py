#!/usr/bin/env python3
"""
modelhub/main.py — ModelHub bundle 统一入口。

ModelHub 是机器人本机的大模型服务层，和 perception / actucore 并列：
  - 管理面：卡片（插件）聚合成一个 MCP HTTP server（默认 15740），agent-core 通过
    MCP JSON-RPC 调用，画布上的卡片打开运维页。
  - 数据面：统一出口网关（默认 15741），OpenAI 兼容，按 `model` 转发到各卡片
    起的推理引擎。agent-core 的 LLM 客户端直接请求它，不经过 MCP。

骨架照 phanthymotus/actucore/main.py：显式注册卡片、注册心跳、一张卡片的 schema
坏了不拖垮其余。不同的是没有 ROS —— 这一层不收发 topic。

MCP 工具命名规则：{plugin_prefix}_{tool_name}；tool 名等于 PREFIX 时就是 PREFIX 本身。
"""

from __future__ import annotations

# 先于一切输出：每行日志一次原子写，并发写不会撕裂 Docker 日志记录。
# 和 perception/actucore 同一个模块，镜像构建时从 phanthymotus 拷进来。
try:
    import logsafe
    logsafe.install()
except ImportError:      # 本地跑测试时没有它
    pass

import json
import logging
import os
import signal
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn

import yaml

from serving.gateway import Router, start_gateway

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger(__name__)
for _quiet in ('urllib3', 'httpcore', 'httpx'):
    logging.getLogger(_quiet).setLevel(logging.WARNING)

_LOG_ARG_CHARS = 500
REGISTER_ALIVE_INTERVAL_S = 1800.0
HERE = Path(__file__).resolve().parent


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def _brief(obj) -> str:
    text = repr(obj)
    if len(text) <= _LOG_ARG_CHARS:
        return text
    return f"{text[:_LOG_ARG_CHARS]}…[+{len(text) - _LOG_ARG_CHARS} chars]"


def _load_config() -> dict:
    config_path = os.environ.get("CONFIG_PATH", str(HERE / "config.yaml"))
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


# ── Bundle ────────────────────────────────────────────────────────────────────

class ModelHubBundle:
    def __init__(self, cfg: dict, router: Router):
        self._plugins: list = []
        plugins_cfg = cfg.get("plugins") or {}
        gw = cfg.get("gateway") or {}
        gateway_url = f"http://127.0.0.1:{int(gw.get('port', 15741))}"
        common = dict(router=router,
                      root=cfg.get("root", "/models/modelhub"),
                      builtin_registry=str(HERE / "models.yaml"),
                      gateway_url=gateway_url)

        # ── 卡片注册区 ────────────────────────────────────────────────────
        # 显式注册，每张卡片一个 if 块。以后的 VLM 卡片复用 serving/ 下的引擎、
        # 下载和网关，只是注册表里 kind 不同、插件声明不同。
        # ──────────────────────────────────────────────────────────────────

        if plugins_cfg.get("llm", {}).get("enabled", False):
            try:
                from plugins.llm import LLMPlugin
                self._plugins.append(LLMPlugin(plugins_cfg["llm"], **common))
                log.info("LLMPlugin loaded")
            except Exception:       # noqa: BLE001 — 一张卡片起不来，其余照常
                log.error("LLMPlugin failed to load", exc_info=True)

        if not self._plugins:
            log.info("no cards enabled — ModelHub is running as an empty MCP host")

    def get_all_tools(self) -> list:
        tools = []
        for p in self._plugins:
            try:
                built = list(p.get_tools())
            except Exception:       # noqa: BLE001 — 一张卡片的 schema，不是整个 bundle 的
                log.exception("card %s failed to build its schema; serving the rest "
                              "without it", getattr(p, "PREFIX", p))
                continue
            for t in built:
                full_name = t['name'] if t['name'] == p.PREFIX else f"{p.PREFIX}_{t['name']}"
                tools.append({**t, "name": full_name})
        return tools

    def dispatch(self, full_name: str, args: dict) -> dict | None:
        prefix, sep, tool_name = full_name.partition("_")
        name = tool_name if sep else prefix
        for p in self._plugins:
            if p.PREFIX == prefix:
                return p.dispatch(name, args)
        return None

    def shutdown(self) -> None:
        for p in self._plugins:
            stop = getattr(p, "shutdown", None)
            if stop:
                try:
                    stop()
                except Exception:   # noqa: BLE001
                    log.exception("card %s failed to shut down", p.PREFIX)


# ── MCP HTTP server ───────────────────────────────────────────────────────────

_bundle: ModelHubBundle | None = None


def make_handler():
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            log.debug(f"{self.address_string()} {fmt % args}")

        def _send(self, status: int, body: str):
            encoded = body.encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Accept")
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self):
            self.send_response(404)
            self.end_headers()

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Accept")
            self.end_headers()

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length)
            try:
                rpc = json.loads(raw)
            except Exception as e:
                self._send(200, json.dumps({"jsonrpc": "2.0", "id": None,
                                            "error": {"code": -32700, "message": f"Parse error: {e}"}}))
                return

            rid    = rpc.get("id")
            method = rpc.get("method", "")
            params = rpc.get("params") or {}
            if rid is None:
                self.send_response(202); self.end_headers(); return

            def ok(result):
                self._send(200, json.dumps({"jsonrpc": "2.0", "id": rid, "result": result}))

            def err(code, msg):
                self._send(200, json.dumps({"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": msg}}))

            try:
                if method == "initialize":
                    ok({"protocolVersion": "2024-11-05", "capabilities": {"tools": {}},
                        "serverInfo": {"name": "modelhub-bundle", "version": "1.0.0"}})
                elif method == "tools/list":
                    ok({"tools": _bundle.get_all_tools()})
                elif method == "tools/call":
                    name = params.get("name", "")
                    args = params.get("arguments") or {}
                    # info 是心跳探测、ui 回的是整页 HTML —— 都不进 INFO 日志
                    # chat_poll：运维页对话时每 250 ms 一次
                    quiet = args.get('action') in ('info', 'models', 'logs', 'chat_poll',
                                                   'bench_status', 'metrics', 'events')
                    if not quiet:
                        log.info(f"[mcp] tools/call: {name}({_brief(args)})")
                    result = _bundle.dispatch(name, args)
                    if result is None:
                        err(-32601, f"Unknown tool: {name}")
                    else:
                        if not quiet and args.get('action') != 'ui':
                            log.info(f"[mcp] tools/call result: {json.dumps(result, ensure_ascii=False)[:200]}")
                        ok({"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}]})
                else:
                    err(-32601, f"Method not found: {method}")
            except BrokenPipeError:
                log.debug("Client disconnected before response")
            except Exception as e:
                log.error(f"RPC error: {e}", exc_info=True)
                try:
                    err(-32603, str(e))
                except BrokenPipeError:
                    pass

    return Handler


# ── Entry point ───────────────────────────────────────────────────────────────

def _start_registration(mcp_port: int, name: str, category: str):
    """向 agent-core 注册，此后每 30s 心跳一次。和 actucore 同一套：只记状态变化，外加慢速存活行。"""
    import ssl as _ssl
    import time as _t
    import urllib.request as _urllib
    agent_core_url = os.environ.get("AGENT_CORE_URL", "https://localhost:15678")
    _ctx = _ssl.create_default_context()
    _ctx.check_hostname = False
    _ctx.verify_mode = _ssl.CERT_NONE
    payload = json.dumps({"name": name, "url": f"http://localhost:{mcp_port}/mcp",
                          "category": category}).encode()

    def _run():
        healthy = None
        last_alive = 0.0
        while True:
            try:
                req = _urllib.Request(f"{agent_core_url}/api/mcp", data=payload,
                                      headers={"Content-Type": "application/json"}, method="POST")
                with _urllib.urlopen(req, timeout=3, context=_ctx):
                    now = _t.monotonic()
                    if healthy is not True:
                        log.info(f"[register] heartbeat ok → {agent_core_url}"
                                 + ("" if healthy is None else " (recovered)"))
                        healthy = True
                        last_alive = now
                    elif now - last_alive >= REGISTER_ALIVE_INTERVAL_S:
                        last_alive = now
                        log.info(f"[register] still registered → {agent_core_url}")
                _t.sleep(30)
            except Exception as e:
                log.warning(f"[register] failed: {e}, retrying in 5s")
                healthy = False
                _t.sleep(5)
    threading.Thread(target=_run, daemon=True, name="register").start()


def main():
    global _bundle

    cfg      = _load_config()
    mcp_port = int(cfg.get("mcp_port", 15740))
    gw       = cfg.get("gateway") or {}

    plugins_cfg = cfg.get("plugins") or {}
    enabled = [k for k, v in plugins_cfg.items() if isinstance(v, dict) and v.get("enabled")]
    log.info(f"modelhub bundle starting, mcp_port={mcp_port}")
    log.info(f"config: cards enabled={enabled or '(none)'}")

    router  = Router()
    gateway = start_gateway(router, gw.get("host", "127.0.0.1"), int(gw.get("port", 15741)),
                            str(gw.get("api_key") or ""))
    _bundle = ModelHubBundle(cfg, router)

    _start_registration(mcp_port, "ModelHub", "modelhub")

    server = ThreadingHTTPServer(("", mcp_port), make_handler())
    log.info(f"MCP server → http://0.0.0.0:{mcp_port}")

    def _shutdown(signum, frame):
        log.info(f"signal {signum}, shutting down")
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    try:
        server.serve_forever()
    finally:
        # 先停推理引擎：vLLM 的 worker 占着几十 GB，容器退出前不收掉会被 SIGKILL 硬杀
        _bundle.shutdown()
        gateway.shutdown()


if __name__ == "__main__":
    main()
