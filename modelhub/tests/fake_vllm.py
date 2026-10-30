#!/usr/bin/env python3
"""假的 `vllm serve`：解析 --port，睡 FAKE_VLLM_LOAD_S 秒后在 /health 回 200。

FAKE_VLLM_CRASH=1 时打两行日志后以 3 退出，模拟加载失败。
"""
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

args = sys.argv[1:]
port = int(args[args.index("--port") + 1])
print(f"fake vllm serving {args[1]} on {port}", flush=True)
if os.environ.get("FAKE_VLLM_CRASH"):
    print("CUDA error: out of memory", flush=True)
    sys.exit(3)
time.sleep(float(os.environ.get("FAKE_VLLM_LOAD_S", "0.5")))


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == "/health" else 404)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *a):
        pass


print("Application startup complete.", flush=True)
HTTPServer(("127.0.0.1", port), H).serve_forever()
