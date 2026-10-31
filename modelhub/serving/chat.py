"""
serving/chat.py — 运维页里的对话：后台流式请求网关，页面轮询取增量。

运维页和卡片之间只有 agent-core 的 MCP 代理（每次调用一问一答、20 s 超时），没法
直接流式。所以 `start` 起一个后台线程向网关发 `stream: true` 的请求，把 token 攒在
这里；页面每隔几百毫秒用 `poll` 取「从某个位置往后」的增量，读起来就是逐字出现。

`stop` 关掉到网关的连接 —— 网关再关掉到 vLLM 的，vLLM 据此 abort 这次生成，不会
在没人看的时候把整段生成完。同一时间只保留一个对话任务：新的一开始，旧的就被取消。
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
import uuid

MAX_HISTORY_CHARS = 30_000     # 宿主对每条工作台请求限 40,000 字符
MAX_TOKENS_CAP = 4096
_ROLES = {"system", "user", "assistant"}


def validate_messages(messages) -> list[dict]:
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages 必须是非空列表")
    out, total = [], 0
    for m in messages:
        if not isinstance(m, dict) or m.get("role") not in _ROLES \
                or not isinstance(m.get("content"), str):
            raise ValueError("每条消息要有 role（system/user/assistant）和字符串 content")
        total += len(m["content"])
        out.append({"role": m["role"], "content": m["content"]})
    if total > MAX_HISTORY_CHARS:
        raise ValueError(f"对话太长（{total} 字符），请新开一个对话")
    if out[-1]["role"] != "user":
        raise ValueError("最后一条消息必须是用户消息")
    return out


class ChatJob:
    def __init__(self, gateway_url: str, model: str, messages: list[dict], *,
                 max_tokens: int = 1024, think: bool = False, opener=urllib.request.urlopen):
        self.id = uuid.uuid4().hex[:12]
        self.model = model
        self._url = f"{gateway_url.rstrip('/')}/v1/chat/completions"
        self._body = {
            "model": model, "stream": True, "messages": messages,
            "max_tokens": max(16, min(int(max_tokens), MAX_TOKENS_CAP)),
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": bool(think)},
        }
        self._opener = opener
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._resp = None
        self.text = ""
        self.reasoning = ""
        self.state = "running"          # running / done / stopped / error
        self.error = ""
        self.t0 = time.monotonic()
        self.t_first = None
        self.t_end = None
        self.chunks = 0
        self.completion_tokens = None
        self.prompt_tokens = None

    def start(self) -> "ChatJob":
        threading.Thread(target=self._run, daemon=True, name=f"chat-{self.id}").start()
        return self

    def stop(self) -> None:
        self._cancel.set()
        with self._lock:
            resp = self._resp
        if resp is not None:
            try:
                resp.close()            # 断开 → 网关断开后端 → vLLM abort
            except Exception:           # noqa: BLE001
                pass

    def poll(self, text_from: int = 0, reasoning_from: int = 0) -> dict:
        with self._lock:
            end = self.t_end or time.monotonic()
            gen_s = end - self.t_first if self.t_first else 0
            tokens = self.completion_tokens or self.chunks
            return {
                "job": self.id, "model": self.model, "state": self.state, "error": self.error,
                "text": self.text[max(0, int(text_from)):], "text_len": len(self.text),
                "reasoning": self.reasoning[max(0, int(reasoning_from)):],
                "reasoning_len": len(self.reasoning),
                "stats": {
                    "ttft_ms": round((self.t_first - self.t0) * 1000) if self.t_first else None,
                    "tokens": tokens, "prompt_tokens": self.prompt_tokens,
                    "tok_per_s": round((tokens - 1) / gen_s, 1) if gen_s > 0 and tokens > 1 else None,
                    "elapsed_s": round(end - self.t0, 1),
                },
            }

    def _finish(self, state: str, error: str = "") -> None:
        with self._lock:
            if self.state == "running":
                self.state, self.error = state, error
            self.t_end = self.t_end or time.monotonic()
            self._resp = None

    def _run(self) -> None:
        req = urllib.request.Request(self._url, json.dumps(self._body).encode(),
                                     {"Content-Type": "application/json"})
        try:
            resp = self._opener(req, timeout=300)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            try:
                detail = json.loads(detail)["error"]["message"]
            except (ValueError, KeyError, TypeError):
                pass
            return self._finish("error", f"网关返回 {e.code}：{detail}")
        except (urllib.error.URLError, OSError) as e:
            return self._finish("error", f"连不上网关：{e}")
        with self._lock:
            self._resp = resp
        try:
            for raw in resp:
                if self._cancel.is_set():
                    break
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                d = json.loads(line[5:])
                usage = d.get("usage")
                delta = ((d.get("choices") or [{}])[0].get("delta") or {})
                with self._lock:
                    if usage:
                        self.completion_tokens = usage.get("completion_tokens")
                        self.prompt_tokens = usage.get("prompt_tokens")
                    piece_r = delta.get("reasoning_content") or delta.get("reasoning") or ""
                    piece = delta.get("content") or ""
                    if piece or piece_r:
                        self.t_first = self.t_first or time.monotonic()
                        self.chunks += 1
                        self.reasoning += piece_r
                        self.text += piece
        except (OSError, ValueError, AttributeError) as e:
            if not self._cancel.is_set():
                return self._finish("error", f"生成中断：{e}")
        finally:
            try:
                resp.close()
            except Exception:           # noqa: BLE001
                pass
        self._finish("stopped" if self._cancel.is_set() else "done")
