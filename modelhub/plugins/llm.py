"""
plugins/llm.py — 「本地大模型」卡片：在机器人上跑 LLM，对 agent-core 提供 OpenAI 接口。

卡片本身是个运维入口，不是数据通路：
  - 数据面：agent-core 直接请求网关 `http://127.0.0.1:15741/v1`，不经过 MCP。
  - 管理面：卡片的执行按钮打开运维页（x-workbench），页面里选模型、下载、加载、
    看日志、试一句话。

**生命周期独立于画布项目。** 画布项目启停时 agent-core 会对每张卡片发 start/stop，
这里两者都只回状态、不动模型：agent-core 自己的决策调用不依赖画布，项目一停就
卸模型等于把决策本身弄挂；一次加载几分钟，也不该跟着项目反复起停。容器起来时
自动加载上次选的模型；要卸载用运维页的「卸载」。

对 LLM 只公开 `info`：模型怎么换是运维决定，不该由正在用这个模型做决策的 LLM
自己改。其余动作只在运维页里可用（x-workbench.actions）。
"""

from __future__ import annotations

import copy
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from serving.downloads import DownloadManager
from serving.registry import Registry, is_installed, model_dir, pin_from_modelscope, MODEL_ID_RE
from serving.vllm_engine import READY, VLLMEngine, total_memory_gb

log = logging.getLogger(__name__)

PROTOCOL = "motus.workbench/1"
WORKBENCH_HTML = Path(__file__).with_name("llm_workbench.html")

# 运维页能调的全部动作。info 和 stop 是工作台协议要求必须有的。
ACTIONS = ("ui", "info", "start", "stop", "models", "add_model", "remove_model",
           "download", "delete_files", "load", "unload", "logs", "test")
PUBLIC_ACTIONS = ("info",)

_ACTION_TITLES = {"ui": "打开模型运维页", "info": "查询本地大模型状态"}

TOOLS = [{
    "name": "llm",
    "type": "actuator",
    "description": ("本地大模型服务（vLLM）。在本机加载 LLM，并以 OpenAI 兼容接口对外服务；"
                    "info 返回当前模型、状态和接口地址。"),
    "inputSchema": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": list(ACTIONS)},
            "model": {"type": "string", "description": "模型 id（注册表里的那一列）"},
            "repo": {"type": "string", "description": "ModelScope 仓库，如 Qwen/Qwen3-8B-FP8"},
            "revision": {"type": "string", "description": "分支或 commit，默认 master"},
            "title": {"type": "string"},
            "memory_gb": {"type": "number", "description": "给这个模型的内存预算（GB）"},
            "max_model_len": {"type": "integer", "description": "上下文长度上限"},
            "tool_call_parser": {"type": "string"},
            "reasoning_parser": {"type": "string"},
            "prompt": {"type": "string"},
            "lines": {"type": "integer"},
            "instance_id": {"type": "string"},
            "_trace_id": {"type": "string"},
        },
        "required": ["action"],
    },
}]


def _tool_declaration() -> list[dict]:
    tools = copy.deepcopy(TOOLS)
    schema = tools[0]["inputSchema"]
    schema["properties"]["action"]["oneOf"] = [
        {"const": a, "title": _ACTION_TITLES.get(a, a)} for a in ACTIONS]
    schema["x-action-params"] = {"info": {"params": [], "description": "查询本地大模型的状态和接口地址"}}
    schema["x-resource"] = "modelhub_llm"    # 专用通道：别让决策的其他动作排在它后面
    schema["x-workbench"] = {"protocol": PROTOCOL, "kind": "settings", "title": "本地大模型",
                             "entry": "ui", "actions": list(ACTIONS)}
    return tools


class LLMPlugin:
    PREFIX = "llm"

    def __init__(self, cfg: dict, *, router, models_root: str, state_dir: str,
                 builtin_registry: str, gateway_url: str, engine_factory=VLLMEngine,
                 downloads: DownloadManager | None = None, total_gb: float | None = None):
        self._cfg = cfg
        self._models_root = models_root
        self._state_path = Path(state_dir) / "llm.json"
        self._gateway_url = gateway_url.rstrip("/")
        self._registry = Registry(builtin_registry, Path(state_dir) / "models.user.yaml")
        self._downloads = downloads or DownloadManager(models_root)
        self._engine_factory = engine_factory
        self._total_gb = total_gb if total_gb is not None else total_memory_gb()
        self._lock = threading.RLock()          # 守 self._engine 的替换
        self._switch_lock = threading.Lock()    # 串行化 load/unload，不阻塞 info
        self._engine: VLLMEngine | None = None
        router.add_provider(self._routes)
        Path(state_dir).mkdir(parents=True, exist_ok=True)
        threading.Thread(target=self._autoload, daemon=True, name="llm-autoload").start()

    # ── MCP ─────────────────────────────────────────────────────────────────

    def get_tools(self) -> list[dict]:
        return _tool_declaration()

    def dispatch(self, name: str, args: dict) -> dict | None:
        if name != self.PREFIX:
            return None
        action = str(args.get("action") or "info")
        handler = getattr(self, f"_do_{action}", None)
        if action not in ACTIONS or handler is None:
            return self._error("invalid_argument", f"未知动作：{action}")
        try:
            return handler(args)
        except (ValueError, KeyError, RuntimeError) as error:
            msg = error.args[0] if isinstance(error, KeyError) and error.args else str(error)
            return self._error("invalid_argument", msg)

    def shutdown(self) -> None:
        with self._lock:
            engine, self._engine = self._engine, None
        if engine:
            engine.stop()

    # ── 动作 ────────────────────────────────────────────────────────────────

    def _do_ui(self, args):
        return {"protocol": PROTOCOL, "html": WORKBENCH_HTML.read_text(encoding="utf-8")}

    def _do_info(self, args):
        engine = self._current()
        st = engine.status() if engine else {"state": "idle", "model": None, "error": ""}
        saved = self._load_state()
        return {
            "state": st["state"], "model": st.get("model"), "error": st.get("error", ""),
            "busy": False,     # 下载/加载都在后台继续，关掉运维页不影响它们
            "serving": {"base_url": f"{self._gateway_url}/v1",
                        "models": [{"id": r["id"], "state": r["state"]} for r in self._routes()]},
            "engine": st,
            "memory": {"total_gb": round(self._total_gb, 1),
                       "default_budget_gb": self._cfg.get("default_memory_gb", 16)},
            "selected": saved,
            "download": self._downloads.job(self._downloads.active) if self._downloads.active else None,
        }

    def _do_start(self, args):
        return self._do_info(args)   # 画布项目启动：模型服务不随项目起停，见模块说明

    def _do_stop(self, args):
        return self._do_info(args)   # 同上；卸载请用 unload

    def _do_models(self, args):
        jobs = self._downloads.jobs()
        current = self._current()
        out = []
        for spec in self._registry.all().values():
            row = spec.to_public()
            row["installed"] = is_installed(spec, self._models_root)
            row["download"] = jobs.get(spec.id)
            row["loaded"] = bool(current and current.model_id == spec.id)
            out.append(row)
        return {"models": out, "total_gb": round(self._total_gb, 1)}

    def _do_add_model(self, args):
        repo = str(args.get("repo") or "").strip()
        model_id = str(args.get("model") or repo.rsplit("/", 1)[-1]).strip().lower()
        if not repo:
            raise ValueError("缺 repo（ModelScope 仓库名）")
        if not MODEL_ID_RE.match(model_id):
            raise ValueError(f"模型 id 不合法：{model_id!r}")
        entry = {"title": str(args.get("title") or repo.rsplit("/", 1)[-1]), "kind": "llm",
                 **pin_from_modelscope(repo, str(args.get("revision") or "master"))}
        serve = {k: args[k] for k in ("max_model_len", "tool_call_parser", "reasoning_parser",
                                     "memory_gb") if args.get(k) not in (None, "")}
        if serve:
            entry["serve"] = serve
        spec = self._registry.add_user_model(model_id, entry)
        return {"added": spec.to_public()}

    def _do_remove_model(self, args):
        spec = self._registry.get(self._model_arg(args))
        if spec.builtin:
            raise ValueError("内置模型不能从列表里删；只能删它的文件（delete_files）")
        self._ensure_not_loaded(spec.id)
        self._downloads.remove(spec)
        self._registry.remove_user_model(spec.id)
        return {"removed": spec.id}

    def _do_download(self, args):
        spec = self._registry.get(self._model_arg(args))
        return {"download": self._downloads.start(spec)}

    def _do_delete_files(self, args):
        spec = self._registry.get(self._model_arg(args))
        self._ensure_not_loaded(spec.id)
        self._downloads.remove(spec)
        return {"deleted": spec.id}

    def _do_load(self, args):
        spec = self._registry.get(self._model_arg(args))
        if not is_installed(spec, self._models_root):
            raise ValueError(f"{spec.id} 还没下载")
        budget = float(args.get("memory_gb") or spec.serve.get("memory_gb")
                       or self._cfg.get("default_memory_gb", 16))
        serve = dict(spec.serve)
        if args.get("max_model_len"):
            serve["max_model_len"] = int(args["max_model_len"])
        # 先建引擎（参数不合法会在这里抛），再保存选择、在后台切换
        engine = self._engine_factory(
            model_id=spec.id, model_path=str(model_dir(self._models_root, spec.id)),
            port=int(self._cfg.get("backend_port", 15750)), serve=serve, budget_gb=budget,
            total_gb=self._total_gb, load_timeout_s=float(self._cfg.get("load_timeout_s", 1800)),
            executable=self._cfg.get("vllm_executable", "vllm"))
        self._save_state({"model": spec.id, "memory_gb": budget,
                          "max_model_len": serve.get("max_model_len"), "autoload": True})
        threading.Thread(target=self._switch_to, args=(engine,), daemon=True,
                         name=f"llm-load-{spec.id}").start()
        return {"loading": spec.id, "memory_gb": budget}

    def _do_unload(self, args):
        state = self._load_state()
        state["autoload"] = False
        self._save_state(state)
        threading.Thread(target=self._switch_to, args=(None,), daemon=True,
                         name="llm-unload").start()
        return {"unloading": True}

    def _do_logs(self, args):
        engine = self._current()
        return {"lines": engine.logs(int(args.get("lines") or 200)) if engine else []}

    def _do_test(self, args):
        """经网关发一句话，回首 token 延迟和吞吐 —— 走的就是 agent-core 走的那条路。

        运维页经 agent-core 的 MCP 代理调过来，代理有 20s 超时，所以这里限 16s、
        最多 128 token。
        """
        engine = self._current()
        if not engine or engine.state != READY:
            raise RuntimeError("没有就绪的模型")
        prompt = str(args.get("prompt") or "用一句话介绍你自己。")
        body = json.dumps({
            "model": engine.model_id, "stream": True, "max_tokens": 128,
            "messages": [{"role": "user", "content": prompt}],
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode()
        req = urllib.request.Request(f"{self._gateway_url}/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        t0 = time.monotonic()
        first = None
        text, chunks = [], 0
        try:
            with urllib.request.urlopen(req, timeout=16) as resp:
                for raw in resp:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:") or line == "data: [DONE]":
                        continue
                    delta = (json.loads(line[5:]).get("choices") or [{}])[0].get("delta") or {}
                    piece = delta.get("content") or delta.get("reasoning_content") or ""
                    if piece:
                        first = first or time.monotonic()
                        text.append(piece)
                        chunks += 1
                    if time.monotonic() - t0 > 15:
                        break
        except (urllib.error.URLError, OSError, ValueError) as error:
            raise RuntimeError(f"测试请求失败：{error}") from error
        end = time.monotonic()
        gen_s = end - (first or end)
        return {"model": engine.model_id, "reply": "".join(text),
                "ttft_ms": round(((first or end) - t0) * 1000),
                "chunks": chunks, "chunks_per_s": round(chunks / gen_s, 1) if gen_s > 0 else None}

    # ── 内部 ────────────────────────────────────────────────────────────────

    def _routes(self) -> list[dict]:
        engine = self._current()
        if not engine:
            return []
        return [{"id": engine.model_id, "state": engine.state, "base_url": engine.base_url,
                 "kind": "llm"}]

    def _current(self) -> VLLMEngine | None:
        with self._lock:
            return self._engine

    def _switch_to(self, engine: VLLMEngine | None) -> None:
        with self._switch_lock:
            with self._lock:
                old, self._engine = self._engine, engine
            if old:
                old.stop()             # 统一内存装不下两份：先停旧的再起新的
            if engine:
                engine.start()

    def _autoload(self) -> None:
        state = self._load_state()
        model_id = state.get("model")
        if not state.get("autoload") or not model_id:
            return
        try:
            args = {"model": model_id, "memory_gb": state.get("memory_gb"),
                    "max_model_len": state.get("max_model_len")}
            result = self._do_load(args)
            log.info("[llm] autoload: %s", result)
        except Exception:          # noqa: BLE001 — 自动加载失败不能拖垮整个服务
            log.exception("[llm] autoload of %s failed", model_id)

    def _model_arg(self, args) -> str:
        model_id = str(args.get("model") or "").strip()
        if not model_id:
            raise ValueError("缺 model")
        return model_id

    def _ensure_not_loaded(self, model_id: str) -> None:
        engine = self._current()
        if engine and engine.model_id == model_id:
            raise RuntimeError(f"{model_id} 正在使用，先卸载")

    def _load_state(self) -> dict:
        try:
            return json.loads(self._state_path.read_text())
        except (OSError, ValueError):
            return {}

    def _save_state(self, state: dict) -> None:
        tmp = self._state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False))
        os.replace(tmp, self._state_path)

    @staticmethod
    def _error(code: str, message: str) -> dict:
        return {"error": {"code": code, "message": message}}
