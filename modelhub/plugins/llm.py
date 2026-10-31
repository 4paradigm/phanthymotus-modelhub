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

**选了还没下载的模型就先下载再加载**（运维页点「加载」、容器启动自动加载都是）。
下载和加载期间 info 回 `state: loading` + 一行进度 `message`：agent-core 启动画布
项目时会一直轮询 info，把这行字实时显示在启动弹窗里 —— 和 perception 卡片冷下载
模型时是同一套约定（perception/utils/model_progress.py 的措辞）。

持久化：本卡片的一切都在 <root>（默认 /models/modelhub，即主机的
/opt/embodied/models/modelhub）下 —— models/<id> 权重、state/ 选择与用户模型表、
cache/ 编译缓存。容器里不留任何要活过重建的东西。

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

from serving.chat import MAX_HISTORY_CHARS, ChatJob
from serving.downloads import DownloadManager, dir_bytes, disk_usage
from serving.bench import BenchRunner
from serving.metrics import RANGES, MetricsStore, Sampler
from serving.vllm_info import ParserChoices
from serving.registry import (MODEL_ID_RE, Registry, installed_vllm_version, is_installed,
                              model_dir, pin_from_modelscope, unsupported_reason)
from serving.vllm_engine import (DEFAULT_HOST_OVERHEAD_GB, ERROR, IDLE, LOADING, NON_KV_MARGIN_GB,
                                 READY, STOPPING, VLLMEngine, footprint_gb, kv_bytes_per_token,
                                 total_memory_gb)

log = logging.getLogger(__name__)

PROTOCOL = "motus.workbench/1"
WORKBENCH_HTML = Path(__file__).with_name("llm_workbench.html")

# 运维页能调的全部动作。info 和 stop 是工作台协议要求必须有的。
ACTIONS = ("ui", "info", "start", "stop", "models", "add_model", "remove_model",
           "download_pause", "download_resume", "download_cancel",
           "config_get", "config_set", "set_autoload",
           "chat_history", "chat_clear", "chat_start", "chat_poll", "chat_stop",
           "metrics", "events", "bench_start", "bench_status", "bench_stop", "bench_list",
           "bench_delete",
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
            "text": {"type": "string", "description": "对话：用户这一句"},
            "values": {"type": "object", "description": "启动配置"},
            "enabled": {"type": "boolean"},
            "level": {"type": "string"},
            "q": {"type": "string"},
            "range": {"type": "string", "enum": list(RANGES)},
            "id": {"type": "string"},
            "think": {"type": "boolean"},
            "max_tokens": {"type": "integer"},
            "job": {"type": "string"},
            "text_from": {"type": "integer"},
            "reasoning_from": {"type": "integer"},
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


def _mem_available_gb(meminfo: str = "/proc/meminfo") -> float:
    try:
        with open(meminfo) as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except OSError:
        pass
    return 0.0


class LLMPlugin:
    PREFIX = "llm"

    def __init__(self, cfg: dict, *, router, root: str, builtin_registry: str,
                 gateway_url: str, engine_factory=VLLMEngine,
                 downloads: DownloadManager | None = None, total_gb: float | None = None,
                 engine_version: str | None = None):
        self._cfg = cfg
        self._engine_version = (engine_version if engine_version is not None
                                else installed_vllm_version())
        root = Path(root)
        self._models_root = str(root / "models")
        self._cache_dir = str(root / "cache")
        state_dir = root / "state"
        for d in (self._models_root, self._cache_dir, state_dir):
            Path(d).mkdir(parents=True, exist_ok=True)
        self._state_path = state_dir / "llm.json"
        self._gateway_url = gateway_url.rstrip("/")
        self._registry = Registry(builtin_registry, state_dir / "models.user.yaml")
        self._state_dir = state_dir
        own_downloads = downloads is None
        self._downloads = downloads or DownloadManager(self._models_root,
                                                       state_dir / "downloads.json")
        self._engine_factory = engine_factory
        self._total_gb = total_gb if total_gb is not None else total_memory_gb()
        self._overhead_gb = float(cfg.get("host_overhead_gb", DEFAULT_HOST_OVERHEAD_GB))
        self._lock = threading.RLock()          # 守 _engine / _pending / _generation
        self._switch_lock = threading.Lock()    # 串行化 load/unload，不阻塞 info
        self._engine: VLLMEngine | None = None
        self._chat: ChatJob | None = None       # 运维页里当前的那一个对话任务
        # 「已选、还没起引擎」阶段（下载中 / 下载失败）。引擎起来后它就清空。
        self._pending: dict | None = None
        # 每次 load/unload +1。后台的「下载→加载」线程拿着自己那一代的号，号过期就
        # 放弃 —— 不然下载期间换了模型，旧的那条线程下完还会把旧模型加载上去。
        self._generation = 0
        self._chat_path = state_dir / "playground.json"
        self._bench = BenchRunner(state_dir / "benchmarks", self._gateway_url)
        self._parsers = ParserChoices(probe=bool(cfg.get("probe_parsers", True)))
        router.add_provider(self._routes)
        if own_downloads:
            self._downloads.restore(self._registry.all())   # 重启前没下完的接着下
        self._metrics = None
        if cfg.get("metrics", True):
            self._metrics = MetricsStore(str(state_dir / "metrics.db"))
            self._seen = {"engine": None, "downloads": {}}
            sampler = Sampler(self._metrics, self._current, router.snapshot)
            sampler.watchers = [self._watch_events]
            sampler.start()
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
        if hasattr(self._downloads, "close"):
            self._downloads.close()            # 下到一半的留着，重启后续传
        with self._lock:
            engine, self._engine = self._engine, None
        if engine:
            engine.stop()

    # ── 动作 ────────────────────────────────────────────────────────────────

    def _do_ui(self, args):
        return {"protocol": PROTOCOL, "html": WORKBENCH_HTML.read_text(encoding="utf-8")}

    def _do_info(self, args):
        engine = self._current()
        st = engine.status() if engine else {"state": IDLE, "model": None, "error": ""}
        state, message, model = self._card_state()
        saved = self._load_state()
        return {
            # state 用 agent-core 的启动约定：loading 时它会一直轮询，并显示 message
            "state": state, "message": message, "model": model,
            "error": message if state == "error" else "",
            "busy": False,     # 下载/加载都在后台继续，关掉运维页不影响它们
            "serving": {"base_url": f"{self._gateway_url}/v1",
                        "models": [{"id": r["id"], "state": r["state"]} for r in self._routes()]},
            "engine": st,
            "engine_version": self._engine_version,
            "memory": {"total_gb": round(self._total_gb, 1),
                       "available_gb": round(_mem_available_gb(), 1),
                       "overhead_gb": self._overhead_gb,
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
        default_budget = float(self._cfg.get("default_memory_gb", 16))
        out = []
        for spec in self._registry.all().values():
            row = spec.to_public()
            row["footprint_gb"] = footprint_gb(
                float(spec.serve.get("memory_gb") or default_budget), self._overhead_gb)
            row["installed"] = is_installed(spec, self._models_root)
            row["unsupported"] = unsupported_reason(spec, self._engine_version)
            row["download"] = jobs.get(spec.id)
            row["loaded"] = bool(current and current.model_id == spec.id)
            out.append(row)
        queue = self._downloads.queue() if hasattr(self._downloads, "queue") else []
        for row in out:
            row["disk_gb"] = round(dir_bytes(model_dir(self._models_root, row["id"])) / 1e9, 2)
            row["queue_pos"] = queue.index(row["id"]) if row["id"] in queue else None
        return {"models": out, "total_gb": round(self._total_gb, 1),
                "overhead_gb": self._overhead_gb, "engine_version": self._engine_version,
                "disk": {**disk_usage(self._models_root),
                         "models_gb": round(dir_bytes(Path(self._models_root)) / 1e9, 2)},
                "queue": queue}

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

    def _do_download_pause(self, args):
        return {"download": self._downloads.pause(self._model_arg(args))}

    def _do_download_resume(self, args):
        return {"download": self._downloads.resume(self._registry.get(self._model_arg(args)))}

    def _do_download_cancel(self, args):
        model_id = self._model_arg(args)
        job = self._downloads.cancel(model_id)
        with self._lock:                       # 正在为「加载」而下载的那个被取消了
            if self._pending and self._pending.get("model") == model_id:
                self._generation += 1
                self._pending = None
        return {"download": job}

    def _do_delete_files(self, args):
        spec = self._registry.get(self._model_arg(args))
        self._ensure_not_loaded(spec.id)
        self._downloads.remove(spec)
        return {"deleted": spec.id}

    def _do_load(self, args):
        """选定一个模型：没下载就先下载，下完自动加载。立即返回，进度看 info。"""
        spec = self._registry.get(self._model_arg(args))
        reason = unsupported_reason(spec, self._engine_version)
        if reason:
            raise ValueError(f"{spec.title}：{reason}")   # 别为一个起不来的模型下几个 GB
        state = self._load_state()
        saved = dict((state.get("configs") or {}).get(spec.id) or {})
        for key in ("memory_gb", "max_model_len"):        # 老接口：load 时直接带参数
            if args.get(key):
                saved[key] = float(args[key]) if key == "memory_gb" else int(args[key])
        serve = self._effective_config(spec, saved)
        budget = float(serve.pop("memory_gb"))
        # 先建引擎（参数不合法会在这里抛，用户立刻看到），再保存选择、在后台准备
        engine = self._engine_factory(
            model_id=spec.id, model_path=str(model_dir(self._models_root, spec.id)),
            port=int(self._cfg.get("backend_port", 15750)), serve=serve, budget_gb=budget,
            total_gb=self._total_gb, load_timeout_s=float(self._cfg.get("load_timeout_s", 1800)),
            executable=self._cfg.get("vllm_executable", "vllm"),
            overhead_gb=self._overhead_gb, cache_dir=self._cache_dir,
            weights_bytes=spec.weights_bytes)
        configs = dict(state.get("configs") or {})
        configs[spec.id] = saved
        self._save_state({**state, "model": spec.id, "autoload": True, "configs": configs})
        installed = is_installed(spec, self._models_root)
        with self._lock:
            self._generation += 1
            generation = self._generation
            self._pending = None if installed else {"model": spec.id, "title": spec.title,
                                                    "phase": "downloading", "error": ""}
        threading.Thread(target=self._prepare_and_switch, args=(spec, engine, generation),
                         daemon=True, name=f"llm-load-{spec.id}").start()
        return {"loading": spec.id, "memory_gb": budget, "download_first": not installed,
                "footprint_gb": engine.footprint_gb}

    def _do_unload(self, args):
        state = self._load_state()
        state["autoload"] = False
        self._save_state(state)
        with self._lock:
            self._generation += 1
            self._pending = None
        threading.Thread(target=self._switch_to, args=(None,), daemon=True,
                         name="llm-unload").start()
        return {"unloading": True}

    def _do_chat_history(self, args):
        doc = self._load_chat()
        with self._lock:
            job = self._chat
        running = job.poll() if job and job.state == "running" else None
        return {"messages": doc["messages"], "running": running}

    def _do_chat_clear(self, args):
        with self._lock:
            job, self._chat = self._chat, None
        if job:
            job.stop()
        self._save_chat({"messages": []})
        return {"messages": []}

    def _do_chat_start(self, args):
        """用户说一句。对话存在服务端，回复生成完由服务端追加 —— 关掉页面也不丢。"""
        engine = self._current()
        if not engine or engine.state != READY:
            raise RuntimeError("没有就绪的模型")
        text = str(args.get("text") or "").strip()
        if not text:
            raise ValueError("消息是空的")
        with self._lock:
            old = self._chat
        if old and old.state == "running":
            old.stop()                  # 一次只跑一个：新消息取消旧的
        doc = self._load_chat()
        doc["messages"].append({"role": "user", "content": text, "ts": time.time()})
        self._save_chat(doc)
        history = [{"role": m["role"], "content": m["content"]} for m in doc["messages"]
                   if m["role"] == "user" or (m["role"] == "assistant" and m.get("content"))]
        while sum(len(m["content"]) for m in history) > MAX_HISTORY_CHARS and len(history) > 1:
            history.pop(0)              # 太长就丢最早的几轮
        job = ChatJob(self._gateway_url, engine.model_id, history,
                      max_tokens=int(args.get("max_tokens") or 1024),
                      on_finish=self._chat_finished).start()
        with self._lock:
            self._chat = job
        return {"job": job.id, "model": job.model}

    def _chat_finished(self, job) -> None:
        doc = self._load_chat()
        doc["messages"].append({"role": "assistant", "content": job.text, "model": job.model,
                                "ts": time.time(), "state": job.state, "error": job.error})
        self._save_chat(doc)

    def _load_chat(self) -> dict:
        try:
            doc = json.loads(self._chat_path.read_text())
            if isinstance(doc.get("messages"), list):
                return doc
        except (OSError, ValueError):
            pass
        return {"messages": []}

    def _save_chat(self, doc: dict) -> None:
        tmp = self._chat_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(doc, ensure_ascii=False))
        os.replace(tmp, self._chat_path)

    def _chat_job(self, args) -> ChatJob:
        with self._lock:
            job = self._chat
        if not job or job.id != str(args.get("job") or ""):
            raise KeyError("这个对话任务已经不在了")
        return job

    def _do_chat_poll(self, args):
        return self._chat_job(args).poll(int(args.get("text_from") or 0),
                                         int(args.get("reasoning_from") or 0))

    def _do_chat_stop(self, args):
        job = self._chat_job(args)
        job.stop()
        return job.poll(job.poll()["text_len"], job.poll()["reasoning_len"])

    # 启动配置：部署默认 ← 模型注册表 ← 运维页里改过的，后者覆盖前者
    CONFIG_FIELDS = {
        "memory_gb": float, "max_model_len": int, "max_num_seqs": int,
        "cudagraph_capture_sizes": list, "tool_call_parser": str, "reasoning_parser": str,
        "enforce_eager": bool, "extra_args": list,
    }

    def _effective_config(self, spec, saved: dict) -> dict:
        eff = {**(self._cfg.get("serve_defaults") or {}), **spec.serve, **saved}
        eff.setdefault("memory_gb", float(self._cfg.get("default_memory_gb", 16)))
        return eff

    def _do_config_get(self, args):
        spec = self._registry.get(self._model_arg(args))
        state = self._load_state()
        saved = (state.get("configs") or {}).get(spec.id) or {}
        eff = self._effective_config(spec, saved)
        try:
            model_cfg = json.loads((model_dir(self._models_root, spec.id) / "config.json").read_text())
        except (OSError, ValueError):
            model_cfg = {}
        # 页面用这几个数随预算实时算「KV cache 放得下多少 token」，和引擎的算法一致：
        # KV = 预算 - 权重 - 运行余量（见 vllm_engine.kv_cache_bytes）
        kv = {"bytes_per_token": kv_bytes_per_token(model_cfg, eff.get("kv_cache_dtype")),
              "weights_gib": round(spec.weights_bytes / 2**30, 3), "margin_gib": NON_KV_MARGIN_GB}
        return {"model": spec.id, "config": {k: eff.get(k) for k in self.CONFIG_FIELDS},
                "kv": kv, "choices": self._parsers.get(),
                "overridden": sorted(saved), "autoload": bool(state.get("autoload")),
                "selected": state.get("model"), "footprint_gb": footprint_gb(float(eff["memory_gb"]),
                                                                           self._overhead_gb)}

    def _do_config_set(self, args):
        """保存某个模型的启动配置；下次启动（或重启）生效。传 null 恢复默认。"""
        spec = self._registry.get(self._model_arg(args))
        values = args.get("values")
        if not isinstance(values, dict):
            raise ValueError("values 必须是对象")
        state = self._load_state()
        configs = dict(state.get("configs") or {})
        saved = dict(configs.get(spec.id) or {})
        for key, value in values.items():
            kind = self.CONFIG_FIELDS.get(key)
            if kind is None:
                raise ValueError(f"不认识的配置项：{key}")
            if value in (None, ""):
                saved.pop(key, None)
                continue
            if kind is list and isinstance(value, str):
                value = [v for v in value.replace(",", " ").split() if v]
            if kind is list and key == "cudagraph_capture_sizes":
                value = [int(v) for v in value]
            if kind in (int, float):
                value = kind(value)
                if value <= 0:
                    raise ValueError(f"{key} 必须大于 0")
            if kind is bool:
                value = bool(value)
            saved[key] = value
        eff = self._effective_config(spec, saved)
        from serving.vllm_engine import build_command   # 现在就校验：内存、参数格式
        build_command("/m", spec.id, 1, {k: v for k, v in eff.items() if k != "memory_gb"},
                      float(eff["memory_gb"]), self._total_gb, overhead_gb=self._overhead_gb,
                      weights_bytes=spec.weights_bytes)
        configs[spec.id] = saved
        self._save_state({**state, "configs": configs})
        return self._do_config_get({"model": spec.id})

    def _do_set_autoload(self, args):
        state = self._load_state()
        state["autoload"] = bool(args.get("enabled"))
        self._save_state(state)
        return {"autoload": state["autoload"]}

    def _do_logs(self, args):
        engine = self._current()
        if not engine:
            return {"lines": []}
        return {"lines": engine.logs(int(args.get("lines") or 200), str(args.get("level") or ""),
                                     str(args.get("q") or ""))}

    # ── 性能测试 ────────────────────────────────────────────────────────────
    def _do_bench_start(self, args):
        engine = self._current()
        if not engine or engine.state != READY:
            raise RuntimeError("先启动一个模型再测")
        spec = self._registry.get(engine.model_id)
        saved = (self._load_state().get("configs") or {}).get(spec.id) or {}
        config = self._effective_config(spec, saved)
        config["memory_gb"] = engine.budget_gb
        return {"run": self._bench.start(spec.id, spec.title, config, self._engine_version,
                                         self._total_gb)}

    def _do_bench_status(self, args):
        return {"run": self._bench.status()}

    def _do_bench_stop(self, args):
        self._bench.stop()
        return {"run": self._bench.status()}

    def _do_bench_list(self, args):
        return {"runs": self._bench.list()}

    def _do_bench_delete(self, args):
        self._bench.delete(str(args.get("id") or ""))
        return {"runs": self._bench.list()}

    def _do_metrics(self, args):
        if not self._metrics:
            raise RuntimeError("监控没有开启")
        return self._metrics.query(str(args.get("range") or "1h"))

    def _do_events(self, args):
        if not self._metrics:
            raise RuntimeError("监控没有开启")
        return {"events": self._metrics.events(str(args.get("range") or "24h"))}

    def _watch_events(self) -> None:
        """采样时顺带记事件：引擎出错、下载失败。网关 5xx 由采样器按差分记。"""
        engine = self._current()
        st = engine.status() if engine else None
        key = (st["model"], st["state"], st.get("error")) if st else None
        if st and st["state"] == "error" and key != self._seen["engine"]:
            self._metrics.event("error", "engine", f"{st['model']}：{st.get('error') or '出错'}")
        self._seen["engine"] = key
        for model_id, job in self._downloads.jobs().items():
            if job.get("state") == "error" and self._seen["downloads"].get(model_id) != job.get("error"):
                self._metrics.event("error", "download", f"{model_id}：{job.get('error')}")
            self._seen["downloads"][model_id] = job.get("error") if job.get("state") == "error" else None

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

    def _prepare_and_switch(self, spec, engine: VLLMEngine, generation: int) -> None:
        """后台：必要时先下载，再切到新引擎。被更新的一次 load/unload 取代就放弃。"""
        if not is_installed(spec, self._models_root):
            self._downloads.start(spec)            # 有别的在下就排队
            while True:
                if not self._is_current(generation):
                    return                         # 下载本身继续，下完留在盘上
                job = self._downloads.job(spec.id) or {}
                if job.get("state") == "done":
                    break
                if job.get("state") == "error":
                    self._set_pending_error(generation, f"下载 {spec.title} 失败：{job.get('error')}")
                    return
                if job.get("state") == "cancelled":
                    return
                time.sleep(1.0)
        with self._lock:
            if self._generation != generation:
                return
            self._pending = None
        self._switch_to(engine, generation)

    def _is_current(self, generation: int) -> bool:
        with self._lock:
            return self._generation == generation

    def _set_pending_error(self, generation: int, message: str) -> None:
        log.error("[llm] %s", message)
        with self._lock:
            if self._generation == generation and self._pending:
                self._pending.update(phase="error", error=message)

    def _card_state(self) -> tuple[str, str, str | None]:
        """(state, message, model)：卡片现在在干什么，一句话说给人看。"""
        with self._lock:
            pending = dict(self._pending) if self._pending else None
            engine = self._engine
        if pending:
            if pending["phase"] == "error":
                return "error", pending["error"], pending["model"]
            job = self._downloads.job(pending["model"]) or {}
            title = pending["title"]
            if job.get("state") == "downloading":
                return ("loading", f"正在下载模型 '{title}' … {job.get('pct', 0)}% "
                        f"({job.get('mb_done', 0)}/{job.get('mb_total', 0)} MB)", pending["model"])
            if job.get("state") == "queued":
                return "loading", f"模型 '{title}' 排队等待下载 …", pending["model"]
            if job.get("state") == "paused":
                return "loading", f"模型 '{title}' 下载已暂停（{job.get('pct', 0)}%）", pending["model"]
            if job.get("state") == "verifying":
                return "loading", f"正在校验模型 '{title}' …", pending["model"]
            return "loading", f"正在准备模型 '{pending['title']}' …", pending["model"]
        if engine is None:
            return "idle", "未加载模型（在运维页选择并加载）", None
        st = engine.status()
        title = self._title(engine.model_id)
        if st["state"] == LOADING:
            return ("loading", f"正在加载模型 '{title}'（已 {st.get('load_s', 0):.0f}s）…",
                    engine.model_id)
        if st["state"] == STOPPING:
            return "loading", "正在停止上一个模型 …", engine.model_id
        if st["state"] == ERROR:
            return "error", st.get("error") or "vLLM 出错", engine.model_id
        if st["state"] == READY:
            return "ready", f"'{title}' 就绪", engine.model_id
        return "idle", "未加载模型", engine.model_id

    def _title(self, model_id: str) -> str:
        try:
            return self._registry.get(model_id).title
        except (KeyError, ValueError):
            return model_id

    def _switch_to(self, engine: VLLMEngine | None, generation: int | None = None) -> None:
        with self._switch_lock:
            with self._lock:
                if generation is not None and generation != self._generation:
                    return                 # 排队期间又换了模型
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
            result = self._do_load({"model": model_id})
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
            state = json.loads(self._state_path.read_text())
        except (OSError, ValueError):
            return {}
        # 老格式：预算和上下文在顶层 —— 挪进 configs[当前模型]
        legacy = {k: state.pop(k) for k in ("memory_gb", "max_model_len") if k in state}
        legacy = {k: v for k, v in legacy.items() if v}
        if legacy and state.get("model"):
            state.setdefault("configs", {}).setdefault(state["model"], {}).update(
                {k: v for k, v in legacy.items()
                 if k not in state["configs"][state["model"]]})
        return state

    def _save_state(self, state: dict) -> None:
        tmp = self._state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False))
        os.replace(tmp, self._state_path)

    @staticmethod
    def _error(code: str, message: str) -> dict:
        return {"error": {"code": code, "message": message}}
