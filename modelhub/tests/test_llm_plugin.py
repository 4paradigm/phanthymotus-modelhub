"""LLM 卡片：对 agent-core 的声明契约、生命周期不随画布项目、加载与持久化。"""

import json
import threading
import time
from pathlib import Path

import pytest

from plugins.llm import ACTIONS, LLMPlugin
from serving.gateway import Router
from serving.registry import mark_installed, model_dir

BUILTIN = Path(__file__).resolve().parents[1] / "models.yaml"
# agent-core/src/api/mcp_manage.py 注册时只保留这些顶层字段，其余静默丢弃
AGENT_CORE_TOP_LEVEL = {"name", "description", "type", "multiInstance", "inputSchema",
                        "configSchema", "topic_out", "topic_in"}


class _FakeEngine:
    instances = []

    def __init__(self, *, model_id, model_path, port, serve, budget_gb, total_gb,
                 load_timeout_s, executable, overhead_gb, cache_dir, weights_bytes):
        self.model_id, self.serve, self.budget_gb = model_id, serve, budget_gb
        self.model_path, self.cache_dir = model_path, cache_dir
        self.footprint_gb = budget_gb + overhead_gb
        self.base_url = f"http://127.0.0.1:{port}/v1"
        self.state = "idle"
        self.stopped = threading.Event()
        _FakeEngine.instances.append(self)

    def start(self):
        self.state = "ready"

    def stop(self):
        self.state = "idle"
        self.stopped.set()

    def status(self):
        return {"model": self.model_id, "state": self.state, "error": "",
                "budget_gb": self.budget_gb, "load_s": 0}

    def logs(self, n):
        return ["line"]


class _FakeDownloads:
    """下载由测试手动推进：progress() 改进度，finish() 写文件并打安装标记。"""

    def __init__(self, models_root):
        self.models_root = models_root
        self.active = None
        self._jobs = {}
        self.started = []

    def start(self, spec):
        if self.active and self.active != spec.id:
            raise RuntimeError(f"正在下载 {self.active}，等它完成再下一个")
        self.active = spec.id
        self.started.append(spec.id)
        self._jobs[spec.id] = {"model": spec.id, "state": "downloading", "pct": 0,
                               "mb_done": 0, "mb_total": round(spec.size_bytes / 2**20)}
        self._specs = getattr(self, "_specs", {})
        self._specs[spec.id] = spec

    def progress(self, model_id, pct):
        self._jobs[model_id].update(pct=pct, mb_done=self._jobs[model_id]["mb_total"] * pct // 100)

    def finish(self, model_id, ok=True):
        if ok:
            _write_files(Path(self.models_root), self._specs[model_id])
            self._jobs[model_id].update(state="done", pct=100)
        else:
            self._jobs[model_id].update(state="error", error="HTTP 404")
        self.active = None

    def job(self, model_id):
        job = self._jobs.get(model_id)
        return dict(job) if job else None

    def jobs(self):
        return {k: dict(v) for k, v in self._jobs.items()}


CFG = {"default_memory_gb": 8, "backend_port": 15750, "host_overhead_gb": 4.4, "metrics": False,
       "probe_parsers": False,
       "serve_defaults": {"max_num_seqs": 8, "cudagraph_capture_sizes": [1, 2, 4, 8]}}


@pytest.fixture()
def make(tmp_path):
    _FakeEngine.instances.clear()
    downloads = _FakeDownloads(str(tmp_path / "root" / "models"))

    def build(engine_version=""):
        router = Router()
        plugin = LLMPlugin(CFG, router=router, root=str(tmp_path / "root"),
                           builtin_registry=str(BUILTIN), gateway_url="http://127.0.0.1:15741",
                           engine_factory=_FakeEngine, downloads=downloads, total_gb=59.0,
                           engine_version=engine_version)
        plugin.fake_downloads = downloads
        return plugin, router
    return build


def _write_files(models_root: Path, spec):
    d = model_dir(models_root, spec.id)
    d.mkdir(parents=True, exist_ok=True)
    for name, meta in spec.files.items():
        (d / name).parent.mkdir(parents=True, exist_ok=True)
        with open(d / name, "wb") as f:
            f.truncate(meta["size"])
    mark_installed(spec, models_root)


def _install(tmp_path, plugin, model_id):
    _write_files(tmp_path / "root" / "models", plugin._registry.get(model_id))


def _wait(pred, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end and not pred():
        time.sleep(0.02)
    return pred()


def test_declaration_meets_agent_core_and_workbench_contract(make):
    plugin, _ = make()
    [tool] = plugin.get_tools()
    assert set(tool) <= AGENT_CORE_TOP_LEVEL         # x-* 都在 inputSchema 里
    assert tool["type"] == "actuator"
    schema = tool["inputSchema"]
    wb = schema["x-workbench"]
    assert wb["protocol"] == "motus.workbench/1" and wb["entry"] == "ui"
    assert {"info", "stop", "ui"} <= set(wb["actions"])   # workbench.js 要求 info 和 stop
    assert set(wb["actions"]) == set(ACTIONS) == set(schema["properties"]["action"]["enum"])
    assert list(schema["x-action-params"]) == ["info"]  # LLM 不能自己换模型


def test_workbench_page_fits_the_host_limit(make):
    plugin, _ = make()
    page = plugin.dispatch("llm", {"action": "ui"})
    assert page["protocol"] == "motus.workbench/1"
    assert len(page["html"]) <= 250_000                  # workbench.js 的上限
    assert "motus.workbench/1" in page["html"]
    assert "http://" not in page["html"] and "https://" not in page["html"]  # 宿主 CSP 禁外链
    # 宿主 iframe 只有 sandbox="allow-scripts"：表单提交被拦，连 submit 事件都没有。
    # 第一版对话就是这么发不出消息的。
    assert "<form" not in page["html"] and "requestSubmit" not in page["html"]


def test_project_start_and_stop_do_not_touch_the_model(make, tmp_path):
    plugin, _ = make()
    _install(tmp_path, plugin, "qwen3-0.6b")
    plugin.dispatch("llm", {"action": "load", "model": "qwen3-0.6b"})
    assert _wait(lambda: plugin.dispatch("llm", {"action": "info"})["state"] == "ready")
    for action in ("stop", "start", "stop"):
        assert plugin.dispatch("llm", {"action": action, "instance_id": "c1"})["state"] == "ready"
    assert not _FakeEngine.instances[0].stopped.is_set()


def test_info_reports_serving_for_agent_core_discovery(make, tmp_path):
    plugin, router = make()
    _install(tmp_path, plugin, "qwen3-0.6b")
    plugin.dispatch("llm", {"action": "load", "model": "qwen3-0.6b", "memory_gb": 6})
    assert _wait(lambda: router.resolve("qwen3-0.6b")[1] == 200)
    info = plugin.dispatch("llm", {"action": "info"})
    assert info["busy"] is False
    assert info["serving"] == {"base_url": "http://127.0.0.1:15741/v1",
                               "models": [{"id": "qwen3-0.6b", "state": "ready"}]}


def test_loading_a_model_not_on_disk_downloads_it_first_with_progress(make):
    """卡片准备时要看得到下载进度：info 的 loading + message 就是 agent-core
    启动弹窗里显示的那一行。"""
    plugin, router = make()
    r = plugin.dispatch("llm", {"action": "load", "model": "qwen3-0.6b"})
    assert r["download_first"] is True
    dl = plugin.fake_downloads
    assert _wait(lambda: dl.started == ["qwen3-0.6b"])
    dl.progress("qwen3-0.6b", 40)
    info = plugin.dispatch("llm", {"action": "info"})
    assert info["state"] == "loading"
    assert info["message"].startswith("正在下载模型 'Qwen3 0.6B' … 40% (")
    assert plugin.dispatch("llm", {"action": "start", "instance_id": "c"})["state"] == "loading"
    assert not _FakeEngine.instances[0].state == "ready"       # 没下完不起引擎
    dl.finish("qwen3-0.6b")
    assert _wait(lambda: router.resolve("qwen3-0.6b")[1] == 200)
    assert plugin.dispatch("llm", {"action": "info"})["state"] == "ready"


def test_a_failed_download_is_an_error_the_card_reports(make):
    plugin, _ = make()
    plugin.dispatch("llm", {"action": "load", "model": "qwen3-0.6b"})
    assert _wait(lambda: plugin.fake_downloads.started)
    plugin.fake_downloads.finish("qwen3-0.6b", ok=False)
    assert _wait(lambda: plugin.dispatch("llm", {"action": "info"})["state"] == "error")
    assert "HTTP 404" in plugin.dispatch("llm", {"action": "info"})["message"]


def test_switching_models_mid_download_never_loads_the_old_one(make, tmp_path):
    plugin, router = make()
    plugin.dispatch("llm", {"action": "load", "model": "qwen3-8b-fp8"})
    assert _wait(lambda: plugin.fake_downloads.started)
    _install(tmp_path, plugin, "qwen3-0.6b")
    plugin.dispatch("llm", {"action": "load", "model": "qwen3-0.6b"})
    assert _wait(lambda: router.resolve("qwen3-0.6b")[1] == 200)
    plugin.fake_downloads.finish("qwen3-8b-fp8")       # 旧的下载下完了
    time.sleep(1.5)
    assert plugin.dispatch("llm", {"action": "info"})["model"] == "qwen3-0.6b"
    assert [e.model_id for e in _FakeEngine.instances if e.state == "ready"] == ["qwen3-0.6b"]


def test_unknown_model_is_rejected(make):
    plugin, _ = make()
    assert plugin.dispatch("llm", {"action": "load", "model": "nope"})["error"]


def test_everything_persistent_lives_under_the_one_root(make, tmp_path):
    """权重、选择、用户模型表、编译缓存都在 <root> 下 —— 也就是挂进来的卷上。"""
    plugin, _ = make()
    _install(tmp_path, plugin, "qwen3-0.6b")
    plugin.dispatch("llm", {"action": "load", "model": "qwen3-0.6b"})
    assert _wait(lambda: _FakeEngine.instances)
    root = tmp_path / "root"
    eng = _FakeEngine.instances[-1]
    assert eng.model_path == str(root / "models" / "qwen3-0.6b")
    assert eng.cache_dir == str(root / "cache")
    assert (root / "state" / "llm.json").is_file()
    assert eng.serve["cudagraph_capture_sizes"] == [1, 2, 4, 8]   # 部署默认垫底
    assert eng.serve["tool_call_parser"] == "hermes"              # 模型自己的参数在上


def test_switching_stops_the_old_engine_first_and_autoload_restores(make, tmp_path):
    plugin, _ = make()
    _install(tmp_path, plugin, "qwen3-0.6b")
    _install(tmp_path, plugin, "qwen3-8b-fp8")
    plugin.dispatch("llm", {"action": "load", "model": "qwen3-0.6b"})
    assert _wait(lambda: plugin.dispatch("llm", {"action": "info"})["model"] == "qwen3-0.6b")
    plugin.dispatch("llm", {"action": "load", "model": "qwen3-8b-fp8", "memory_gb": 22})
    assert _wait(lambda: plugin.dispatch("llm", {"action": "info"})["model"] == "qwen3-8b-fp8")
    assert _FakeEngine.instances[0].stopped.is_set()     # 统一内存装不下两份
    saved = json.loads((tmp_path / "root" / "state" / "llm.json").read_text())
    assert saved == {"model": "qwen3-8b-fp8", "autoload": True,
                     "configs": {"qwen3-0.6b": {}, "qwen3-8b-fp8": {"memory_gb": 22.0}}}

    # 容器重启：新实例自动加载上次的选择
    plugin2, router2 = make()
    assert _wait(lambda: router2.resolve("qwen3-8b-fp8")[1] == 200)

    # 卸载后重启不再自动加载
    plugin2.dispatch("llm", {"action": "unload"})
    assert _wait(lambda: plugin2.dispatch("llm", {"action": "info"})["state"] == "idle")
    n = len(_FakeEngine.instances)
    plugin3, _ = make()
    time.sleep(0.2)
    assert len(_FakeEngine.instances) == n


def test_loaded_model_files_cannot_be_deleted(make, tmp_path):
    plugin, _ = make()
    _install(tmp_path, plugin, "qwen3-0.6b")
    plugin.dispatch("llm", {"action": "load", "model": "qwen3-0.6b"})
    assert _wait(lambda: plugin.dispatch("llm", {"action": "info"})["state"] == "ready")
    r = plugin.dispatch("llm", {"action": "delete_files", "model": "qwen3-0.6b"})
    assert "正在使用" in r["error"]["message"]


def test_unknown_action_is_an_error_not_an_exception(make):
    plugin, _ = make()
    assert plugin.dispatch("llm", {"action": "rm -rf"})["error"]["code"] == "invalid_argument"
    assert plugin.dispatch("other", {"action": "info"}) is None


def test_a_model_too_new_for_the_local_vllm_is_refused_before_downloading(make):
    """jp71 的基础镜像是 vLLM 0.11：别为一个起不来的模型先下 9 GB。"""
    plugin, _ = make(engine_version="0.11.0+582e4e37.nv25.11")
    rows = {m["id"]: m for m in plugin.dispatch("llm", {"action": "models"})["models"]}
    assert "需要 vLLM" in rows["phanthymotus-0.1"]["unsupported"]
    assert rows["qwen3-0.6b"]["unsupported"] == ""
    r = plugin.dispatch("llm", {"action": "load", "model": "phanthymotus-0.1"})
    assert "需要 vLLM" in r["error"]["message"]
    assert plugin.fake_downloads.started == []


def test_launch_config_is_saved_per_model_and_used_on_the_next_load(make, tmp_path):
    plugin, _ = make()
    _install(tmp_path, plugin, "qwen3-0.6b")
    got = plugin.dispatch("llm", {"action": "config_get", "model": "qwen3-0.6b"})
    assert got["config"]["max_model_len"] == 8192 and got["overridden"] == []
    assert got["defaults"]["max_model_len"] == 8192          # 页面在改过的字段旁写「默认 8192」
    r = plugin.dispatch("llm", {"action": "config_set", "model": "qwen3-0.6b",
                                "values": {"max_model_len": 4096, "max_num_seqs": 4,
                                           "cudagraph_capture_sizes": "1, 2 4"}})
    assert r["config"]["cudagraph_capture_sizes"] == [1, 2, 4]
    assert r["overridden"] == ["cudagraph_capture_sizes", "max_model_len", "max_num_seqs"]
    plugin.dispatch("llm", {"action": "load", "model": "qwen3-0.6b"})
    assert _wait(lambda: _FakeEngine.instances)
    eng = _FakeEngine.instances[-1]
    assert eng.serve["max_model_len"] == 4096 and eng.serve["max_num_seqs"] == 4
    # 传空恢复默认
    r = plugin.dispatch("llm", {"action": "config_set", "model": "qwen3-0.6b",
                                "values": {"max_model_len": None}})
    assert r["config"]["max_model_len"] == 8192


def test_config_set_rejects_what_would_not_start(make):
    plugin, _ = make()
    r = plugin.dispatch("llm", {"action": "config_set", "model": "qwen3-0.6b",
                                "values": {"memory_gb": 55}})
    assert "90%" in r["error"]["message"]          # 59 GB 的机器上 55 + 4.4 装不下
    r = plugin.dispatch("llm", {"action": "config_set", "model": "qwen3-0.6b",
                                "values": {"nope": 1}})
    assert "不认识" in r["error"]["message"]


def test_old_state_file_is_migrated_into_per_model_config(make, tmp_path):
    state = tmp_path / "root" / "state"
    state.mkdir(parents=True)
    (state / "llm.json").write_text(json.dumps({"model": "qwen3-0.6b", "memory_gb": 6.0,
                                                "max_model_len": 8192, "autoload": False}))
    plugin, _ = make()
    got = plugin.dispatch("llm", {"action": "config_get", "model": "qwen3-0.6b"})
    assert got["config"]["memory_gb"] == 6.0 and "memory_gb" in got["overridden"]


def test_autoload_toggle(make):
    plugin, _ = make()
    assert plugin.dispatch("llm", {"action": "set_autoload", "enabled": False})["autoload"] is False
    assert plugin.dispatch("llm", {"action": "config_get", "model": "qwen3-0.6b"})["autoload"] is False


def test_models_report_disk(make, tmp_path):
    plugin, _ = make()
    _install(tmp_path, plugin, "qwen3-0.6b")
    r = plugin.dispatch("llm", {"action": "models"})
    rows = {m["id"]: m for m in r["models"]}
    assert rows["qwen3-0.6b"]["disk_gb"] > 1.0 and rows["qwen3-8b-fp8"]["disk_gb"] == 0
    assert r["disk"]["total_gb"] > 0 and r["disk"]["models_gb"] >= rows["qwen3-0.6b"]["disk_gb"]


def test_playground_conversation_survives_and_can_be_cleared(make, tmp_path):
    """回到页面之前的内容还在：对话存在服务端，回复由服务端在生成完时追加。"""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.end_headers()
            for piece in ("你", "好"):
                self.wfile.write(f'data: {json.dumps({"choices": [{"delta": {"content": piece}}]})}\n\n'.encode())
            self.wfile.write(b"data: [DONE]\n\n")

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    plugin, _ = make()
    plugin._gateway_url = f"http://127.0.0.1:{srv.server_address[1]}"
    _install(tmp_path, plugin, "qwen3-0.6b")
    plugin.dispatch("llm", {"action": "load", "model": "qwen3-0.6b"})
    assert _wait(lambda: plugin.dispatch("llm", {"action": "info"})["state"] == "ready")
    plugin.dispatch("llm", {"action": "chat_start", "text": "嗨"})
    assert _wait(lambda: len(plugin.dispatch("llm", {"action": "chat_history"})["messages"]) == 2)

    plugin2, _ = make()                                  # 页面关了、容器重启了也一样
    msgs = plugin2.dispatch("llm", {"action": "chat_history"})["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[1]["content"] == "你好" and msgs[1]["state"] == "done"
    assert plugin2.dispatch("llm", {"action": "chat_clear"})["messages"] == []
    assert plugin2.dispatch("llm", {"action": "chat_history"})["messages"] == []
    srv.shutdown()


def test_cancelling_the_download_a_load_waits_for_drops_the_pending_load(make):
    plugin, _ = make()
    plugin.fake_downloads.cancel = lambda mid: {"model": mid, "state": "cancelled"}
    plugin.dispatch("llm", {"action": "load", "model": "qwen3-8b-fp8"})
    assert _wait(lambda: plugin.fake_downloads.started)
    plugin.dispatch("llm", {"action": "download_cancel", "model": "qwen3-8b-fp8"})
    assert plugin.dispatch("llm", {"action": "info"})["state"] == "idle"


def test_a_context_longer_than_the_kv_cache_is_refused_when_saved(make, tmp_path):
    """vLLM 要求 KV 至少放得下一个最长请求。N4 上 4B、131k 上下文、24 GB 预算实测起不来。"""
    plugin, _ = make()
    _install(tmp_path, plugin, "qwen3-4b-instruct-2507")
    cfg = model_dir(tmp_path / "root" / "models", "qwen3-4b-instruct-2507") / "config.json"
    cfg.write_text(json.dumps({"num_hidden_layers": 36, "num_key_value_heads": 8, "num_attention_heads": 32,
                               "head_dim": 128, "hidden_size": 2560, "torch_dtype": "bfloat16"}))
    r = plugin.dispatch("llm", {"action": "config_set", "model": "qwen3-4b-instruct-2507",
                                "values": {"max_model_len": 131072, "memory_gb": 24}})
    assert "需要 KV cache 18.0 GiB" in r["error"]["message"] and "至少 28 GB" in r["error"]["message"]
    ok = plugin.dispatch("llm", {"action": "config_set", "model": "qwen3-4b-instruct-2507",
                                 "values": {"max_model_len": 106496, "memory_gb": 25}})
    assert "error" not in ok and ok["kv"]["bytes_per_token"] == 147456
