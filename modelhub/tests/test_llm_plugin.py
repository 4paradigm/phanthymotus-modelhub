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
                 load_timeout_s, executable, overhead_gb, cache_dir):
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


CFG = {"default_memory_gb": 8, "backend_port": 15750, "host_overhead_gb": 4.4,
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
    assert saved == {"model": "qwen3-8b-fp8", "memory_gb": 22.0,
                     "max_model_len": 32768, "autoload": True}

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
