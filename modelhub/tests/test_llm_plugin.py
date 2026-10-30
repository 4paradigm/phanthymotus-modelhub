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
                 load_timeout_s, executable):
        self.model_id, self.serve, self.budget_gb = model_id, serve, budget_gb
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
    active = None

    def job(self, _):
        return None

    def jobs(self):
        return {}


@pytest.fixture()
def make(tmp_path):
    _FakeEngine.instances.clear()

    def build():
        router = Router()
        plugin = LLMPlugin({"default_memory_gb": 8, "backend_port": 15750}, router=router,
                           models_root=str(tmp_path / "llm"), state_dir=str(tmp_path / "state"),
                           builtin_registry=str(BUILTIN), gateway_url="http://127.0.0.1:15741",
                           engine_factory=_FakeEngine, downloads=_FakeDownloads(), total_gb=59.0)
        return plugin, router
    return build


def _install(tmp_path, plugin, model_id):
    spec = plugin._registry.get(model_id)
    d = model_dir(tmp_path / "llm", model_id)
    d.mkdir(parents=True)
    for name, meta in spec.files.items():
        (d / name).parent.mkdir(parents=True, exist_ok=True)
        with open(d / name, "wb") as f:
            f.truncate(meta["size"])
    mark_installed(spec, tmp_path / "llm")


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


def test_load_refuses_a_model_that_is_not_downloaded(make):
    plugin, _ = make()
    r = plugin.dispatch("llm", {"action": "load", "model": "qwen3-0.6b"})
    assert "还没下载" in r["error"]["message"]
    assert plugin.dispatch("llm", {"action": "load", "model": "nope"})["error"]


def test_switching_stops_the_old_engine_first_and_autoload_restores(make, tmp_path):
    plugin, _ = make()
    _install(tmp_path, plugin, "qwen3-0.6b")
    _install(tmp_path, plugin, "qwen3-8b-fp8")
    plugin.dispatch("llm", {"action": "load", "model": "qwen3-0.6b"})
    assert _wait(lambda: plugin.dispatch("llm", {"action": "info"})["model"] == "qwen3-0.6b")
    plugin.dispatch("llm", {"action": "load", "model": "qwen3-8b-fp8", "memory_gb": 22})
    assert _wait(lambda: plugin.dispatch("llm", {"action": "info"})["model"] == "qwen3-8b-fp8")
    assert _FakeEngine.instances[0].stopped.is_set()     # 统一内存装不下两份
    saved = json.loads((tmp_path / "state" / "llm.json").read_text())
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
