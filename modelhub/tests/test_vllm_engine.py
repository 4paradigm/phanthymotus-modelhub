"""vLLM 进程管理：内存换算、命令行、以及对真子进程（假的 vllm）的起停探活。"""

import os
import socket
import time
from pathlib import Path

import pytest

from serving import vllm_engine
from serving.vllm_engine import (ERROR, IDLE, LOADING, READY, VLLMEngine, build_command,
                                 memory_fraction)

FAKE_VLLM = str(Path(__file__).with_name("fake_vllm.py"))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait(pred, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.05)
    return False


def test_memory_fraction_is_budget_over_whole_unified_memory():
    assert memory_fraction(16, 59.0) == pytest.approx(0.2712, abs=1e-4)
    with pytest.raises(ValueError, match="90%"):
        memory_fraction(55, 59.0)       # 会把系统和 perception 挤死
    with pytest.raises(ValueError):
        memory_fraction(0, 59.0)


def test_command_binds_loopback_and_enables_tool_calls():
    cmd = build_command("/models/llm/m", "m", 15750,
                        {"max_model_len": 8192, "tool_call_parser": "hermes",
                         "reasoning_parser": "qwen3", "extra_args": ["--seed", "1"]},
                        budget_gb=6, total_gb=59)
    joined = " ".join(cmd)
    assert cmd[:3] == ["vllm", "serve", "/models/llm/m"]
    assert "--host 127.0.0.1" in joined          # 只有网关对外
    assert "--served-model-name m" in joined     # 网关按这个名字路由
    assert "--enable-auto-tool-choice --tool-call-parser hermes" in joined
    assert "--reasoning-parser qwen3" in joined
    assert cmd[-2:] == ["--seed", "1"]
    with pytest.raises(ValueError):
        build_command("/m", "m", 1, {"extra_args": "--seed 1"}, 6, 59)


def _engine(port, **env):
    return VLLMEngine(model_id="m", model_path="/models/llm/m", port=port, serve={},
                      budget_gb=6, total_gb=59, executable=FAKE_VLLM, load_timeout_s=10,
                      env={k: str(v) for k, v in env.items()})


def test_loading_becomes_ready_and_stop_kills_the_process():
    port = _free_port()
    eng = _engine(port, FAKE_VLLM_LOAD_S=0.3)
    eng.start()
    assert eng.state == LOADING
    assert _wait(lambda: eng.state == READY), eng.status()
    pid = eng.status()["pid"]
    assert any("startup complete" in line for line in eng.logs())
    eng.stop(timeout_s=5)
    assert eng.state == IDLE
    assert _wait(lambda: not _alive(pid))


def test_crash_while_loading_is_an_error_with_the_log_tail():
    eng = _engine(_free_port(), FAKE_VLLM_CRASH=1)
    eng.start()
    assert _wait(lambda: eng.state == ERROR)
    assert "out of memory" in eng.status()["error"]   # 原因要直接出现在运维页上


def test_stop_during_loading_cancels_it():
    eng = _engine(_free_port(), FAKE_VLLM_LOAD_S=30)
    eng.start()
    t0 = time.monotonic()
    eng.stop(timeout_s=5)
    assert eng.state == IDLE
    assert time.monotonic() - t0 < 5


def _alive(pid) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # 已退出但还没被回收的僵尸也算死
    try:
        return "Z" not in Path(f"/proc/{pid}/stat").read_text().split()[2]
    except OSError:
        return True
