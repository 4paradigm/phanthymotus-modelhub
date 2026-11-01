"""vLLM 进程管理：内存换算、命令行、以及对真子进程（假的 vllm）的起停探活。"""

import json
import os
import socket
import time
from pathlib import Path

import pytest

from serving import vllm_engine
from serving.vllm_engine import (ERROR, IDLE, LOADING, READY, VLLMEngine, build_command,
                                 cache_env, footprint_gb, memory_fraction, serve_env)

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


def test_the_90_percent_cap_counts_the_process_overhead_not_just_the_budget():
    """Thor 实测：预算 6 GB 时整机多占 9.8 GB。只看预算会放行一个实际吃掉 92% 的配置。"""
    assert 50 / 59.0 < 0.9                       # 只看预算：放行
    with pytest.raises(ValueError, match="实际约占 54.4 GB"):
        memory_fraction(50, 59.0, overhead_gb=4.4)
    assert footprint_gb(6, 4.4) == 10.4


def test_command_binds_loopback_and_enables_tool_calls():
    cmd = build_command("/models/modelhub/models/m", "m", 15750,
                        {"max_model_len": 8192, "tool_call_parser": "hermes",
                         "reasoning_parser": "qwen3", "extra_args": ["--seed", "1"]},
                        budget_gb=6, total_gb=59)
    joined = " ".join(cmd)
    assert cmd[:3] == ["vllm", "serve", "/models/modelhub/models/m"]
    assert "--host 127.0.0.1" in joined          # 只有网关对外
    assert "--served-model-name m" in joined     # 网关按这个名字路由
    assert "--enable-auto-tool-choice --tool-call-parser hermes" in joined
    assert "--reasoning-parser qwen3" in joined
    assert cmd[-2:] == ["--seed", "1"]
    with pytest.raises(ValueError):
        build_command("/m", "m", 1, {"extra_args": "--seed 1"}, 6, 59)


def test_cached_prompt_tokens_are_reported():
    """不开这个参数，usage 里没有 cached_tokens，前缀缓存命中与否从外面看不出来。"""
    assert "--enable-prompt-tokens-details" in build_command("/m", "m", 1, {}, 6, 59)


def test_a_models_own_env_reaches_the_vllm_process(tmp_path):
    """FP8 在 Thor 上要 VLLM_TEST_FORCE_FP8_MARLIN=1 才不出 NaN —— 开关写在模型的 serve 段里。"""
    serve = {"env": {"VLLM_TEST_FORCE_FP8_MARLIN": 1}}
    assert serve_env(serve) == {"VLLM_TEST_FORCE_FP8_MARLIN": "1"}
    eng = VLLMEngine(model_id="m", model_path="/m", port=1, serve=serve, budget_gb=6,
                     total_gb=59, cache_dir=str(tmp_path), env={"FAKE_VLLM_LOAD_S": "0"})
    assert eng._env["VLLM_TEST_FORCE_FP8_MARLIN"] == "1"
    assert eng._env["VLLM_CACHE_ROOT"] == f"{tmp_path}/vllm"     # 缓存目录照旧
    assert "env" not in " ".join(eng.command)                     # 不会混进命令行
    for bad in ({"env": ["A=1"]}, {"env": {"A": True}}, {"env": {"A": None}}):
        with pytest.raises(ValueError):
            serve_env(bad)


def test_cudagraph_capture_sizes_become_a_compilation_config():
    cmd = build_command("/m", "m", 1, {"cudagraph_capture_sizes": [8, 1, 4, 2, 2]}, 6, 59)
    i = cmd.index("--compilation-config")
    assert json.loads(cmd[i + 1]) == {"cudagraph_capture_sizes": [1, 2, 4, 8]}
    with pytest.raises(ValueError):
        build_command("/m", "m", 1, {"cudagraph_capture_sizes": [0]}, 6, 59)


def test_every_cache_goes_to_the_volume(tmp_path):
    """容器内 ~/.cache 随容器重建丢失；Thor 实测编译 18 s → 命中缓存 0.4 s。"""
    env = cache_env(str(tmp_path))
    assert set(env) >= {"VLLM_CACHE_ROOT", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR",
                        "XDG_CACHE_HOME", "HF_HOME"}
    assert all(v.startswith(str(tmp_path)) for v in env.values())
    eng = VLLMEngine(model_id="m", model_path="/m", port=1, serve={}, budget_gb=6,
                     total_gb=59, cache_dir=str(tmp_path))
    assert eng._env["VLLM_CACHE_ROOT"] == f"{tmp_path}/vllm"


def _engine(port, **env):
    return VLLMEngine(model_id="m", model_path="/models/modelhub/models/m", port=port, serve={},
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


def test_log_lines_are_cleaned_and_our_own_polls_dropped():
    """vLLM 的日志带终端颜色码；监控每 5 秒抓 /metrics 会把日志刷满。"""
    from serving.vllm_engine import _OWN_POLLS, clean_log_line, log_level
    raw = "\x1b[1;36m(APIServer pid=139)\x1b[0;0m INFO 10-31 11:13:43 [loggers.py:127] Engine 000: ok"
    line = clean_log_line(raw)
    assert line == "INFO 10-31 11:13:43 [loggers.py:127] Engine 000: ok"
    assert log_level(line) == "INFO"
    assert log_level("torch.AcceleratorError: CUDA error: boom") == "ERROR"
    assert _OWN_POLLS.search('INFO:     127.0.0.1:1 - "GET /metrics HTTP/1.1" 200 OK')
    assert not _OWN_POLLS.search('INFO:     127.0.0.1:1 - "POST /v1/chat/completions HTTP/1.1" 200 OK')



def test_kv_cache_is_given_so_vllm_skips_its_memory_profiling():
    """Thor 上 vLLM 的显存 profiling 拿 MemFree 当空闲；page cache 被回收时它断言失败。"""
    from serving.vllm_engine import kv_cache_bytes
    cmd = build_command("/m", "m", 1, {}, budget_gb=16, total_gb=57.8,
                        weights_bytes=int(8.06e9))
    i = cmd.index("--kv-cache-memory-bytes")
    kv_gb = int(cmd[i + 1]) / 2**30
    assert 5.5 < kv_gb < 6.1                      # 16 - 7.5 GiB 权重 - 2.5 余量
    assert cmd[cmd.index("--gpu-memory-utilization") + 1] == "0.01"
    with pytest.raises(ValueError, match="放不下"):
        kv_cache_bytes(8, int(8.06e9))
    # 不知道权重大小时退回比例
    assert "--kv-cache-memory-bytes" not in build_command("/m", "m", 1, {}, 6, 59)


def test_kv_bytes_per_token_from_the_models_own_config():
    """页面据此估「KV cache 放得下多少 token」。Qwen3-4B 在 N4 上：估 43,637，引擎报 43,744。"""
    from serving.vllm_engine import kv_bytes_per_token
    qwen3_4b = {"num_hidden_layers": 36, "num_key_value_heads": 8, "num_attention_heads": 32,
                "head_dim": 128, "hidden_size": 2560, "torch_dtype": "bfloat16"}
    assert kv_bytes_per_token(qwen3_4b) == 147456
    assert kv_bytes_per_token(qwen3_4b, "fp8") == 73728
    # Qwen3.5：多模态把语言部分放在 text_config；每 4 层只有 1 层全注意力
    qwen35 = {"text_config": {"num_hidden_layers": 32, "num_key_value_heads": 4, "num_attention_heads": 16,
                              "head_dim": 256, "hidden_size": 2560, "dtype": "bfloat16",
                              "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 8}}
    assert kv_bytes_per_token(qwen35) == 2 * 8 * 4 * 256 * 2
    assert kv_bytes_per_token({}) is None
