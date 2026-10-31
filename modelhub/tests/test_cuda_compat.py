"""镜像的 CUDA 和本机驱动：比不过就说清楚换哪个版本；vLLM 退出时先找根因。"""

from serving.cuda_compat import image_cuda, mismatch
from serving.vllm_engine import explain_exit


def test_image_newer_than_driver_names_the_version_to_install():
    msg = mismatch((13, 0), (13, 2))
    assert "CUDA 13.2" in msg and "CUDA 13.0（JetPack 7.1）" in msg
    assert "jetson-jp7.1" in msg and "设置 → 部署" in msg


def test_same_or_older_image_is_fine():
    assert mismatch((13, 2), (13, 0)) == ""      # 新驱动跑旧 CUDA 没问题
    assert mismatch((13, 0), (13, 0)) == ""


def test_unknown_driver_or_image_does_not_block():
    assert mismatch(None, (13, 4)) == ""
    assert mismatch((13, 0), None) == ""
    assert mismatch((12, 6), (13, 0)).endswith("和本机 JetPack 对应的版本。")


def test_image_cuda_reads_the_ngc_variable():
    assert image_cuda({"CUDA_VERSION": "13.4.3.001"}) == (13, 4)
    assert image_cuda({"CUDA_VERSION": "13.2.1"}) == (13, 2)
    assert image_cuda({}) is None


def test_vllm_exit_names_the_root_cause_not_the_last_traceback():
    # thor_pack 2026-10-31 实测：根因在前，最后几行是外层堆栈
    logs = ["INFO starting",
            "  [[System has unsupported display driver / cuda driver combination "
            "(CUDA_ERROR_SYSTEM_DRIVER_MISMATCH) cuInit()=803]]",
            "Traceback (most recent call last):",
            "RuntimeError: Engine core initialization failed. See root cause above."]
    msg = explain_exit(1, logs)
    assert "不匹配" in msg and "Traceback" not in msg
    assert "内存不够" in explain_exit(1, ["torch.OutOfMemoryError: CUDA out of memory."])
    assert explain_exit(1, ["something else"]) == "vLLM 退出（code 1）：something else"


def test_startup_phases_follow_a_real_vllm_log():
    """thor_pack 2026-10-31 上 Qwen3 4B 一次成功启动的输出（vLLM 0.11）。"""
    from serving.vllm_engine import next_phase
    log = [
        ("INFO [core.py:77] Initializing a V1 LLM engine (v0.11.0) with config: ...", "start"),
        ("INFO [gpu_model_runner.py:2634] Loading model from scratch...", "weights"),
        ("Loading safetensors checkpoint shards:  67% Completed | 2/3 [00:02<00:01]", "weights"),
        ("INFO [default_loader.py:267] Loading weights took 5.88 seconds", "weights"),
        ("INFO [backends.py:559] Dynamo bytecode transform time: 6.21 s", "compile"),
        ("INFO [monitor.py:34] torch.compile takes 6.21 s in total", "kv"),
        ("INFO [kv_cache_utils.py:1087] GPU KV cache size: 109,280 tokens", "kv"),
        ("Capturing CUDA graphs (decode, FULL):  67%|██████▋   | 2/3", "graph"),
        ("INFO [core.py:210] init engine (profile, create kv cache, warmup model) took 60.72 seconds", "serve"),
        ("INFO [api_server.py:1915] Starting vLLM API server 0 on http://127.0.0.1:15750", "serve"),
        ("INFO Loading weights took 1 s", "serve"),          # 只往前走，不会退回去
    ]
    phase = "start"
    for line, want in log:
        phase = next_phase(phase, line)
        assert phase == want, line
