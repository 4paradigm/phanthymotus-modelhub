"""性能测试：跑完三组、存档、列表、删除；失败和停止不存档。"""

import time

import pytest

from serving.bench import BenchRunner


def _fake(calls):
    def one(url, model, prompt, max_tokens, timeout=300):
        calls.append((len(prompt), max_tokens))
        return {"ttft_ms": 30.0 + len(prompt) / 100, "total_s": 1.0, "prompt_tokens": len(prompt) // 2,
                "gen_tokens": max_tokens, "decode_tps": 100.0, "itl": [0.01] * max_tokens}
    return one


def _wait(pred, t=5):
    end = time.monotonic() + t
    while time.monotonic() < end and not pred():
        time.sleep(0.01)
    return pred()


CFG = {"max_model_len": 8192, "max_num_seqs": 8, "memory_gb": 6.0}


def test_a_run_measures_all_three_groups_and_is_kept(tmp_path):
    calls = []
    b = BenchRunner(tmp_path, "http://gw", one=_fake(calls))
    b.start("m", "M", CFG, "0.11.0", 57.8)
    assert _wait(lambda: b.status()["state"] == "done")
    [run] = b.list()
    r = run["results"]
    assert [t["prompt_tokens"] for t in r["ttft"]] and len(r["ttft"]) == 3
    assert r["decode"]["tps"] == 100.0
    assert [c["n"] for c in r["concurrency"]] == [1, 4, 8]
    pc = r["prefix_cache"]
    assert pc["cold_ttft_ms"] and pc["warm_ttft_ms"] and pc["speedup"]
    assert run["config"]["max_model_len"] == 8192 and run["engine_version"] == "0.11.0"
    b.delete(run["id"])
    assert b.list() == []


def test_prompt_lengths_and_concurrency_respect_the_config(tmp_path):
    b = BenchRunner(tmp_path, "http://gw", one=_fake([]))
    b.start("m", "M", {"max_model_len": 2048, "max_num_seqs": 4}, "", 59)
    assert _wait(lambda: b.status()["state"] == "done")
    r = b.list()[0]["results"]
    assert len(r["ttft"]) == 2                        # 4096 的长提示放不进 2048 的上下文
    assert [c["n"] for c in r["concurrency"]] == [1, 4]


def test_a_failed_run_is_reported_and_not_kept(tmp_path):
    def boom(*a, **k):
        raise OSError("网关 503")
    b = BenchRunner(tmp_path, "http://gw", one=boom)
    b.start("m", "M", CFG, "", 59)
    assert _wait(lambda: b.status()["state"] == "error")
    assert "503" in b.status()["error"] and b.list() == []


def test_only_one_run_at_a_time_and_ids_are_checked(tmp_path):
    def slow(*a, **k):
        time.sleep(0.2)
        return _fake([])(*a, **k)
    b = BenchRunner(tmp_path, "http://gw", one=slow)
    b.start("m", "M", CFG, "", 59)
    with pytest.raises(RuntimeError):
        b.start("m", "M", CFG, "", 59)
    b.stop()
    assert _wait(lambda: b.status()["state"] == "stopped", 10)
    with pytest.raises(ValueError):
        b.delete("../x")
