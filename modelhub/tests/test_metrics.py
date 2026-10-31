"""监控：Prometheus 解析、差分出速率、5 秒 → 1 分钟汇总、按时间段查、保留期、事件。"""

from serving.metrics import (MINUTE_KEEP_S, RAW_KEEP_S, MetricsStore, Sampler,
                             parse_prometheus)

PROM = """# HELP x
vllm:num_requests_running{engine="0",model_name="m"} 2.0
vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.125
vllm:generation_tokens_total{engine="0",model_name="m"} 1000.0
vllm:prompt_tokens_total{engine="0",model_name="m"} 400.0
vllm:request_success_total{engine="0",finished_reason="stop",model_name="m"} 6.0
vllm:request_success_total{engine="0",finished_reason="length",model_name="m"} 1.0
"""


def test_prometheus_counters_are_summed_across_labels():
    p = parse_prometheus(PROM)
    assert p["vllm:request_success_total"] == 7.0
    assert p["vllm:num_requests_running"] == 2.0


class _Eng:
    state, base_url = "ready", "http://x/v1"

    def pgid(self):
        return None


def test_sampler_turns_counters_into_rates_and_records_5xx(tmp_path):
    store = MetricsStore(str(tmp_path / "m.db"))
    texts = [PROM, PROM.replace("1000.0", "1500.0").replace("400.0", "450.0")]
    gw = [{"requests": 10, "errors_5xx": 0}, {"requests": 20, "errors_5xx": 3}]
    s = Sampler(store, lambda: _Eng(), lambda: gw.pop(0), fetch=lambda url: texts.pop(0))
    first = s.sample(now=1000.0)
    assert "gen_tps" not in first                       # 第一次没有上一个值，算不出速率
    assert first["running"] == 2.0 and first["kv_pct"] == 12.5
    second = s.sample(now=1005.0)
    assert second["gen_tps"] == 100.0 and second["prompt_tps"] == 10.0
    assert second["req_rate"] == 2.0 and second["err_rate"] == 0.6
    ev = store.events("24h", now=1010)
    assert ev and ev[0]["source"] == "gateway" and "3 个请求返回 5xx" in ev[0]["message"]


def test_engine_restart_does_not_produce_a_negative_rate(tmp_path):
    store = MetricsStore(str(tmp_path / "m.db"))
    texts = [PROM, PROM.replace("1000.0", "10.0")]
    s = Sampler(store, lambda: _Eng(), lambda: {}, fetch=lambda url: texts.pop(0))
    s.sample(now=0.0)
    assert "gen_tps" not in s.sample(now=5.0)


def test_rollup_query_and_retention(tmp_path):
    store = MetricsStore(str(tmp_path / "m.db"))
    now = 10 * 86400
    for t in range(now - 600, now, 5):                   # 10 分钟，每 5 秒一个点
        store.add(t, {"gen_tps": 100.0, "temp_gpu": 40.0})
    store.add(now - RAW_KEEP_S - 60, {"gen_tps": 1.0})   # 超过 24 小时的原始点
    store.rollup(now)
    q = store.query("15m", now=now)
    assert q["bucket_s"] == 5 and len(q["ts"]) == 120
    assert set(q["series"]["gen_tps"]) == {100.0}
    assert q["series"]["gpu_util"][0] is None            # 没采到的指标是空，不是 0
    raw_left = store._db.execute("SELECT COUNT(*) FROM raw WHERE ts < ?", (now - RAW_KEEP_S,)).fetchone()[0]
    assert raw_left == 0
    minutes = store.query("7d", now=now)                 # 超过 6 小时的范围走 minute 表
    assert minutes["bucket_s"] >= 60 and minutes["series"]["gen_tps"][-1] == 100.0
    store.add(now + MINUTE_KEEP_S, {"gen_tps": 5.0})
    store.rollup(now + MINUTE_KEEP_S + 120)
    assert store._db.execute("SELECT MIN(ts) FROM minute").fetchone()[0] > now


def test_query_never_returns_more_than_a_chart_needs(tmp_path):
    store = MetricsStore(str(tmp_path / "m.db"))
    now = 100000
    for t in range(now - 6 * 3600, now, 5):
        store.add(t, {"gen_tps": 1.0})
    assert len(store.query("6h", now=now)["ts"]) <= 241
