#!/usr/bin/env python3
"""
tools/pin_model.py — 从 ModelScope 生成一行钉死 revision/size/SHA256 的注册表条目。

    python3 modelhub/tools/pin_model.py Qwen/Qwen3-8B-FP8 --id qwen3-8b-fp8 \
        --title "Qwen3 8B (FP8)" --memory-gb 20 --max-model-len 32768 \
        --tool-call-parser hermes --reasoning-parser qwen3

输出贴进 modelhub/models.yaml 的 models: 下面。运维页的「添加模型」走的是同一个函数。
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serving.registry import pin_from_modelscope  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("--revision", default="master")
    ap.add_argument("--id", required=True)
    ap.add_argument("--title")
    ap.add_argument("--note", default="")
    ap.add_argument("--memory-gb", type=float)
    ap.add_argument("--max-model-len", type=int)
    ap.add_argument("--tool-call-parser")
    ap.add_argument("--reasoning-parser")
    a = ap.parse_args()
    entry = {"title": a.title or a.repo.rsplit("/", 1)[-1], "kind": "llm"}
    if a.note:
        entry["note"] = a.note
    entry.update(pin_from_modelscope(a.repo, a.revision))
    serve = {k: v for k, v in {"memory_gb": a.memory_gb, "max_model_len": a.max_model_len,
                               "tool_call_parser": a.tool_call_parser,
                               "reasoning_parser": a.reasoning_parser}.items() if v}
    if serve:
        entry["serve"] = serve
    print(yaml.safe_dump({a.id: entry}, allow_unicode=True, sort_keys=False, width=120))


if __name__ == "__main__":
    main()
