"""
serving/vllm_info.py — 本机 vLLM 认哪些解析器，给运维页的下拉选单用。

「工具调用解析器」「思考解析器」是有限选项，而且随 vLLM 版本变（jp71 的 0.11 有 24 种
工具解析器）。写死一张表会和镜像里的版本对不上，所以启动时在子进程里问一次 vLLM 自己
—— 子进程是因为 import vllm 要四秒多、还会拉起 torch，不该进 MCP 主进程。
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading

# 常见的几个说清楚是给哪类模型的；其余的照原名列出
TOOL_LABELS = {
    "hermes": "hermes — Qwen3、Qwen2.5 等（<tool_call> 里放 JSON）",
    "qwen3_coder": "qwen3_coder — Qwen3.5、Qwen3-Coder（XML 格式）",
    "qwen3_xml": "qwen3_xml — Qwen3 的 XML 格式",
    "llama3_json": "llama3_json — Llama 3.1 / 3.2",
    "llama4_json": "llama4_json — Llama 4",
    "mistral": "mistral — Mistral 系列",
    "deepseek_v3": "deepseek_v3 — DeepSeek V3",
    "deepseek_v31": "deepseek_v31 — DeepSeek V3.1",
    "glm45": "glm45 — GLM-4.5",
    "kimi_k2": "kimi_k2 — Kimi K2",
    "openai": "openai — gpt-oss",
    "pythonic": "pythonic — 输出 Python 函数调用的模型",
}
REASONING_LABELS = {
    "qwen3": "qwen3 — Qwen3 的 <think> 思考",
    "deepseek_r1": "deepseek_r1 — DeepSeek-R1",
    "glm45": "glm45 — GLM-4.5",
    "mistral": "mistral — Mistral 推理模型",
    "openai_gptoss": "openai_gptoss — gpt-oss",
}

_PROBE = r"""
import json
try:
    from vllm.entrypoints.openai.tool_parsers import ToolParserManager
except ImportError:
    from vllm.tool_parsers import ToolParserManager
from vllm.reasoning import ReasoningParserManager
print("@@" + json.dumps({"tool": sorted(ToolParserManager.tool_parsers),
                         "reasoning": sorted(ReasoningParserManager.reasoning_parsers)}))
"""


def _options(names, labels) -> list[dict]:
    known = [n for n in labels if n in names]
    rest = [n for n in names if n not in labels]
    return [{"value": n, "label": labels.get(n, n)} for n in known + rest]


class ParserChoices:
    def __init__(self, probe: bool = True, python: str = sys.executable):
        self._lock = threading.Lock()
        self._names = {"tool": [], "reasoning": []}
        if probe:
            threading.Thread(target=self._probe, args=(python,), daemon=True,
                             name="vllm-parsers").start()

    def _probe(self, python: str) -> None:
        try:
            out = subprocess.run([python, "-c", _PROBE], capture_output=True, text=True,
                                 timeout=120).stdout
            line = next(l for l in out.splitlines() if l.startswith("@@"))
            names = json.loads(line[2:])
        except (OSError, subprocess.SubprocessError, StopIteration, ValueError):
            return                                   # 拿不到就退回成自由输入
        with self._lock:
            self._names = names

    def get(self) -> dict:
        with self._lock:
            names = dict(self._names)
        return {"tool_call_parser": _options(names["tool"], TOOL_LABELS),
                "reasoning_parser": _options(names["reasoning"], REASONING_LABELS)}
