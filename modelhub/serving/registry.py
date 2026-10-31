"""
serving/registry.py — 模型注册表：能选哪些模型、从哪下、怎么起。

一个模型就是一行：来源（ModelScope 仓库 + 固定 revision）、每个文件钉死的
size/SHA256、以及起服务需要的参数。加一个模型是加一行，不是写代码。

两份来源，后者覆盖前者：
  - models.yaml           随镜像发的内置表
  - <root>/state/models.user.yaml  运维页「添加模型」写进来的，留在 /models 卷上，
                                   换镜像不丢

文件为什么要钉 SHA256：LLM 权重动辄十几 GB，在 Thor 上走 ModelScope 要下几十分钟。
一个截断的 shard 能让 vLLM 起到一半才报错，读起来像引擎问题。钉死之后，下载器
（perception 的 model_downloader.ensure_verified_bundle）在落盘前就会拒绝它。
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import yaml

MODEL_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

MODELSCOPE_API = "https://www.modelscope.cn/api/v1/models"

# 下载时不要的文件：文档、图片、以及有 safetensors 时的旧格式权重。
_SKIP_NAMES = {".gitattributes", "configuration.json",
               # 训练产物：微调仓库常把它们一起传上来，推理用不到
               "trainer_state.json", "training_args.bin", "args.json"}
_SKIP_SUFFIXES = (".md", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf")
_LEGACY_WEIGHT_SUFFIXES = (".bin", ".pth", ".pt", ".gguf", ".h5", ".msgpack", ".onnx")

KINDS = ("llm",)   # 以后加 vlm / embedding 时扩展；vla 走另一套协议，不在这里


@dataclass(frozen=True)
class ModelSpec:
    id: str
    title: str
    kind: str
    repo: str                      # ModelScope 仓库名，如 Qwen/Qwen3-8B-FP8
    revision: str                  # 固定的 commit；不接受 master 这种会漂的引用
    files: dict                    # {relpath: {size, sha256}}
    serve: dict = field(default_factory=dict)
    note: str = ""
    builtin: bool = True
    # 能加载这个模型的最低 vLLM 版本。基础镜像按 JetPack 线不同、vLLM 版本差很多
    # （jp71 是 0.11，jp72 是 0.19），新架构在旧 vLLM 上会在引擎初始化深处才失败。
    min_vllm: str = ""

    @property
    def size_bytes(self) -> int:
        return sum(int(m["size"]) for m in self.files.values())

    def sources(self) -> list[str]:
        """model_downloader 认的源模板列表（带 {file} 占位）。"""
        return [f"{MODELSCOPE_API}/{self.repo}/repo?Revision={self.revision}&FilePath={{file}}"]

    def to_public(self) -> dict:
        return {
            "id": self.id, "title": self.title, "kind": self.kind,
            "repo": self.repo, "revision": self.revision,
            "size_gb": round(self.size_bytes / 1e9, 2), "files": len(self.files),
            "serve": dict(self.serve), "note": self.note, "builtin": self.builtin,
            "min_vllm": self.min_vllm,
        }


def _validate(model_id: str, raw: dict, builtin: bool) -> ModelSpec:
    if not MODEL_ID_RE.match(model_id):
        raise ValueError(f"模型 id 不合法：{model_id!r}（小写字母数字开头，只含 a-z0-9._-）")
    if not isinstance(raw, dict):
        raise ValueError(f"{model_id}: 条目必须是 mapping")
    kind = raw.get("kind", "llm")
    if kind not in KINDS:
        raise ValueError(f"{model_id}: kind 只能是 {KINDS}，收到 {kind!r}")
    source = raw.get("source") or {}
    repo, revision = source.get("modelscope"), source.get("revision")
    if not repo or "/" not in str(repo):
        raise ValueError(f"{model_id}: source.modelscope 必须是 <org>/<name>")
    if not revision or revision in ("master", "main"):
        raise ValueError(f"{model_id}: source.revision 必须是固定 commit，不能是分支名")
    files = raw.get("files") or {}
    if not files:
        raise ValueError(f"{model_id}: files 为空")
    for name, meta in files.items():
        if not isinstance(meta, dict) or not isinstance(meta.get("size"), int) \
                or not _SHA256_RE.match(str(meta.get("sha256", ""))):
            raise ValueError(f"{model_id}: 文件 {name} 缺 size(int)/sha256(64 hex)")
        if name.startswith("/") or ".." in name.split("/"):
            raise ValueError(f"{model_id}: 文件名越界：{name}")
    serve = raw.get("serve") or {}
    if not isinstance(serve, dict):
        raise ValueError(f"{model_id}: serve 必须是 mapping")
    min_vllm = str(raw.get("min_vllm") or "")
    if min_vllm and not re.match(r"^\d+(\.\d+)*$", min_vllm):
        raise ValueError(f"{model_id}: min_vllm 必须是版本号，如 0.19，收到 {min_vllm!r}")
    return ModelSpec(id=model_id, title=str(raw.get("title") or model_id), kind=kind,
                     repo=str(repo), revision=str(revision), files=dict(files),
                     serve=dict(serve), note=str(raw.get("note") or ""), builtin=builtin,
                     min_vllm=min_vllm)


def _load_file(path: Path, builtin: bool) -> dict[str, ModelSpec]:
    if not path.is_file():
        return {}
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out = {}
    for model_id, raw in (doc.get("models") or {}).items():
        out[model_id] = _validate(str(model_id), raw, builtin)
    return out


class Registry:
    """内置表 + 用户表。每次读都重新加载 —— 文件很小，换来的是不用管缓存失效。"""

    def __init__(self, builtin_path: str | Path, user_path: str | Path):
        self.builtin_path = Path(builtin_path)
        self.user_path = Path(user_path)

    def all(self) -> dict[str, ModelSpec]:
        models = _load_file(self.builtin_path, builtin=True)
        # 用户表里一行坏了不能拖垮整张表：内置模型照样能选
        try:
            models.update(_load_file(self.user_path, builtin=False))
        except (ValueError, yaml.YAMLError) as error:
            raise ValueError(f"{self.user_path} 有误：{error}") from error
        return models

    def get(self, model_id: str) -> ModelSpec:
        models = self.all()
        if model_id not in models:
            raise KeyError(f"未知模型：{model_id}")
        return models[model_id]

    def add_user_model(self, model_id: str, entry: dict) -> ModelSpec:
        spec = _validate(model_id, entry, builtin=False)
        if spec.id in _load_file(self.builtin_path, builtin=True):
            raise ValueError(f"{model_id} 和内置模型重名")
        doc = {}
        if self.user_path.is_file():
            doc = yaml.safe_load(self.user_path.read_text(encoding="utf-8")) or {}
        doc.setdefault("models", {})[model_id] = entry
        self.user_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.user_path.with_suffix(".tmp")
        tmp.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")
        os.replace(tmp, self.user_path)
        return spec

    def remove_user_model(self, model_id: str) -> None:
        if not self.user_path.is_file():
            raise KeyError(f"{model_id} 不是用户添加的模型")
        doc = yaml.safe_load(self.user_path.read_text(encoding="utf-8")) or {}
        if model_id not in (doc.get("models") or {}):
            raise KeyError(f"{model_id} 不是用户添加的模型")
        del doc["models"][model_id]
        tmp = self.user_path.with_suffix(".tmp")
        tmp.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")
        os.replace(tmp, self.user_path)


# ── 引擎版本 ──────────────────────────────────────────────────────────────────

def version_tuple(version: str) -> tuple[int, ...]:
    """'0.11.0+582e4e37.nv25.11' -> (0, 11, 0)：只取开头的数字段。"""
    m = re.match(r"^(\d+(?:\.\d+)*)", version or "")
    return tuple(int(p) for p in m.group(1).split(".")) if m else ()


def unsupported_reason(spec: ModelSpec, engine_version: str) -> str:
    """本机引擎能不能加载这个模型；能就返回空串。版本未知时不拦。"""
    if not spec.min_vllm or not version_tuple(engine_version):
        return ""
    if version_tuple(engine_version) < version_tuple(spec.min_vllm):
        return (f"需要 vLLM ≥ {spec.min_vllm}，本机是 {engine_version.split('+')[0]}"
                f"（基础镜像受宿主机驱动的 CUDA 版本限制，见 Dockerfile.jetson）")
    return ""


def installed_vllm_version() -> str:
    try:
        from importlib.metadata import version
        return version("vllm")
    except Exception:       # noqa: BLE001 — 本地跑测试时没有 vLLM
        return ""


# ── 本地安装状态 ──────────────────────────────────────────────────────────────

VERIFIED_MARKER = ".modelhub-verified"


def model_dir(models_root: str | Path, model_id: str) -> Path:
    return Path(models_root) / model_id


def is_installed(spec: ModelSpec, models_root: str | Path) -> bool:
    """标记文件 + 逐文件大小。

    不在这里重算 SHA256：十几 GB 要读一分钟，而 info 每几秒被问一次。完整校验
    只在下载完成那一刻做一次（下载器做的），做完写标记；标记里记着 revision，
    换了 revision 就自然算没装。
    """
    root = model_dir(models_root, spec.id)
    marker = root / VERIFIED_MARKER
    try:
        if json.loads(marker.read_text()).get("revision") != spec.revision:
            return False
        return all((root / name).is_file() and (root / name).stat().st_size == meta["size"]
                   for name, meta in spec.files.items())
    except (OSError, ValueError):
        return False


def mark_installed(spec: ModelSpec, models_root: str | Path) -> None:
    marker = model_dir(models_root, spec.id) / VERIFIED_MARKER
    marker.write_text(json.dumps({"repo": spec.repo, "revision": spec.revision}))


# ── 从 ModelScope 生成一行注册表 ──────────────────────────────────────────────

def _wanted(paths: list[str]) -> set[str]:
    has_safetensors = any(p.endswith(".safetensors") for p in paths)
    keep = set()
    for p in paths:
        base = p.rsplit("/", 1)[-1]
        if base in _SKIP_NAMES or base.upper().startswith(("README", "LICENSE", "NOTICE")):
            continue
        if p.lower().endswith(_SKIP_SUFFIXES):
            continue
        if has_safetensors and p.lower().endswith(_LEGACY_WEIGHT_SUFFIXES):
            continue
        if p.startswith(("original/", "figures/", "assets/", "imgs/", "images/")):
            continue
        keep.add(p)
    return keep


def pin_from_modelscope(repo: str, revision: str = "master", timeout: float = 20.0,
                        opener=urllib.request.urlopen) -> dict:
    """查 ModelScope 的文件列表，生成一行钉死 revision/size/SHA256 的注册表条目。

    分支名先解析成仓库 HEAD 的 commit（列表接口的 `LatestCommitter`），再按这个
    commit 重新列一遍文件，取每个文件的 Sha256。

    不能用列表里每个文件自带的 `Revision`：那是「最后改动这个文件的提交」，各文件
    各不相同。Qwen3-0.6B 上实测，拿 config.json 的 0ce186d9 当整仓版本，那个版本
    的 model.safetensors 是另一个文件（SHA256 a5aca0… 而不是 f47f71…）—— 钉出来的
    条目永远下载不成功，因为下到的每一份都过不了校验。
    """
    def listing(rev: str) -> dict:
        url = f"{MODELSCOPE_API}/{repo}/repo/files?Recursive=true&Revision={rev}"
        with opener(url, timeout=timeout) as resp:
            doc = json.load(resp)
        if doc.get("Code") not in (200, None) or not doc.get("Data"):
            raise ValueError(f"ModelScope 查不到 {repo}@{rev}：{doc.get('Message') or doc}")
        return doc["Data"]

    head = (listing(revision).get("LatestCommitter") or {})
    pinned_rev = head.get("Id") or head.get("ShortId")
    if not pinned_rev:
        raise ValueError(f"{repo}@{revision}: 解析不出 HEAD commit，无法钉死")
    blobs = [f for f in listing(pinned_rev).get("Files") or [] if f.get("Type") == "blob"]
    keep = _wanted([f["Path"] for f in blobs])
    files = {}
    for f in blobs:
        if f["Path"] not in keep:
            continue
        sha = str(f.get("Sha256") or "")
        if not _SHA256_RE.match(sha):
            raise ValueError(f"{repo}: {f['Path']} 没有 SHA256，无法钉死")
        files[f["Path"]] = {"size": int(f["Size"]), "sha256": sha}
    if not any(p.endswith(".safetensors") for p in files):
        raise ValueError(f"{repo}: 没有 safetensors 权重，vLLM 起不来这种格式")
    return {"source": {"modelscope": repo, "revision": pinned_rev},
            "files": dict(sorted(files.items()))}
