"""模型注册表：内置表本身合法、用户表增删、ModelScope 钉版本。"""

import io
import json
from pathlib import Path

import pytest

from serving.registry import (Registry, is_installed, mark_installed, model_dir,
                              pin_from_modelscope)

BUILTIN = Path(__file__).resolve().parents[1] / "models.yaml"
SHA = "a" * 64


def _entry(**over):
    entry = {"title": "T", "kind": "llm",
             "source": {"modelscope": "Org/Model", "revision": "abc1234"},
             "files": {"model.safetensors": {"size": 3, "sha256": SHA}}}
    entry.update(over)
    return entry


def test_builtin_registry_is_valid_and_pinned():
    models = Registry(BUILTIN, "/nonexistent.yaml").all()
    assert models, "内置表不能为空"
    for spec in models.values():
        assert spec.revision not in ("master", "main")
        assert any(name.endswith(".safetensors") for name in spec.files)
        # agent-core 每轮都带 tools；没有 parser，模型会把工具调用写进正文
        assert spec.serve.get("tool_call_parser"), spec.id
        assert spec.serve.get("memory_gb"), spec.id


def test_branch_revision_is_rejected(tmp_path):
    reg = Registry(BUILTIN, tmp_path / "u.yaml")
    with pytest.raises(ValueError, match="固定 commit"):
        reg.add_user_model("x", _entry(source={"modelscope": "Org/M", "revision": "master"}))


def test_bad_sha_and_escaping_path_are_rejected(tmp_path):
    reg = Registry(BUILTIN, tmp_path / "u.yaml")
    with pytest.raises(ValueError, match="sha256"):
        reg.add_user_model("x", _entry(files={"m.safetensors": {"size": 1, "sha256": "zz"}}))
    with pytest.raises(ValueError, match="越界"):
        reg.add_user_model("x", _entry(files={"../m.safetensors": {"size": 1, "sha256": SHA}}))


def test_user_models_round_trip_and_cannot_shadow_builtin(tmp_path):
    reg = Registry(BUILTIN, tmp_path / "u.yaml")
    reg.add_user_model("mine", _entry())
    assert reg.get("mine").builtin is False
    builtin_id = next(iter(Registry(BUILTIN, "/none").all()))
    with pytest.raises(ValueError, match="重名"):
        reg.add_user_model(builtin_id, _entry())
    reg.remove_user_model("mine")
    with pytest.raises(KeyError):
        reg.get("mine")


def test_installed_needs_marker_with_same_revision_and_sizes(tmp_path):
    reg = Registry(BUILTIN, tmp_path / "u.yaml")
    spec = reg.add_user_model("mine", _entry())
    d = model_dir(tmp_path, "mine")
    d.mkdir()
    (d / "model.safetensors").write_bytes(b"abc")
    assert not is_installed(spec, tmp_path)            # 没有标记
    mark_installed(spec, tmp_path)
    assert is_installed(spec, tmp_path)
    (d / "model.safetensors").write_bytes(b"ab")
    assert not is_installed(spec, tmp_path)            # 大小不对
    (d / "model.safetensors").write_bytes(b"abc")
    newer = reg.add_user_model("mine", _entry(source={"modelscope": "Org/Model",
                                                      "revision": "def5678"}))
    assert not is_installed(newer, tmp_path)           # 换了 revision


class _FakeModelScope:
    """按 Revision 返回不同的文件列表，复现「每个文件的 Revision 各不相同」。"""

    def __init__(self):
        self.urls = []

    def __call__(self, url, timeout=None):
        self.urls.append(url)
        rev = url.rsplit("Revision=", 1)[1]
        weights_sha = {"head1234": "1" * 64, "master": "1" * 64}.get(rev, "0" * 64)
        files = [
            # 文件自带的 Revision 是「最后改它的提交」：config 是旧提交 old0000
            {"Path": "config.json", "Type": "blob", "Size": 10, "Sha256": "c" * 64,
             "Revision": "old0000"},
            {"Path": "model.safetensors", "Type": "blob", "Size": 99, "Sha256": weights_sha,
             "Revision": "head1234"},
            {"Path": "README.md", "Type": "blob", "Size": 5, "Sha256": "d" * 64,
             "Revision": "old0000"},
            {"Path": "pytorch_model.bin", "Type": "blob", "Size": 99, "Sha256": "e" * 64,
             "Revision": "old0000"},
        ]
        doc = {"Code": 200, "Data": {"Files": files,
                                     "LatestCommitter": {"Id": "", "ShortId": "head1234"}}}
        return io.BytesIO(json.dumps(doc).encode())


def test_pin_uses_repo_head_not_a_files_own_revision():
    """回归：曾经拿第一个文件的 Revision 当整仓版本。

    Qwen3-0.6B 实测，那个旧提交下的 model.safetensors 是另一个文件，钉出来的条目
    永远过不了下载校验。
    """
    fake = _FakeModelScope()
    entry = pin_from_modelscope("Org/Model", "master", opener=fake)
    assert entry["source"]["revision"] == "head1234"
    assert entry["files"]["model.safetensors"]["sha256"] == "1" * 64
    assert fake.urls[-1].endswith("Revision=head1234")    # 文件列表是按 HEAD 取的
    assert set(entry["files"]) == {"config.json", "model.safetensors"}  # 文档、旧格式权重不要
