"""镜像里带的 compose 片段（/deploy/service.yml）就是部署本身。

agent-core 和 run-pr-image.sh 都原样合进主机的 docker-compose.yml，这里少一行，
每台机器上就少一行。
"""

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

FRAGMENT = Path(__file__).resolve().parents[1] / "deploy" / "service.yml"


@pytest.fixture(scope="module")
def service():
    doc = yaml.safe_load(FRAGMENT.read_text())
    assert list(doc) == ["modelhub"]          # agent-core 取第一个 key 当服务名
    return doc["modelhub"]


def test_gpu_comes_in_through_the_nvidia_runtime(service):
    """驱动库只能用宿主机的，由 nvidia runtime 挂进来；和 perception/actucore 同一种写法。

    不用 CDI：星源智 N4（R38.4）上 nvidia runtime 是默认 runtime、没有 CDI 描述文件，
    CDI 设备请求在那里直接失败。
    """
    assert service["runtime"] == "nvidia"
    assert "deploy" not in service


def test_image_placeholder_and_container_name(service):
    assert service["image"] == "__IMAGE__"
    assert service["container_name"] == "embodied-modelhub"


def test_host_network_and_ipc(service):
    # host 网络：agent-core 用 127.0.0.1:15741 访问网关；ipc: vLLM worker 走共享内存
    assert service["network_mode"] == "host"
    assert service["ipc"] == "host"


def test_models_volume_and_restart(service):
    assert "/opt/embodied/models:/models" in service["volumes"]
    assert service["restart"] == "always"
    assert service["logging"] == {"driver": "local",
                                  "options": {"max-size": "10m", "max-file": "3"}}


def test_pid_1_reaps_orphans(service):
    """vLLM 的 EngineCore 被过继给 1 号进程；main.py 不 wait，每次换模型留一个僵尸。"""
    assert service["init"] is True
