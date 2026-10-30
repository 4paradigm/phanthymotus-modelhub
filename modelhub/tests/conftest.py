"""modelhub 测试公共设置：把 modelhub/ 放上 sys.path（镜像里它就是 /work）。

Run: cd modelhub && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests -q
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
