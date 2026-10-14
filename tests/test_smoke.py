import subprocess
import sys

import analyst


def test_package_imports():
    assert analyst.__version__


def test_runs_as_module():
    out = subprocess.run(
        [sys.executable, "-m", "analyst"], capture_output=True, text=True, check=True
    )
    assert "OpsFleet Analyst" in out.stdout
