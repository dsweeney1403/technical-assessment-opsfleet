import subprocess
import sys

import pytest

import analyst


def test_package_imports():
    assert analyst.__version__


@pytest.mark.slow
def test_runs_as_module():
    out = subprocess.run(
        [sys.executable, "-m", "analyst", "--help"], capture_output=True, text=True, check=True
    )
    assert "OpsFleet Analyst" in out.stdout
    assert "chat" in out.stdout and "doctor" in out.stdout
