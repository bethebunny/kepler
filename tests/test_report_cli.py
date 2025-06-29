from pathlib import Path
import sys
import subprocess

import pytest


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Decode error on Windows, haven't gotten a windows machine to debug",
)
def test_report_cli(test_data: Path):
    with open(test_data / "simple_log.json") as f:
        subprocess.check_call([sys.executable, "-m", "kepler.report"], stdin=f)
