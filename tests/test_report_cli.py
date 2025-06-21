from pathlib import Path
import sys
import subprocess


def test_report_cli(test_data: Path):
    with open(test_data / "simple_log.json") as f:
        subprocess.check_call([sys.executable, "-m", "kepler.report"], stdin=f)
