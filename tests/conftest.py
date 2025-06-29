from pathlib import Path
import pytest

from kepler import Timer
from kepler.timer import TimerContext
from kepler.event import Log


@pytest.fixture
def timer():
    """Timer fixture that provides a timer context"""
    with (timer := Timer()).context:
        yield timer


@pytest.fixture
def context():
    with TimerContext() as context:
        yield context


@pytest.fixture
def test_data():
    yield Path(__file__).parent / "data"


def assert_log_json_roundtrip(log: Log):
    """Assert that log structure is preserved when serialized to JSON and back"""
    assert Log.from_json(log.json()) == log
