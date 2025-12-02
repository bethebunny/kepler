from dataclasses import dataclass
from pathlib import Path

import pytest

from kepler.context import Context
from kepler.event import Event
from kepler.log import Log
from kepler.timer import TimingEvent


@pytest.fixture
def context():
    with Context() as context:
        yield context


@pytest.fixture
def test_data():
    yield Path(__file__).parent / "data"


def assert_log_json_roundtrip(log: Log):
    """Assert that log structure is preserved when serialized to JSON and back"""
    assert Log.from_json(log.json()) == log


CallStackLabels = tuple[str, ...]


@dataclass
class LogStructure:
    event_counts: list[tuple[CallStackLabels, int]]


def log_structure(log: Log, event_type: type[Event] = TimingEvent) -> LogStructure:
    return LogStructure(
        event_counts=[
            (
                tuple(e.label for e in scoped_events.call_stack),
                len(scoped_events.events.get(event_type, ())),
            )
            for scoped_events in log.events
        ]
    )
