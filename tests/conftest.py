import collections
from pathlib import Path

import pytest

from kepler.event import Event
from kepler.scope import Scope


@pytest.fixture
def scope():
    with Scope() as scope:
        yield scope


@pytest.fixture
def test_data():
    yield Path(__file__).parent / "data"


def assert_log_json_roundtrip(scope: Scope):
    """Assert that log structure is preserved when serialized to JSON and back"""
    assert Scope.from_json(scope.json()) == scope


CallStackLabels = tuple[str, ...]


def event_counts(scope: Scope) -> dict[type[Event], list[tuple[tuple[str, ...], int]]]:
    typed_events = collections.defaultdict(list)
    for event_type, events in scope.events.items():
        if events:  # ignoring empty events for testing
            typed_events[event_type].append(((), len(events)))
    for caller_id, subscope in scope.scopes.items():
        for event_type, counts in event_counts(subscope).items():
            for call_stack, count in counts:
                typed_events[event_type].append(((caller_id.label, *call_stack), count))
    return typed_events
