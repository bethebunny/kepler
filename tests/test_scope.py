import json
import sys
from contextlib import ExitStack
from datetime import datetime, timedelta
from pathlib import Path

import pytest

import kepler
from kepler import stopwatch
from kepler.scope import Scope
from kepler.timer import TimingEvent

from .conftest import assert_log_json_roundtrip, event_counts


def optional_scope(ctx):
    context_manager = ExitStack()
    if ctx is not None:
        context_manager.enter_context(ctx)
    return context_manager


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Decode error on Windows, haven't gotten a windows machine to debug",
)
def test_simple_log(test_data: Path):
    with open(test_data / "simple_log.json") as f:
        scope = Scope.from_json(json.load(f))

    assert_log_json_roundtrip(scope)


def test_nested_scopes(scope: Scope):
    """Test scopes nested within other scopes"""

    with scope:
        with kepler.time("outer"):
            with kepler.time("inner"):
                pass

    assert_log_json_roundtrip(scope)

    assert event_counts(scope) == {
        TimingEvent: [
            (("outer",), 1),
            (("outer", "inner"), 1),
        ],
    }


def test_nested_functions(scope: Scope):
    """Test function decorators nested within each other"""

    @kepler.time("outer")
    def outer():
        inner()

    @kepler.time("inner")
    def inner():
        pass

    with scope:
        outer()

    assert_log_json_roundtrip(scope)

    assert event_counts(scope) == {
        TimingEvent: [
            (("outer",), 1),
            (("outer", "inner"), 1),
        ]
    }


def test_simple_scope(scope: Scope):
    """Test simple single scope"""

    with scope:
        with kepler.time("simple"):
            pass

    assert_log_json_roundtrip(scope)

    assert event_counts(scope) == {
        TimingEvent: [
            (("simple",), 1),
        ]
    }


def test_iter(scope: Scope):
    with scope:
        for _ in kepler.time("range", range(20)):
            pass

    assert_log_json_roundtrip(scope)

    assert event_counts(scope) == {
        TimingEvent: [
            (("range",), 20),
        ]
    }


def test_two_separate_scopes(scope: Scope):
    """Test two separate scopes"""

    with scope:
        with kepler.time("first"):
            pass
        with kepler.time("second"):
            pass

    assert_log_json_roundtrip(scope)

    assert event_counts(scope) == {
        TimingEvent: [
            (("first",), 1),
            (("second",), 1),
        ]
    }


def test_function_nested_within_scope(scope: Scope):
    """Test function decorator used inside a scope"""

    @kepler.time("inner")
    def inner():
        pass

    with scope:
        with kepler.time("scope"):
            inner()

    assert_log_json_roundtrip(scope)

    assert event_counts(scope) == {
        TimingEvent: [
            (("scope",), 1),
            (("scope", "inner"), 1),
        ]
    }


def test_function_nested_within_conditional_scope(
    scope: Scope,
):
    """Test function decorator used inside a conditional scope"""

    @kepler.time("inner")
    def inner():
        pass

    with scope:
        for enabled in [True, False]:
            with optional_scope(kepler.time("conditional") if enabled else None):
                inner()

    assert_log_json_roundtrip(scope)

    assert event_counts(scope) == {
        TimingEvent: [
            (("conditional",), 1),  # condition enabled
            (("conditional", "inner"), 1),  # inner in enabled conditional
            (("inner",), 1),  # inner in disabled conditional
        ]
    }


def test_scope_nested_within_function(scope: Scope):
    """Test scope manager used inside a function decorator"""

    @kepler.time("outer")
    def outer():
        with kepler.time("inner"):
            pass

    with scope:
        outer()

    assert_log_json_roundtrip(scope)

    assert event_counts(scope) == {
        TimingEvent: [
            (("outer",), 1),
            (("outer", "inner"), 1),
        ]
    }


def test_conditional_scope_nested_within_function(
    scope: Scope,
):
    """Test conditional scope nested within function"""

    @kepler.time("outer")
    def outer(enabled: bool):
        with optional_scope(kepler.time("conditional_inner") if enabled else None):
            pass

    with scope:
        outer(True)  # Creates conditional scope
        outer(False)  # No conditional scope

    assert_log_json_roundtrip(scope)

    assert event_counts(scope) == {
        TimingEvent: [
            (("outer",), 2),
            (("outer", "conditional_inner"), 1),
        ]
    }


def test_two_functions_with_same_label(scope: Scope):
    """Test two different functions that resolve to the same label"""

    @kepler.time("function")
    def f1():
        pass

    @kepler.time("function")
    def f2():
        pass

    with scope:
        f1()
        f2()

    assert_log_json_roundtrip(scope)

    assert event_counts(scope) == {
        TimingEvent: [
            (("function",), 1),
            (("function",), 1),
        ]
    }


def test_two_scopes_with_same_label(scope: Scope):
    """Test two scopes with the same label"""

    with scope:
        with kepler.time("same_label"):
            pass
        with kepler.time("same_label"):
            pass

    assert_log_json_roundtrip(scope)

    assert event_counts(scope) == {
        TimingEvent: [
            (("same_label",), 1),
            (("same_label",), 1),
        ]
    }


def test_function_and_scope_with_same_label(scope: Scope):
    """Test function and scope with the same label"""

    @kepler.time("shared_label")
    def shared_label():
        pass

    with scope:
        shared_label()
        with kepler.time("shared_label"):
            pass

    assert_log_json_roundtrip(scope)

    assert event_counts(scope) == {
        TimingEvent: [
            (("shared_label",), 1),
            (("shared_label",), 1),
        ]
    }


def test_recursive_function(scope: Scope):
    """Test recursive function with timing"""

    @kepler.time("recursive")
    def recursive(n):
        assert n >= 0
        if n > 0:
            recursive(n - 1)

    with scope:
        recursive(3)

    assert_log_json_roundtrip(scope)

    assert event_counts(scope) == {
        TimingEvent: [
            (("recursive",), 1),
            (("recursive", "recursive"), 1),
            (("recursive", "recursive", "recursive"), 1),
            (("recursive", "recursive", "recursive", "recursive"), 1),
        ]
    }


def test_mutually_recursive_functions(scope: Scope):
    """Test mutually recursive functions"""

    @kepler.time("f")
    def f(n):
        g(n)

    @kepler.time("g")
    def g(n):
        assert n >= 0
        if n > 0:
            f(n - 1)

    with scope:
        f(2)

    assert_log_json_roundtrip(scope)

    assert event_counts(scope) == {
        TimingEvent: [
            (("f",), 1),
            (("f", "g"), 1),
            (("f", "g", "f"), 1),
            (("f", "g", "f", "g"), 1),
            (("f", "g", "f", "g", "f"), 1),
            (("f", "g", "f", "g", "f", "g"), 1),
        ]
    }


def test_recursive_function_with_scope(scope: Scope):
    """Test recursive function that uses scopes internally"""

    @kepler.time("recursive")
    def recursive(n):
        assert n >= 0
        with kepler.time("inner"):
            if n > 0:
                recursive(n - 1)

    with scope:
        recursive(2)

    assert_log_json_roundtrip(scope)

    assert event_counts(scope) == {
        TimingEvent: [
            (("recursive",), 1),
            (("recursive", "inner"), 1),
            (("recursive", "inner", "recursive"), 1),
            (("recursive", "inner", "recursive", "inner"), 1),
            (("recursive", "inner", "recursive", "inner", "recursive"), 1),
            (
                (
                    "recursive",
                    "inner",
                    "recursive",
                    "inner",
                    "recursive",
                    "inner",
                ),
                1,
            ),
        ]
    }


def test_nested_scopes_with_same_label(scope: Scope):
    """Test nested scopes with the same label"""

    with scope:
        with kepler.time("nested"):
            with kepler.time("nested"):
                pass

    assert_log_json_roundtrip(scope)

    assert event_counts(scope) == {
        TimingEvent: [
            (("nested",), 1),
            (("nested", "nested"), 1),
        ]
    }


def test_nested_conditional_scopes_with_same_label(
    scope: Scope,
):
    """Test nested conditional scopes with same label"""

    def nested_conditional_scope(outer_enabled: bool, inner_enabled: bool):
        with optional_scope(kepler.time("conditional") if outer_enabled else None):
            with optional_scope(kepler.time("conditional") if inner_enabled else None):
                pass

    with scope:
        nested_conditional_scope(outer_enabled=True, inner_enabled=True)
        nested_conditional_scope(outer_enabled=True, inner_enabled=False)
        nested_conditional_scope(outer_enabled=False, inner_enabled=True)
        nested_conditional_scope(outer_enabled=False, inner_enabled=False)

    assert_log_json_roundtrip(scope)

    assert event_counts(scope) == {
        TimingEvent: [
            (("conditional",), 2),  # outer enabled, inner either enabled or not
            (("conditional", "conditional"), 1),  # outer enabled, inner enabled
            (("conditional",), 1),  # outer disabled, inner enabled
        ]
    }


def test_stopwatch_splits(scope: Scope):
    """Test basic stopwatch functionality"""

    with scope:
        split = stopwatch("watch")
        split("start")
        split("middle")
        split("end")

    assert_log_json_roundtrip(scope)

    assert event_counts(scope) == {
        TimingEvent: [
            ((":stopwatch: watch", "start"), 1),
            ((":stopwatch: watch", "middle"), 1),
            ((":stopwatch: watch", "end"), 1),
        ]
    }


def test_stopwatch_splits_with_same_label(scope: Scope):
    """Test stopwatch splits with same label"""

    with scope:
        split = stopwatch("watch")
        split("same")
        split("same")
        split("same")

    assert_log_json_roundtrip(scope)

    assert event_counts(scope) == {
        TimingEvent: [
            ((":stopwatch: watch", "same"), 1),
            ((":stopwatch: watch", "same"), 1),
            ((":stopwatch: watch", "same"), 1),
        ]
    }


def test_stopwatch_splits_with_same_label_as_scope(
    scope: Scope,
):
    """Test stopwatch split label same as scope label"""

    with scope:
        with kepler.time("shared"):
            pass
        split = stopwatch("watch")
        split("shared")

    assert_log_json_roundtrip(scope)

    assert event_counts(scope) == {
        TimingEvent: [
            (("shared",), 1),
            ((":stopwatch: watch", "shared"), 1),
        ]
    }


def test_stopwatch_splits_with_same_label_as_function(
    scope: Scope,
):
    """Test stopwatch split label same as function label"""

    @kepler.time("shared")
    def shared():
        pass

    with scope:
        shared()
        split = stopwatch("watch")
        split("shared")

    assert_log_json_roundtrip(scope)

    assert event_counts(scope) == {
        TimingEvent: [
            (("shared",), 1),
            ((":stopwatch: watch", "shared"), 1),
        ]
    }


def test_stopwatch_splits_with_conditional_scope(scope: Scope):
    """Test stopwatch with conditional scope"""

    def stopwatch_with_condition(enabled: bool):
        split = stopwatch("watch")
        with optional_scope(kepler.time("conditional") if enabled else None):
            split("inside_scope")
        split("outside_scope")

    with scope:
        stopwatch_with_condition(enabled=True)
        stopwatch_with_condition(enabled=False)

    assert_log_json_roundtrip(scope)

    assert event_counts(scope) == {
        TimingEvent: [
            ((":stopwatch: watch", "inside_scope"), 2),
            ((":stopwatch: watch", "outside_scope"), 2),
            (("conditional",), 1),
        ]
    }


def test_log_timestamps_use_system_time(scope: Scope):
    """Test log timestamps use system time"""

    with scope:
        with kepler.time("test"):
            pass

    now = datetime.now()
    for _, typed_events in scope.export():
        for event in typed_events[TimingEvent]:
            assert isinstance(event, TimingEvent)
            ts = datetime.fromtimestamp(event.timestamp / 1e9)
            # Timers aren't precise enough to guarantee this; it's true
            # in principle but sometimes fails in practice.
            # assert now > ts
            # The `abs()` check just checks that the recorded timestamp is
            # ~close to system time.
            assert abs(now - ts) < timedelta(seconds=1)


def test_import_events(scope: Scope):
    with Scope() as inner_scope:
        kepler.log(TimingEvent(0, 0))
        with kepler.time("inner"):
            pass

    with scope:
        with kepler.time("outer"):
            kepler.scope.import_events(inner_scope)

    assert_log_json_roundtrip(scope)

    assert event_counts(scope) == {
        TimingEvent: [
            (("outer",), 2),
            (("outer", "inner"), 1),
        ],
    }
