from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from pathlib import Path
import pytest
import sys

import kepler
from kepler import stopwatch
from kepler import timer
from kepler.event import Log
from .conftest import assert_log_json_roundtrip


def captured_log(context: timer.TimerContext) -> Log:
    """Capture current timing log."""
    return Log.from_events(context.export())


def optional_context(ctx):
    context_manager = ExitStack()
    if ctx is not None:
        context_manager.enter_context(ctx)
    return context_manager


CallStackLabels = tuple[str, ...]


@dataclass
class LogStructure:
    event_counts: list[tuple[CallStackLabels, int]]


def log_structure(log: Log) -> LogStructure:
    return LogStructure(
        event_counts=[
            (tuple(e.label for e in event.call_stack), len(event.events))
            for event in log.events
        ]
    )


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Decode error on Windows, haven't gotten a windows machine to debug",
)
def test_simple_log(test_data: Path):
    with open(test_data / "simple_log.json") as f:
        log = Log.from_json(json.load(f))

    assert_log_json_roundtrip(log)


def test_nested_contexts(context: timer.TimerContext):
    """Test contexts nested within other contexts"""

    with context:
        with kepler.time("outer"):
            with kepler.time("inner"):
                pass

    log = captured_log(context)
    assert_log_json_roundtrip(log)

    assert log_structure(log) == LogStructure(
        event_counts=[
            (("outer",), 1),
            (("outer", "inner"), 1),
        ]
    )


def test_nested_functions(context: timer.TimerContext):
    """Test function decorators nested within each other"""

    @kepler.time("outer")
    def outer():
        inner()

    @kepler.time("inner")
    def inner():
        pass

    with context:
        outer()

    log = captured_log(context)
    assert_log_json_roundtrip(log)

    assert log_structure(log) == LogStructure(
        event_counts=[
            (("outer",), 1),
            (("outer", "inner"), 1),
        ]
    )


def test_simple_context(context: timer.TimerContext):
    """Test simple single context"""

    with context:
        with kepler.time("simple"):
            pass

    log = captured_log(context)
    assert_log_json_roundtrip(log)

    assert log_structure(log) == LogStructure(
        event_counts=[
            (("simple",), 1),
        ]
    )


def test_two_separate_contexts(context: timer.TimerContext):
    """Test two separate contexts"""

    with context:
        with kepler.time("first"):
            pass
        with kepler.time("second"):
            pass

    log = captured_log(context)
    assert_log_json_roundtrip(log)

    assert log_structure(log) == LogStructure(
        event_counts=[
            (("first",), 1),
            (("second",), 1),
        ]
    )


def test_function_nested_within_context(context: timer.TimerContext):
    """Test function decorator used inside a context"""

    @kepler.time("inner")
    def inner():
        pass

    with context:
        with kepler.time("context"):
            inner()

    log = captured_log(context)
    assert_log_json_roundtrip(log)

    assert log_structure(log) == LogStructure(
        event_counts=[
            (("context",), 1),
            (("context", "inner"), 1),
        ]
    )


def test_function_nested_within_conditional_context(
    context: timer.TimerContext,
):
    """Test function decorator used inside a conditional context"""

    @kepler.time("inner")
    def inner():
        pass

    with context:
        for enabled in [True, False]:
            with optional_context(kepler.time("conditional") if enabled else None):
                inner()

    log = captured_log(context)
    assert_log_json_roundtrip(log)

    assert log_structure(log) == LogStructure(
        event_counts=[
            (("conditional",), 1),  # condition enabled
            (("conditional", "inner"), 1),  # inner in enabled conditional
            (("inner",), 1),  # inner in disabled conditional
        ]
    )


def test_context_nested_within_function(context: timer.TimerContext):
    """Test context manager used inside a function decorator"""

    @kepler.time("outer")
    def outer():
        with kepler.time("inner"):
            pass

    with context:
        outer()

    log = captured_log(context)
    assert_log_json_roundtrip(log)

    assert log_structure(log) == LogStructure(
        event_counts=[
            (("outer",), 1),
            (("outer", "inner"), 1),
        ]
    )


def test_conditional_context_nested_within_function(
    context: timer.TimerContext,
):
    """Test conditional context nested within function"""

    @kepler.time("outer")
    def outer(enabled: bool):
        with optional_context(kepler.time("conditional_inner") if enabled else None):
            pass

    with context:
        outer(True)  # Creates conditional context
        outer(False)  # No conditional context

    log = captured_log(context)
    assert_log_json_roundtrip(log)

    assert log_structure(log) == LogStructure(
        event_counts=[
            (("outer",), 2),
            (("outer", "conditional_inner"), 1),
        ]
    )


def test_two_functions_with_same_label(context: timer.TimerContext):
    """Test two different functions that resolve to the same label"""

    @kepler.time("function")
    def f1():
        pass

    @kepler.time("function")
    def f2():
        pass

    with context:
        f1()
        f2()

    log = captured_log(context)
    assert_log_json_roundtrip(log)

    assert log_structure(log) == LogStructure(
        event_counts=[
            (("function",), 1),
            (("function",), 1),
        ]
    )


def test_two_contexts_with_same_label(context: timer.TimerContext):
    """Test two contexts with the same label"""

    with context:
        with kepler.time("same_label"):
            pass
        with kepler.time("same_label"):
            pass

    log = captured_log(context)
    assert_log_json_roundtrip(log)

    assert log_structure(log) == LogStructure(
        event_counts=[
            (("same_label",), 1),
            (("same_label",), 1),
        ]
    )


def test_function_and_context_with_same_label(context: timer.TimerContext):
    """Test function and context with the same label"""

    @kepler.time("shared_label")
    def shared_label():
        pass

    with context:
        shared_label()
        with kepler.time("shared_label"):
            pass

    log = captured_log(context)
    assert_log_json_roundtrip(log)

    assert log_structure(log) == LogStructure(
        event_counts=[
            (("shared_label",), 1),
            (("shared_label",), 1),
        ]
    )


def test_recursive_function(context: timer.TimerContext):
    """Test recursive function with timing"""

    @kepler.time("recursive")
    def recursive(n):
        assert n >= 0
        if n > 0:
            recursive(n - 1)

    with context:
        recursive(3)

    log = captured_log(context)
    assert_log_json_roundtrip(log)

    assert log_structure(log) == LogStructure(
        event_counts=[
            (("recursive",), 1),
            (("recursive", "recursive"), 1),
            (("recursive", "recursive", "recursive"), 1),
            (("recursive", "recursive", "recursive", "recursive"), 1),
        ]
    )


def test_mutually_recursive_functions(context: timer.TimerContext):
    """Test mutually recursive functions"""

    @kepler.time("f")
    def f(n):
        g(n)

    @kepler.time("g")
    def g(n):
        assert n >= 0
        if n > 0:
            f(n - 1)

    with context:
        f(2)

    log = captured_log(context)
    assert_log_json_roundtrip(log)

    assert log_structure(log) == LogStructure(
        event_counts=[
            (("f",), 1),
            (("f", "g"), 1),
            (("f", "g", "f"), 1),
            (("f", "g", "f", "g"), 1),
            (("f", "g", "f", "g", "f"), 1),
            (("f", "g", "f", "g", "f", "g"), 1),
        ]
    )


def test_recursive_function_with_context(context: timer.TimerContext):
    """Test recursive function that uses contexts internally"""

    @kepler.time("recursive")
    def recursive(n):
        assert n >= 0
        with kepler.time("inner"):
            if n > 0:
                recursive(n - 1)

    with context:
        recursive(2)

    log = captured_log(context)
    assert_log_json_roundtrip(log)

    assert log_structure(log) == LogStructure(
        event_counts=[
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
    )


def test_nested_contexts_with_same_label(context: timer.TimerContext):
    """Test nested contexts with the same label"""

    with context:
        with kepler.time("nested"):
            with kepler.time("nested"):
                pass

    log = captured_log(context)
    assert_log_json_roundtrip(log)

    assert log_structure(log) == LogStructure(
        event_counts=[
            (("nested",), 1),
            (("nested", "nested"), 1),
        ]
    )


def test_nested_conditional_contexts_with_same_label(
    context: timer.TimerContext,
):
    """Test nested conditional contexts with same label"""

    def nested_conditional_context(outer_enabled: bool, inner_enabled: bool):
        with optional_context(kepler.time("conditional") if outer_enabled else None):
            with optional_context(
                kepler.time("conditional") if inner_enabled else None
            ):
                pass

    with context:
        nested_conditional_context(outer_enabled=True, inner_enabled=True)
        nested_conditional_context(outer_enabled=True, inner_enabled=False)
        nested_conditional_context(outer_enabled=False, inner_enabled=True)
        nested_conditional_context(outer_enabled=False, inner_enabled=False)

    log = captured_log(context)
    assert_log_json_roundtrip(log)

    assert log_structure(log) == LogStructure(
        event_counts=[
            (("conditional",), 2),  # outer enabled, inner either enabled or not
            (("conditional", "conditional"), 1),  # outer enabled, inner enabled
            (("conditional",), 1),  # outer disabled, inner enabled
        ]
    )


def test_stopwatch_splits(context: timer.TimerContext):
    """Test basic stopwatch functionality"""

    with context:
        split = stopwatch("watch")
        split("start")
        split("middle")
        split("end")

    log = captured_log(context)
    assert_log_json_roundtrip(log)

    assert log_structure(log) == LogStructure(
        event_counts=[
            ((":stopwatch: watch", "start"), 1),
            ((":stopwatch: watch", "middle"), 1),
            ((":stopwatch: watch", "end"), 1),
        ]
    )


def test_stopwatch_splits_with_same_label(context: timer.TimerContext):
    """Test stopwatch splits with same label"""

    with context:
        split = stopwatch("watch")
        split("same")
        split("same")
        split("same")

    log = captured_log(context)
    assert_log_json_roundtrip(log)

    assert log_structure(log) == LogStructure(
        event_counts=[
            ((":stopwatch: watch", "same"), 1),
            ((":stopwatch: watch", "same"), 1),
            ((":stopwatch: watch", "same"), 1),
        ]
    )


def test_stopwatch_splits_with_same_label_as_context(
    context: timer.TimerContext,
):
    """Test stopwatch split label same as context label"""

    with context:
        with kepler.time("shared"):
            pass
        split = stopwatch("watch")
        split("shared")

    log = captured_log(context)
    assert_log_json_roundtrip(log)

    assert log_structure(log) == LogStructure(
        event_counts=[
            (("shared",), 1),
            ((":stopwatch: watch", "shared"), 1),
        ]
    )


def test_stopwatch_splits_with_same_label_as_function(
    context: timer.TimerContext,
):
    """Test stopwatch split label same as function label"""

    @kepler.time("shared")
    def shared():
        pass

    with context:
        shared()
        split = stopwatch("watch")
        split("shared")

    log = captured_log(context)
    assert_log_json_roundtrip(log)

    assert log_structure(log) == LogStructure(
        event_counts=[
            (("shared",), 1),
            ((":stopwatch: watch", "shared"), 1),
        ]
    )


def test_stopwatch_splits_with_conditional_context(context: timer.TimerContext):
    """Test stopwatch with conditional context"""

    def stopwatch_with_condition(enabled: bool):
        split = stopwatch("watch")
        with optional_context(kepler.time("conditional") if enabled else None):
            split("inside_context")
        split("outside_context")

    with context:
        stopwatch_with_condition(enabled=True)
        stopwatch_with_condition(enabled=False)

    log = captured_log(context)
    assert_log_json_roundtrip(log)

    assert log_structure(log) == LogStructure(
        event_counts=[
            (("conditional",), 1),
            ((":stopwatch: watch", "inside_context"), 2),
            ((":stopwatch: watch", "outside_context"), 2),
        ]
    )


def test_log_timestamps_use_system_time(context: timer.TimerContext):
    """Test log timestamps use system time"""

    with context:
        with kepler.time("test"):
            pass

    log = captured_log(context)
    now = datetime.now()
    for scoped_events in log.events:
        for event in scoped_events.events:
            ts = datetime.fromtimestamp(event.timestamp / 1e9)
            assert ts - now < timedelta(seconds=1)
