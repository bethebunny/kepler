import json
import sys

from .reporting import RichReporter
from .scope import Scope


DEFAULT_REPORTS = {}


def register_default_report(report: RichReporter):
    DEFAULT_REPORTS[report.event_type] = report


def report(name: str = "", scope: Scope | None = None):
    scope = scope or Scope.current
    all_event_types = set()
    for _, typed_events in scope.export():
        all_event_types |= typed_events.keys()

    for event_type in all_event_types:
        DEFAULT_REPORTS[event_type].report(name, scope)


if __name__ == "__main__":
    scope = Scope.from_json(json.load(sys.stdin))
    report(scope)
