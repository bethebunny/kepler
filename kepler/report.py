import json
import sys

from .reporting import RichReporter
from .scope import Scope


def report(name: str = "", scope: Scope | None = None):
    from .reporting import RichReporter

    reporter = RichReporter(name)
    reporter.report(scope or Scope.current)


if __name__ == "__main__":
    scope = Scope.from_json(json.load(sys.stdin))
    reporter = RichReporter("stdin")
    reporter.report(scope)
