import json
import sys

from .context import Context
from .log import Log
from .reporting import RichReporter


def report(name: str = "", log: Log | None = None):
    from .reporting import RichReporter

    log = log or Log.from_events(Context.current.export())
    reporter = RichReporter(name)
    reporter.report(log)


if __name__ == "__main__":
    log = Log.from_json(json.load(sys.stdin))
    reporter = RichReporter("stdin")
    reporter.report(log)
