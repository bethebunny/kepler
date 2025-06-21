import json
import sys
from .event import Log
from .reporting import RichReporter


if __name__ == "__main__":
    log = Log.from_json(json.load(sys.stdin))
    reporter = RichReporter("stdin")
    reporter.report(log)
