from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Sequence, TypeVar


from .format import Formatter, Measurement, Metadata, Pretty, Stat
from ..event import Event
from ..scope import Scope

EventType = TypeVar("EventType", bound=type[Event])


@dataclass
class Statistic(Generic[EventType, Measurement, Stat]):
    name: str
    compute: Callable[[Sequence[Measurement]], Stat]
    measure: Callable[[EventType], Measurement] = lambda x: x
    formatter: Formatter[Stat] = Pretty()
    rich_args: dict[str, Any] = field(default_factory=dict)

    def metadata(self, scope: Scope, event_type: EventType) -> Metadata[Stat]:
        measurements = [
            [self.measure(event) for event in events[event_type]]
            for _, events in scope.export()
        ]
        return Metadata([self.compute(m) for m in measurements if m])

    def format(self, events: Sequence[EventType], meta: Metadata[Stat]):
        if not events:
            return ""
        value = self.compute([self.measure(event) for event in events])
        return self.formatter.format(value, meta)
