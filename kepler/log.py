from dataclasses import dataclass
from typing import Iterable

from .context import Context
from .event import ScopedEvents


@dataclass
class Log:
    events: list[ScopedEvents]

    @classmethod
    def from_context(cls, context: Context):
        return cls.from_events(context.export())

    @classmethod
    def from_current_context(cls):
        return cls.from_context(Context.current)

    @classmethod
    def from_events(cls, events: Iterable[ScopedEvents]):
        return cls(events=list(events))

    def json(self):
        return [e.json() for e in self.events]

    @classmethod
    def from_json(cls, data: list[dict]):
        return cls(events=[ScopedEvents.from_json(e) for e in data])
