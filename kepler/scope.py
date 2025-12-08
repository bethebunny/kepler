from __future__ import annotations

import collections
import contextvars
from typing import Iterator, MutableMapping, Sequence, TypeAlias

from .event import CallerID, CallStack, Event, TypedEvents


# From https://stackoverflow.com/a/76301341
class _classproperty:
    def __init__(self, func) -> None:
        self.fget = func

    def __get__(self, instance, owner):
        return self.fget(owner)


LogRow: TypeAlias = tuple[CallStack, TypedEvents]
Log: TypeAlias = Sequence[LogRow]


class Scope:
    def __init__(self):
        self.events: TypedEvents = collections.defaultdict(list)
        self.scopes: MutableMapping[CallerID, Scope] = collections.defaultdict(Scope)
        self._tokens: list[contextvars.Token[Scope]] = []

    @_classproperty
    def current(cls) -> Scope:
        return _CURRENT_SCOPE.get()

    def __getitem__(self, caller_id: CallerID) -> Scope:
        return self.scopes[caller_id]

    def __setitem__(self, caller_id: CallerID, scope: Scope):
        self.scopes[caller_id] = scope

    def __enter__(self):
        self._tokens.append(_CURRENT_SCOPE.set(self))
        return self

    def __exit__(self, *_):
        _CURRENT_SCOPE.reset(self._tokens.pop())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Scope):
            return False
        return self.events == other.events and self.scopes == other.scopes

    def log(self, event: Event):
        self.events[type(event)].append(event)

    def export(self) -> Iterator[LogRow]:
        yield (), self.events
        for caller_id, scope in self.scopes.items():
            for call_stack, events in scope.export():
                yield (caller_id, *call_stack), events

    def json(self):
        return [
            {
                "call_stack": [e.json() for e in call_stack],
                "events": {
                    event_type.__name__: [e.json() for e in events]
                    for event_type, events in typed_events.items()
                },
            }
            for call_stack, typed_events in self.export()
        ]

    @classmethod
    def from_json(cls, data: list):
        scope = cls()
        scopes_by_caller_id = collections.defaultdict(list)
        for row in data:
            if call_stack := row["call_stack"]:
                caller_id, *rest = call_stack
                caller_id = CallerID(**caller_id)
                scopes_by_caller_id[caller_id].append({**row, "call_stack": rest})
            else:
                scope.events.update(
                    {
                        (type := Event.TYPES[typename]): [type(**e) for e in events]
                        for typename, events in row["events"].items()
                    }
                )
        for caller_id, scope_data in scopes_by_caller_id.items():
            scope[caller_id] = cls.from_json(scope_data)
        return scope


def log(event: Event) -> None:
    Scope.current.log(event)


def import_events(scope: Scope):
    for event_type, events in scope.events.items():
        Scope.current.events[event_type].extend(events)
    for caller_id, subscope in scope.scopes.items():
        with Scope.current[caller_id]:
            import_events(subscope)


def scope(label: str):
    return Scope.current[CallerID.from_caller(label)]


_CURRENT_SCOPE = contextvars.ContextVar[Scope]("_CURRENT_SCOPE")
_CURRENT_SCOPE.set(Scope())
