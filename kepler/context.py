from __future__ import annotations

import collections
import contextvars
from typing import Iterable, MutableMapping

from .event import CallerID, Event, ExportContext, ScopedEvents


class Scope:
    def __init__(self):
        self.events: MutableMapping[type[Event], list[Event]] = collections.defaultdict(
            list
        )
        self.context = Context()

    def log(self, event: Event):
        self.events[type(event)].append(event)

    def export(self, ctx: ExportContext | None = None) -> Iterable[ScopedEvents]:
        ctx = ctx or ExportContext()
        yield ScopedEvents(
            call_stack=(),
            events={
                type: [e.export(ctx) for e in events]
                for type, events in self.events.items()
            },
        )
        yield from self.context.export(ctx)


# From https://stackoverflow.com/a/76301341
class _classproperty:
    def __init__(self, func) -> None:
        self.fget = func

    def __get__(self, instance, owner):
        return self.fget(owner)


class Context:
    @_classproperty
    def current(cls) -> Context:
        return _CURRENT_CONTEXT.get()

    def __init__(self):
        self.scopes: MutableMapping[CallerID, Scope] = collections.defaultdict(Scope)
        self._tokens: list[contextvars.Token[Context]] = []

    def __getitem__(self, caller_id: CallerID) -> Scope:
        return self.scopes[caller_id]

    def __enter__(self):
        self._tokens.append(_CURRENT_CONTEXT.set(self))
        return self

    def __exit__(self, *_):
        _CURRENT_CONTEXT.reset(self._tokens.pop())

    def export(self, ctx: ExportContext | None = None) -> Iterable[ScopedEvents]:
        ctx = ctx or ExportContext()
        for caller_id, scope in self.scopes.items():
            for scoped_events in scope.export(ctx):
                yield scoped_events.nest_under(caller_id)


_CURRENT_CONTEXT = contextvars.ContextVar[Context]("_CURRENT_CONTEXT")
_CURRENT_CONTEXT.set(Context())
