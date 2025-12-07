from __future__ import annotations

from dataclasses import dataclass

from rich import console, table

from ..event import Event
from ..scope import Scope
from .statistic import Statistic


@dataclass
class RichReporter:
    title: str
    event_type: type[Event]
    statistics: tuple[Statistic, ...]

    def report(self, name: str, scope: Scope):
        # Report a table with metrics as column names, events as rows
        title = self.title.format(name=name)
        report = table.Table(
            title=title, row_styles=("", "on black"), title_style="white"
        )

        def all_events_of_type(event_type: type[Event]):
            for _, typed_events in scope.export():
                yield from typed_events[event_type]

        # Rather than orchestrating this metadata, it seems like statistics
        # really want to be computed and formatted as columns, and then
        # transposed it
        metadata = [
            statistic.metadata(scope, self.event_type) for statistic in self.statistics
        ]

        # XXX: remove this :P
        while not scope.events[self.event_type] and len(scope.scopes) == 1:
            scope = next(iter(scope.scopes.values()))

        summary = scope.events[self.event_type]
        report.show_footer = bool(summary)

        # Columns are metrics, plus "Stage" at the beginning for labels
        report.add_column("Stage", footer="Total" if summary else "", style="bold blue")

        for statistic, meta in zip(self.statistics, metadata):
            kwargs = {"justify": "right", **statistic.rich_args}
            footer = statistic.format(summary, meta) if summary else ""
            report.add_column(statistic.name, footer=footer, **kwargs)  # type: ignore

        def report_scope(label: str, scope: Scope, indent: int = 0):
            values = scope.events[self.event_type]
            cells = [
                statistic.format(values, meta)
                for statistic, meta in zip(self.statistics, metadata)
            ]
            report.add_row("  " * indent + label, *cells)
            for caller_id, subscope in scope.scopes.items():
                report_scope(caller_id.label, subscope, indent + 1)

        for caller_id, subscope in scope.scopes.items():
            report_scope(caller_id.label, subscope)

        console.Console().print(report)
