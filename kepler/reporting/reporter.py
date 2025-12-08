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

        metadata = [
            statistic.metadata(scope, self.event_type) for statistic in self.statistics
        ]

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
