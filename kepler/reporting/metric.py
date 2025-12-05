from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Sequence, TypeVar

from .format import Formatter, FormatMetadata, Pretty

T = TypeVar("T")


@dataclass
class Metric(Generic[T]):
    name: str
    compute: Callable[[Sequence[float]], T]
    formatter: Formatter[T] = Pretty()
    rich_args: dict[str, Any] = field(default_factory=dict)

    def format(self, values: Sequence[float], meta: FormatMetadata):
        if not values:
            return ""
        value = self.compute(values)
        return self.formatter.format(value, meta)
