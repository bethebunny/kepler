from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import numpy.typing as npt


CallStack: TypeAlias = list[str]


@dataclass
class Event:
    call_stack: CallStack
    times: npt.NDArray[np.float64]
