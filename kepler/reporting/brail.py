import itertools
from typing import Iterable, TypeVar

# Per https://en.wikipedia.org/wiki/Braille_Patterns
# the hex values of Brail unicode codes begin at 0x2800, and each
# of the 8 dots is present depending on the bit value of the final byte:
#
#   +-------------+
#   | 0x01 | 0x08 |
#   +-------------+
#   | 0x02 | 0x10 |
#   +-------------+
#   | 0x04 | 0x20 |
#   +-------------+
#   | 0x40 | 0x80 |
#   +-------------+


def brail_bar_chr(left: int, right: int = 0):
    # Produces brail characters for which the left and right halves
    # are considered vertical "bars" of a given height, starting at the bottom
    # of the character.
    if not (0 <= left <= 4 and 0 <= right <= 4):
        raise ValueError("Brail bar data must be in range [0, 4]")
    left_offset: int = [0, 0x40, 0x44, 0x46, 0x47][left]
    right_offset: int = [0, 0x80, 0xA0, 0xB0, 0xB8][right]
    return chr(0x2800 + left_offset + right_offset)


def brail_bars(data: Iterable[int]) -> str:
    return "".join(brail_bar_chr(*pair) for pair in _batched(data, 2))


T = TypeVar("T")


def _batched(it: Iterable[T], n: int):
    # Recipe taken from Python itertools, added in 3.12
    # batched('ABCDEFG', 3) → ABC DEF G
    it = iter(it)
    while batch := tuple(itertools.islice(it, n)):
        yield batch
