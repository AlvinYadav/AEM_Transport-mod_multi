

# Python std library
from enum import Enum


class ATElementType(Enum):
    """
    This enum class defines the various elemet types.

    - Circle: x, y and radius
    - Line: x1, y1 and x2, y2
    """

    Circle = 0
    Line = 1


class ATElement:
    def __init__(self, kind: ATElementType):
        self.kind = kind


