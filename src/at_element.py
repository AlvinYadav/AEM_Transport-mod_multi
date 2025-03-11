# Written by Willi Kappler, willi.kappler@uni-tuebingen.de
# Based on code from Anton Köhler



# Python std library
import math
from enum import Enum
from abc import ABC, abstractmethod


class ATElementType(Enum):
    """
    This enum class defines the various elemet types.

    - Circle: x, y and radius
    - Line: x1, y1 and x2, y2
    """

    Circle = 0
    Line = 1


class ATElement(ABC):
    def __init__(self, kind: ATElementType):
        self.kind = kind

    @abstractmethod
    def is_inside(self, x, y) -> bool:
        pass

    @abstractmethod
    def c_inside(self) -> float:
        pass

class ATECircle(ATElement):
    def __init__(self, c: float, x: float, y: float, r: float):
        super().__init__(ATElementType.Circle)
        self.c: float = c # Concentration
        self.x: float = x
        self.y: float = y
        self.r: float = r

        # Calculated parameters:
        self.q = math.nan
        self.d = math.nan
        self.outline = []
        self.target = []
        self.m_list = []
        self.f_list = []

    def calc_d_q(self, alpha_t: float, alpha_l: float, beta: float):
        #d = np.sqrt((r * np.sqrt(alpha_l / alpha_t))**2 - r**2)
        #q = (d**2 * beta**2) / 4
        self.d = math.sqrt(self.r * math.sqrt(alpha_l / alpha_t)**2.0 - self.r**2.0)
        self.q = (self.d**2.0 * beta**2.0) / 4.0

    def set_outline(self, num_cp: int):
        step: float = 2.0 * math.pi / float(num_cp)

        self.outline = []

        for i in range(0, num_cp):
            alpha = float(i) * step
            x = self.r * math.cos(alpha)
            y = self.r * math.sin(alpha)
            self.outline.append((x, y))

    def calc_target(self, ca: float, beta: float, gamma: float):
        #if Ci > 0:
        #    return (Ci*gamma+Ca)*np.exp(-beta*x), Ci, 'r'
        #else:
        #    return (Ci)*np.exp(-beta*x), Ci, 'b'

        f1: float = self.c * gamma + ca if self.c > 0.0 else self.c

        for (x, _) in self.outline:
            f2: float = math.exp(-beta * (x + self.x))
            self.target.append(f1 * f2)

    def is_inside(self, x, y) -> bool:
        d = math.hypot(x - self.x, y - self.y)
        return d <= self.r

    def c_inside(self) -> float:
        return self.c

class ATELine(ATElement):
    def __init__(self, x1: float, y1: float, x2: float, y2: float):
        super().__init__(ATElementType.Line)
        self.x1: float = x1
        self.y1: float = y1
        self.x2: float = x2
        self.y2: float = y2



