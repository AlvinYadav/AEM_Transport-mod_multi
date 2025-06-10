import math
import numpy as np
from enum import Enum
from mathieu_functions_OG import Mathieu


class ATElementType(Enum):
    Circle = 0


class ATElement:
    def __init__(self, kind: ATElementType, x: float, y: float, c: float, r = 1.0):
        self.kind = kind
        self.x = x
        self.y = y
        self.c = c
        self.r = r
        self.d = None
        self.q = None
        self.outline = []
        self.m = None

    def calc_d_q(self, alpha_t, alpha_l, beta):
        r = self.r
        self.d = math.sqrt((r * math.sqrt(alpha_l / alpha_t)) ** 2 - r ** 2)
        self.q = (self.d ** 2 * beta ** 2) / 4
        self.m = Mathieu(self.q)

    def set_outline(self, num_cp):
        phi = np.linspace(0, 2 * np.pi, num_cp, endpoint=False)
        r = self.r
        self.outline = [(self.x + r * np.cos(p), self.y + r * np.sin(p)) for p in phi]

    def uv(self, x, y, alpha_l, alpha_t):
        d = self.d
        Y = math.sqrt(alpha_l / alpha_t) * y
        B = x**2 + Y**2 - d**2
        sqrt_term = math.sqrt(B**2 + 4 * d**2 * x**2)
        p = (-B + sqrt_term) / (2 * d**2)
        q = (-B - sqrt_term) / (2 * d**2)

        psi_0 = math.asin(min(max(math.sqrt(p), -1.0), 1.0))
        if Y >= 0 and x >= 0:
            psi = psi_0
        elif Y < 0 and x >= 0:
            psi = math.pi - psi_0
        elif Y <= 0 and x < 0:
            psi = math.pi + psi_0
        else:
            psi = 2 * math.pi - psi_0

        eta = 0.5 * math.log(1 - 2 * q + 2 * math.sqrt(q**2 - q))
        return eta, psi
