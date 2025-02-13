
# Python std library
import timeit
from datetime import timedelta
import logging
import math

# External imports
import numpy as np

# Local imports
import mathieu_functions_OG as mf
import at_config
from at_element import ATElementType

logger = logging.getLogger(__name__)


class ATSimulation:
    def __init__(self, config: at_config.ATConfiguration):
        self.config = config
        self.coeff = []

    def run(self):
        start = timeit.default_timer()

        alpha_t: float = self.config.alpha_t
        alpha_l: float = self.config.alpha_l
        beta: float = self.config.beta
        gamma: float = self.config.gamma
        ca: float = self.config.ca
        num_cp: int = self.config.num_cp

        for elem in self.config.elements:
            match elem.kind:
                case ATElementType.Circle:
                    elem.calc_d_q(alpha_t, alpha_l, beta)
                    elem.set_outline(num_cp)
                    elem.calc_target(ca, beta, gamma)
                case ATElementType.Line:
                    pass

        self.solve_system()

        xmin: float = self.config.dom_xmin
        ymin: float = self.config.dom_ymin
        xmax: float = self.config.dom_xmax
        ymax: float = self.config.dom_ymax
        inc: float = self.config.dom_inc

        (xaxis, yaxis, values) = self.conc_array(xmin, ymin, xmax, ymax, inc)

        stop = timeit.default_timer()
        sec = int(stop - start)
        cpu_time = timedelta(seconds = sec)
        print(f"Computation time [hh:mm:ss]: {cpu_time}")

        # TODO: plot result

    def conc_array(self, xmin: float, ymin: float, xmax: float, ymax: float,
            inc: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xaxis: np.ndarray = np.arange(xmin, xmax, inc)
        yaxis: np.ndarray = np.arange(ymin, ymax, inc)
        num_xaxis: int = len(xaxis)
        num_yaxis: int = len(yaxis)

        result: np.ndarray = np.zeros((num_xaxis, num_yaxis))

        for i in range(0, num_yaxis):
            for j in range(0, num_xaxis):
                result[j, i] = self.calc_c(xaxis[j], yaxis[i])

        return (xaxis, yaxis, result)

    def calc_c(self, x: float, y: float) -> float:
        for elem in self.config.elements:
            if elem.is_inside(x, y):
                return elem.c_inside()

        threshold: float = 0.0
        factor: float = 0.0
        for elem in self.config.elements:
            (eta, psi) = self.calc_uv(x - elem.x, y - elem.y, elem.d)
            factor = self.coeff[0] * self.calc_mathieu1(0, psi, eta, elem.q)
            for i in range(1, self.config.num_terms):
                (f1, f2) = self.calc_mathieu2(i, psi, eta, elem.q)
                i2: int = i*2
                factor += (self.coeff[i2 - 1] * f1) + (self.coeff[i2] * f2)
            threshold += factor * math.exp(self.config.beta * x)

        ca: float = self.config.ca
        diff: float = threshold - ca
        if threshold > ca:
            return diff / self.config.gamma
        else:
            return diff

    def calc_uv(self, x: float, y: float, d: float) -> tuple[float, float]:
        alpha_l: float = self.config.alpha_l
        alpha_t: float = self.config.alpha_t

        x2: float = x**2.0
        d2: float = d**2.0
        Y: float = math.sqrt(alpha_l / alpha_t) * y
        B: float = x2 + Y**2.0 - d2
        f1: float = math.sqrt(B**2.0 + 4.0 * d2 * x2)
        f2: float = 2.0 * d2
        p: float = (-B + f1) / f2
        q: float = (-B - f1) / f2

        psi: float = math.nan
        psi_0: float = math.asin(math.sqrt(p))

        if Y >= 0.0 and x >= 0.0:
            psi = psi_0
        if Y < 0.0 and x >= 0.0:
            psi = math.pi - psi_0
        if Y <= 0.0 and x < 0.0:
            psi = math.pi + psi_0
        if Y > 0.0 and x < 0.0:
            psi = 2.0 * math.pi - psi_0

        eta: float = 0.5 * math.log(1.0 - 2.0 * q + 2.0 * math.sqrt(q**2.0 - q))
        return (eta, psi)

    def calc_mathieu1(self, order: int, psi: float, eta: float,
            q: float) -> np.ndarray:
        m = mf.Mathieu(q)
        Se = m.ce(order, psi).real
        Ye = m.Ke(order, eta).real
        return Se * Ye

    def calc_mathieu2(self, order: int, psi: float, eta: float,
            q: float) -> tuple[np.ndarray, np.ndarray]:
        m = mf.Mathieu(q)
        Se = m.ce(order, psi).real
        So = m.se(order, psi).real
        Ye = m.Ke(order, eta).real
        Yo = m.Ko(order, eta).real
        return (So * Yo, Se * Ye)

    def solve_system(self):
        elements = self.config.elements
        num_elems: int = len(elements)

        perspective_vector = []
        target_function_vector = []

        for e1 in range(0, num_elems):
            elem1 = elements[e1]
            elem1.m_list = []
            x1: float = elem1.x
            y1: float = elem1.y
            for e2 in range(0, num_elems):
                elem2 = elements[e2]
                x2: float = elem2.x
                y2: float = elem2.y
                d2: float = elem2.d
                q2: float = elem2.q
                for i in range(0, self.config.num_cp):
                    (x, y) = elem1.outline[i]

                    if e1 != e2:
                        dist_x: float = x1 - x2
                        dist_y: float = y1 - y2
                        x = x + dist_x
                        y = y + dist_y

                    (eta, psi) = self.calc_uv(x, y, d2)
                    elem1.m_list.append(self.calc_mathieu1(0, psi, eta, q2))
                    for j in range(1, self.config.num_terms):
                        (f1, f2) = self.calc_mathieu2(j, psi, eta, q2)
                        elem1.m_list.append(f1)
                        elem1.m_list.append(f2)

            m_list_len = len(elem1.m_list)
            s = (num_elems * self.config.num_terms - 1) * num_elems
            for i in range(0, m_list_len, s):
                perspective_vector.append(elem1.m_list[i:i+s])

            target_function_vector.extend(elem1.target)

        # Solve least squares
        coeff = np.linalg.lstsq(perspective_vector, target_function_vector, rcond=None)
        self.coeff = coeff[0]

