
# Python std library
import timeit
from datetime import timedelta
import logging

# External imports
import numpy as np

# Local imports
import mathieu_functions_OG as mf
import at_config

logger = logging.getLogger(__name__)


class ATSimulation:
    def __init__(self, config: at_config.ATConfiguration):
        self.config = config
        self.coeff = []

    def run(self):
        start = timeit.default_timer()

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

    def conc_array(self, xmin: float, ymin: float, xmax: float, ymax: float, inc: float):
        xaxis = np.arange(xmin, xmax, inc)
        yaxis = np.arange(ymin, ymax, inc)
        num_xaxis = len(xaxis)
        num_yaxis = len(yaxis)

        result = np.zeros((num_xaxis, num_yaxis))

        for i in range(0, num_yaxis):
            for j in range(0, num_xaxis):
                result[j, i] = self.calc_c(xaxis[j], yaxis[i])


        return (xaxis, yaxis, result)

    def calc_c(self, x: float, y: float) -> float:
        # TODO: implement
        return 0.0

    def calc_uv(self, x: float, y: float, d: float):
        alpha_l: float = self.config.alpha_l
        alpha_t: float = self.config.alpha_t

        Y: float = np.sqrt(alpha_l / alpha_t) * y
        B: float = x**2.0 + Y**2.0 - d**2.0
        p: float = (-B + np.sqrt(B**2.0 + 4.0 * d**2.0 * x**2.0)) / (2.0 * d**2.0)
        q: float = (-B - np.sqrt(B**2.0 + 4.0 * d**2.0 * x**2.0)) / (2.0 * d**2.0)

        psi: float = np.nan
        psi_0: float = np.arcsin(np.sqrt(p))

        if Y >= 0 and x >= 0:
            psi = psi_0
        if Y < 0 and x >= 0:
            psi = np.pi - psi_0
        if Y <= 0 and x < 0:
            psi = np.pi + psi_0
        if Y > 0 and x < 0:
            psi = 2 * np.pi - psi_0

        eta: float = 0.5 * np.log(1 - 2 * q + 2 * np.sqrt(q**2 - q))
        return (eta, psi)

    def calc_mathieu(self, order: float, psi: float, eta: float, q: float):
        m = mf.mathieu(q)
        Se = m.ce(order, psi).real
        So = m.se(order, psi).real
        Ye = m.Ke(order, eta).real
        Yo = m.Ko(order, eta).real
        return (Se, So, Ye, Yo)

    def calc_d_q(self, r: float):
        alpha_l: float = self.config.alpha_l
        alpha_t: float = self.config.alpha_t
        beta = self.config.beta

        d = np.sqrt((r * np.sqrt(alpha_l / alpha_t))**2.0 - r**2.0)
        q = (d**2.0 * beta**2.0) / 4.0
        return (d, q)

    def solve_system(self):
        # TODO: calculate
        ms_expansion = []
        boundary_cond = []

        num_elems = len(self.config.elements)

        for i in range(0, num_elems):
            for j in range(0, num_elems):
                if i != j:
                    # TODO: match on element kind
                    pass

        # Solve least squares
        coeff = np.linalg.lstsq(ms_expansion, boundary_cond, rcond=None)
        self.coeff = coeff[0]



