# Written by Willi Kappler, willi.kappler@uni-tuebingen.de
# Based on code from Anton Köhler

# Python std library
import json
import logging
from typing import Any

# Local imports:
from at_element import ATECircle, ATELine

logger = logging.getLogger(__name__)


class ATConfiguration:
    def __init__(self):
        self.alpha_l: float = 10.0
        self.alpha_t: float = 1.0
        self.beta: float = 1.0 / (2.0 * self.alpha_l)
        self.gamma: float = 3.5
        self.ca: float = 8.0
        self.num_terms: int = 7 # Number of terms
        self.num_cp: int = 100 # Number of controll points
        self.dom_xmin: float = 10.0
        self.dom_ymin: float = -5.0
        self.dom_xmax: float = 20.0
        self.dom_ymax: float = 6.0
        self.dom_inc: float = 1.0
        self.elements = []

    @staticmethod
    def from_json(file_name) -> Any:
        """
        Load the configuration (JSON format) from the given file name.

        :param file_name: File name of the configuration.
        :return: A valid configuration from the given JSON file.
        """

        logger.debug(f"Load configuration from file: {file_name}.")

        with open(file_name, "r") as f:
            data = json.load(f)

        config = ATConfiguration()

        match data:
            case {"alpha_l": alpha_l}:
                config.alpha_l = alpha_l
            case {"alpha_t": alpha_t}:
                config.alpha_t = alpha_t
            case {"beta": beta}:
                config.beta = beta
            case {"gamma": gamma}:
                config.gamma = gamma
            case {"ca": ca}:
                config.ca = ca
            case {"num_terms": num_terms}:
                config.num_terms = num_terms
            case {"num_cp": num_cp}:
                config.num_cp = num_cp
            case {"dom_xmin": dom_xmin}:
                config.dom_xmin = dom_xmin
            case {"dom_ymin": dom_ymin}:
                config.dom_ymin = dom_ymin
            case {"dom_xmax": dom_xmax}:
                config.dom_xmax = dom_xmax
            case {"dom_ymax": dom_ymax}:
                config.dom_ymax = dom_ymax
            case {"dom_inc": dom_inc}:
                config.dom_inc = dom_inc
            case {"elements": elements}:
                for element in elements:
                    match element:
                        case {"circle": [{"x": x}, {"y": y}, {"r": r}, {"con": con}]}:
                            print(f"{x=}, {y=}, {r=}, {con=}")
                            new_circle = ATECircle(con, x, y, r)
                            config.elements.append(new_circle)
                        case {"line": [{"start": start}, {"end": end}]}:
                            print(f"{start=}, {end=}")
                        case _:
                            raise ValueError(f"Unknown element: {element=}")

        return config

