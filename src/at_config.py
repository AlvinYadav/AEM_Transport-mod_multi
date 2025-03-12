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
        self.dom_xmin: float = 0.0
        self.dom_ymin: float = -20.0
        self.dom_xmax: float = 150.0
        self.dom_ymax: float = 30.0
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

        for key, value in data.items():
            match key:
                case "alpha_l":
                    config.alpha_l = value
                case "alpha_t":
                    config.alpha_t = value
                case "gamma":
                    config.gamma = value
                case "ca":
                    config.ca = value
                case "num_terms":
                    config.num_terms = value
                case "num_cp":
                    config.num_cp = value
                case "dom_xmin":
                    config.dom_xmin = value
                case "dom_ymin":
                    config.dom_ymin = value
                case "dom_xmax":
                    config.dom_xmax = value
                case "dom_ymax":
                    config.dom_ymax = value
                case "dom_inc":
                    config.dom_inc = value
                case "elements":
                    for element in value:
                        match element:
                            case {"circle": {"x": x, "y": y, "r": r, "con": con}}:
                                logger.debug(f"New circle: {x=}, {y=}, {r=}, {con=}")
                                new_circle = ATECircle(con, x, y, r)
                                config.elements.append(new_circle)
                            case {"line": {"start": start, "end": end}}:
                                logger.debug(f"New line: {start=}, {end=}")
                            case _:
                                raise ValueError(f"Unknown element: {element=}")
        
        config.beta = 1.0 / (2.0 * config.alpha_l)

        return config

