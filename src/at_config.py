# Python std library
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class ATConfiguration:
    def __init__(self):
        self.alpha_l: float = 10.0
        self.alpha_t: float = 1.0
        self.beta: float = 1.0 / (2.0 * self.alpha_l)
        self.gamma: float = 3.5
        self.Ca: float = 8.0
        self.n: int = 7
        self.M: int = 100
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

        if "alpha_l" in data:
            config.alpha_l = data["alpha_l"]

        if "alpha_t" in data:
            config.alpha_t = data["alpha_t"]

        if "beta" in data:
            config.beta = data["beta"]

        if "gamma" in data:
            config.gamma = data["gamma"]

        # TODO: Add more config options

        return config

