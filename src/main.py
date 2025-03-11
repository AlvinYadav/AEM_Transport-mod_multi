# Written by Willi Kappler, willi.kappler@uni-tuebingen.de
# Based on code from Anton Köhler


# Python std library
import logging

# External imports

# Local imports
import at_config
import at_simulation


def main():
    log_file_name: str = "aem_transport_simulation.log"
    log_format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    logging.basicConfig(filename=log_file_name, level=logging.DEBUG, format=log_format)


    # TODO: Use command line arguments for config file
    filename = "simulation_config.json"
    config = at_config.ATConfiguration.from_json(filename)
    simulation = at_simulation.ATSimulation(config)
    simulation.run()


if __name__ == "__main__":
    main()

