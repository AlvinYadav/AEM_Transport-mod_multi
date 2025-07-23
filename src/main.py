# Written by Willi Kappler, willi.kappler@uni-tuebingen.de
# Based on code from Anton Köhler

# Python std library
import logging

# Local imports
from at_config import ATConfiguration
from at_simulation import ATSimulation
from at_findalpha_t import process_input_file

# Choose mode: "simulate" to run transport sim, "findalpha" to run alpha-finder
MODE = "findalpha"  # "simulate" or "findalpha"

# When MODE == "simulate":
CONFIG_PATH = "simulation_config.json"

# When MODE == "findalpha":
INPUT_FILE       = "input_values_O2.txt"
OUTPUT_FILE      = "output_results.txt"
ORIENTATION      = "vertical"        # "horizontal" or "vertical"
ALPHA_START      = 0.0001
STEP             = 0.02
TOLERANCE        = 10.0
MAX_ALPHA        = 0.21
MAX_STAGNATION   = 5

def setup_logging():
    log_file = "aem_transport_simulation.log"
    log_format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    logging.basicConfig(filename=log_file,
                        level=logging.DEBUG,
                        format=log_format)


def run_simulation():
    # Load configuration and run the full transport simulation
    config = ATConfiguration.from_json(CONFIG_PATH)
    sim = ATSimulation(config)
    sim.run()
    # Return or process sim.result_tuple as needed
    return sim.result_tuple


def run_findalpha():
    # Run the alpha-finder in batch mode
    process_input_file(
        INPUT_FILE,
        OUTPUT_FILE,
        ORIENTATION,
        ALPHA_START,
        STEP,
        TOLERANCE,
        MAX_ALPHA,
        MAX_STAGNATION
    )


def main():
    setup_logging()

    if MODE == "simulate":
        result = run_simulation()
        # You can add code here to handle or display 'result'
        print("Simulation completed. Result tuple returned.")
    elif MODE == "findalpha":
        run_findalpha()
        print(f"Alpha-finder completed. Results in '{OUTPUT_FILE}'.")
    else:
        raise ValueError(f"Unknown MODE '{MODE}'. Use 'simulate' or 'findalpha'.")


if __name__ == "__main__":
    main()

