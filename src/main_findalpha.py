#!/usr/bin/env python3
# main_findalpha.py

from at_findalpha_t import process_input_file

def main():
    # ─────── User-configurable section ───────
    input_file   = "input_values_O2.txt"   # <-- your input txt
    output_file  = "output_values.txt"     # <-- where to write results
    orientation = "vertical"

    # You can override these defaults however you like:
    alpha_start:float = 0.08
    step:float        = 0.001
    tolerance:float   = 10.0
    max_alpha: float = 0.2
    max_stag: int = 5

    process_input_file(
        input_file,
        output_file,
        orientation,
        alpha_start=alpha_start,
        step=step,
        tolerance=tolerance,
        max_alpha=max_alpha,
        max_stagnation=max_stag
    )

if __name__ == "__main__":
    main()
