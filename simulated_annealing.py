import numpy as np
import itertools
"""Top-level wrapper for the refactored simulated annealing method.

This script keeps the original entry point but delegates implementation to
the `methods.simulated_annealing` module and ensures the new `plots/` layout
exists for backward compatibility.
"""

import os
from methods.simulated_annealing import simulated_annealing


def main():
    os.makedirs(os.path.join("plots", "sa"), exist_ok=True)
    sa_E, sa_config = simulated_annealing()
    print(f"Final SA Energy: {sa_E:.2f}")


if __name__ == "__main__":
    main()

