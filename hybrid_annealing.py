"""Top-level wrapper for the refactored hybrid annealing method.

Delegates implementation to `methods.hybrid_annealing` and ensures the new
`plots/ha` layout exists for backward compatibility.
"""

import os
from methods.hybrid_annealing import hybrid_annealing


def main():
    os.makedirs(os.path.join("plots", "ha"), exist_ok=True)
    solution, energy_trace, final_energy = hybrid_annealing()
    print(f"Best HA Energy: {final_energy}")


if __name__ == "__main__":
    main()
