import numpy as np
import itertools
import random
import os
import shutil
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# === CONFIG ===
GRID_SHAPE = (2, 3, 1)
Lx, Ly, Lz = GRID_SHAPE
# Protein sequence (H: hydrophobic, P: polar, C: charged)
SEQUENCE = "HHPCHP"
# Constants
LAMBDA_CONN = 5.0        # connectivity violation penalty
LAMBDA_DUP = 10.0         # duplicate-site penalty
LAMBDA_UNCOVERED = 2.0    # uncovered-site penalty
# SA parameters
TEMPERATURE = 2.0
ALPHA = 0.95
N_SWEEPS = 100

# === ENERGY MODEL ===
def lattice_coords():
    xs, ys, zs = range(GRID_SHAPE[0]), range(GRID_SHAPE[1]), range(GRID_SHAPE[2])
    return [(x, y, z) for x in xs for y in ys for z in zs]

COORDS = lattice_coords()

def is_neighbor(a, b):
    return sum(abs(a[i] - b[i]) for i in range(3)) == 1

def energy(config, seq):
    """
    Compute the total energy of a protein configuration.
    Includes:
      - Interaction potential (H–H attraction, C–C repulsion)
      - Connectivity penalties
      - Uniqueness (no overlapping residues)
      - Coverage (each lattice site should be occupied at most once)
    """

    E = 0

    # --- Interaction Potential ---
    # Favor H-H contacts, penalize C-C contacts.
    for i in range(len(seq)):
        for j in range(i + 2, len(seq)):  # skip adjacent chain residues
            if is_neighbor(config[i], config[j]):
                if seq[i] == "H" and seq[j] == "H":
                    E -= 1
                elif seq[i] == "C" and seq[j] == "C":
                    E += 1

    # --- Connectivity Penalty ---
    # If two consecutive residues are not neighbors, apply a large penalty.
    for i in range(len(seq) - 1):
        if not is_neighbor(config[i], config[i + 1]):
            E += LAMBDA_CONN

    # --- Uniqueness Penalty ---
    # Penalize if multiple residues occupy the same site.
    seen = {}
    for idx, site in enumerate(config):
        if site in seen:
            # Overlapping sites -> strong penalty
            E += LAMBDA_DUP
        else:
            seen[site] = True

    # --- Coverage Penalty (optional) ---
    # If you want exactly one residue per lattice site in a fixed grid:
    all_sites = [(x, y, z) for x in range(Lx) for y in range(Ly) for z in range(Lz)]
    if len(config) < len(all_sites):
        # Some sites unoccupied (if we’re enforcing 1:1 mapping)
        E += LAMBDA_UNCOVERED * (len(all_sites) - len(config))

    return E


# === BRUTE FORCE OPTIMAL ===
def brute_force_best(seq):
    best_E = float("inf")
    best_configs = []
    for perm in itertools.permutations(COORDS, len(seq)):
        E = energy(perm, seq)
        if E < best_E:
            best_E = E
            best_configs = [perm]
        elif E == best_E:
            best_configs.append(perm)
    return best_E, best_configs

# === VISUALIZATION ===
def plot_structure(config, seq, title, fname):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs, ys, zs = zip(*config)
    colors = {'H':'gold', 'P':'skyblue', 'C':'salmon'}
    for (x, y, z), s in zip(config, seq):
        ax.scatter(x, y, z, c=colors[s], s=200, edgecolor='k')
        ax.text(x+0.05, y+0.05, z+0.05, s, fontsize=12, weight='bold')
    ax.plot(xs, ys, zs, c='gray', alpha=0.6)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(f"plots/{fname}.png")
    plt.close()

# === SA CORE ===
def simulated_annealing(seq, n_sweeps=100, T0=2.0, alpha=0.95):
    coords = COORDS.copy()
    current = random.sample(coords, len(seq))
    best = current[:]
    E = energy(current, seq)
    best_E = E
    T = T0

    energy_trace = [E]  # track energy over sweeps

    print(f"Initial Energy: {E:.2f}")
    for sweep in range(1, n_sweeps+1):
        i, j = random.sample(range(len(seq)), 2)
        candidate = current[:]
        candidate[i], candidate[j] = candidate[j], candidate[i]
        E_new = energy(candidate, seq)
        dE = E_new - E

        prob = np.exp(-dE / T) if dE > 0 else 1.0
        if dE <= 0 or random.random() < prob:
            reason = "better" if dE <= 0 else f"accepted with p={prob:.3f}"
            current, E = candidate, E_new
            print(f"Sweep {sweep:3d}: flipped ({i},{j}), ΔE={dE:+.2f}, {reason}")
            if E < best_E:
                best_E, best = E, candidate[:]
        else:
            print(f"Sweep {sweep:3d}: flipped ({i},{j}), ΔE={dE:+.2f}, rejected with p={prob:.3f}")

        energy_trace.append(E)  # record after each sweep

        if sweep in [10, 50, 100]:
            plot_structure(current, seq, f"SA after {sweep} sweeps (E={E:.2f})", f"sa_{sweep}")
        T *= alpha

    # Plot energy vs sweep
    plt.figure()
    plt.plot(range(len(energy_trace)), energy_trace, color='purple', linewidth=2)
    plt.xlabel("Sweep")
    plt.ylabel("Energy")
    plt.title("Energy vs. Sweep (Simulated Annealing)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/energy_trace.png")
    plt.close()

    return best_E, best

# === MAIN ===
if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    if os.path.exists("plots"):
        shutil.rmtree("plots")
    os.makedirs("plots")

    print(f"Running brute-force optimal search for {SEQUENCE}...")
    best_E, best_configs = brute_force_best(SEQUENCE)
    print(f"\nOptimal energy (brute-force): {best_E:.2f}")
    print(f"Found {len(best_configs)} degenerate optimal configurations.\n")

    for idx, conf in enumerate(best_configs[:3]):  # only show first 3 if many
        print(f"Config #{idx+1}:")
        for i, coord in enumerate(conf):
            print(f"  {SEQUENCE[i]} → {coord}")
        plot_structure(conf, SEQUENCE, f"Optimal #{idx+1} (E={best_E:.2f})", f"optimal_{idx+1}")
        print()

    print("=== Starting Simulated Annealing ===\n")
    sa_E, sa_config = simulated_annealing(SEQUENCE, N_SWEEPS, TEMPERATURE, ALPHA)

    print(f"\nFinal SA Energy: {sa_E:.2f}")
    plot_structure(sa_config, SEQUENCE, f"SA Final (E={sa_E:.2f})", "final")
    print("\nPlots saved to ./plots/")
