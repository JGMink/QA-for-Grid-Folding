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
SEQUENCE = "HHPCHP"
LAMBDA_CONN = 5.0
LAMBDA_DUP = 10.0
LAMBDA_UNCOVERED = 2.0
TEMPERATURE = 2.0
ALPHA = 0.95
N_SWEEPS = 100

# New standardized output locations
PLOTS_ROOT = os.path.join("plots", "sa")


def ensure_plots_dir():
    # Migrate old directory if present
    old = "sa_plots"
    if os.path.exists(old) and not os.path.exists(PLOTS_ROOT):
        try:
            os.makedirs(os.path.dirname(PLOTS_ROOT), exist_ok=True)
            shutil.move(old, PLOTS_ROOT)
        except Exception:
            # If move fails, fall back to creating target dir
            os.makedirs(PLOTS_ROOT, exist_ok=True)
    else:
        os.makedirs(PLOTS_ROOT, exist_ok=True)


def lattice_coords():
    xs, ys, zs = range(GRID_SHAPE[0]), range(GRID_SHAPE[1]), range(GRID_SHAPE[2])
    return [(x, y, z) for x in xs for y in ys for z in zs]


COORDS = lattice_coords()


def is_neighbor(a, b):
    return sum(abs(a[i] - b[i]) for i in range(3)) == 1


def energy(config, seq):
    E = 0
    for i in range(len(seq)):
        for j in range(i + 2, len(seq)):
            if is_neighbor(config[i], config[j]):
                if seq[i] == "H" and seq[j] == "H":
                    E -= 1
                elif seq[i] == "C" and seq[j] == "C":
                    E += 1

    for i in range(len(seq) - 1):
        if not is_neighbor(config[i], config[i + 1]):
            E += LAMBDA_CONN

    seen = {}
    for idx, site in enumerate(config):
        if site in seen:
            E += LAMBDA_DUP
        else:
            seen[site] = True

    all_sites = [(x, y, z) for x in range(Lx) for y in range(Ly) for z in range(Lz)]
    if len(config) < len(all_sites):
        E += LAMBDA_UNCOVERED * (len(all_sites) - len(config))

    return E


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


def plot_structure(config, seq, title, fname):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs, ys, zs = zip(*config)
    colors = {'H': 'gold', 'P': 'skyblue', 'C': 'salmon'}
    for (x, y, z), s in zip(config, seq):
        ax.scatter(x, y, z, c=colors[s], s=200, edgecolor='k')
        ax.text(x + 0.05, y + 0.05, z + 0.05, s, fontsize=12, weight='bold')
    ax.plot(xs, ys, zs, c='gray', alpha=0.6)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_ROOT, f"{fname}.png"))
    plt.close()


def simulated_annealing(seq=SEQUENCE, n_sweeps=N_SWEEPS, T0=TEMPERATURE, alpha=ALPHA):
    ensure_plots_dir()

    coords = COORDS.copy()
    current = random.sample(coords, len(seq))
    best = current[:]
    E = energy(current, seq)
    best_E = E
    T = T0

    energy_trace = [E]

    print(f"Initial Energy: {E:.2f}")
    for sweep in range(1, n_sweeps + 1):
        i, j = random.sample(range(len(seq)), 2)
        candidate = current[:]
        candidate[i], candidate[j] = candidate[j], candidate[i]
        E_new = energy(candidate, seq)
        dE = E_new - E

        prob = np.exp(-dE / T) if dE > 0 else 1.0
        if dE <= 0 or random.random() < prob:
            current, E = candidate, E_new
            if E < best_E:
                best_E, best = E, candidate[:]

        energy_trace.append(E)

        if sweep in [10, 50, 100]:
            plot_structure(current, seq, f"SA after {sweep} sweeps (E={E:.2f})", f"sa_{sweep}")
            print(f"--- Energy breakdown for sweep {sweep} ---")
            print_energy_breakdown(current, seq)
        T *= alpha

    plt.figure()
    plt.plot(range(len(energy_trace)), energy_trace, color='purple', linewidth=2)
    plt.xlabel("Sweep")
    plt.ylabel("Energy")
    plt.title("Energy vs. Sweep (Simulated Annealing)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_ROOT, "energy_trace.png"))
    plt.close()

    return best_E, best


def print_energy_breakdown(config, seq):
    E = 0
    print("\n--- Energy Breakdown ---")
    interaction_E = 0
    for i in range(len(seq)):
        for j in range(i + 2, len(seq)):
            if is_neighbor(config[i], config[j]):
                if seq[i] == "H" and seq[j] == "H":
                    interaction_E -= 1
                elif seq[i] == "C" and seq[j] == "C":
                    interaction_E += 1
    print(f"Interaction Energy: {interaction_E}")
    E += interaction_E

    conn_E = 0
    for i in range(len(seq) - 1):
        if not is_neighbor(config[i], config[i + 1]):
            conn_E += LAMBDA_CONN
    print(f"Connectivity Penalty: {conn_E}")
    E += conn_E

    seen = {}
    dup_E = 0
    for idx, site in enumerate(config):
        if site in seen:
            dup_E += LAMBDA_DUP
        else:
            seen[site] = True
    print(f"Duplication Penalty: {dup_E}")
    E += dup_E

    all_sites = [(x, y, z) for x in range(Lx) for y in range(Ly) for z in range(Lz)]
    cov_E = 0
    if len(config) < len(all_sites):
        cov_E = LAMBDA_UNCOVERED * (len(all_sites) - len(config))
    print(f"Coverage Penalty: {cov_E}")
    E += cov_E

    print(f"Total Energy: {E}\n")
    return E


if __name__ == "__main__":
    # Provide backward-compatible command-line behavior
    os.makedirs("results", exist_ok=True)
    best_E, best_configs = brute_force_best(SEQUENCE)
    print(f"\nOptimal energy (brute-force): {best_E:.2f}")
    for idx, conf in enumerate(best_configs[:3]):
        plot_structure(conf, SEQUENCE, f"Optimal #{idx+1} (E={best_E:.2f})", f"optimal_{idx+1}")

    print("=== Starting Simulated Annealing ===\n")
    sa_E, sa_config = simulated_annealing(SEQUENCE, N_SWEEPS, TEMPERATURE, ALPHA)
    print(f"\nFinal SA Energy: {sa_E:.2f}")
    plot_structure(sa_config, SEQUENCE, f"SA Final (E={sa_E:.2f})", "final")
    print(f"\nPlots saved to {PLOTS_ROOT}")
