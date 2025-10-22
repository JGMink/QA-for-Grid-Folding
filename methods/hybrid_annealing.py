import numpy as np
import itertools
import os
import shutil
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
try:
    from pyqubo import Array
    HAS_PYQUBO = True
except Exception:
    Array = None
    HAS_PYQUBO = False

# === CONFIG ===
GRID_SHAPE = (3, 2, 2)
Lx, Ly, Lz = GRID_SHAPE
SEQUENCE = "HHPCHPPCHHCP"
LAMBDA_CONN = 5.0
LAMBDA_DUP = 10.0
LAMBDA_UNCOVERED = 2.0

PLOTS_ROOT = os.path.join("plots", "ha")


def ensure_plots_dir():
    old = "ha_plots"
    if os.path.exists(old) and not os.path.exists(PLOTS_ROOT):
        try:
            os.makedirs(os.path.dirname(PLOTS_ROOT), exist_ok=True)
            shutil.move(old, PLOTS_ROOT)
        except Exception:
            os.makedirs(PLOTS_ROOT, exist_ok=True)
    else:
        os.makedirs(PLOTS_ROOT, exist_ok=True)


def lattice_coords():
    xs, ys, zs = range(Lx), range(Ly), range(Lz)
    return [(x, y, z) for x in xs for y in ys for z in zs]


COORDS = lattice_coords()
N_SITES = len(COORDS)
N_RESIDUES = len(SEQUENCE)


def is_neighbor(a, b):
    return sum(abs(a[i] - b[i]) for i in range(3)) == 1


def plot_structure(residue_coords, seq, title, fname):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs_all, ys_all, zs_all = zip(*COORDS)
    ax.scatter(xs_all, ys_all, zs_all, c='lightgray', s=50, alpha=0.3)

    xs, ys, zs = zip(*residue_coords)
    colors = {'H': 'gold', 'P': 'skyblue', 'C': 'salmon'}
    for (x, y, z), s in zip(residue_coords, seq):
        ax.scatter(x, y, z, c=colors[s], s=200, edgecolor='k')
        ax.text(x + 0.05, y + 0.05, z + 0.05, s, fontsize=12, weight='bold')

    ax.plot(xs, ys, zs, c='gray', alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_ROOT, f"{fname}.png"))
    plt.close()


if HAS_PYQUBO:
    # Build pyqubo model
    x = Array.create('x', shape=(N_RESIDUES, N_SITES), vartype='BINARY')

    H_conn = 0
    for i in range(N_RESIDUES - 1):
        for s1, s2 in itertools.product(range(N_SITES), repeat=2):
            if not is_neighbor(COORDS[s1], COORDS[s2]):
                H_conn += LAMBDA_CONN * x[i, s1] * x[i + 1, s2]

    H_dup = 0
    for s in range(N_SITES):
        for i in range(N_RESIDUES):
            for j in range(i + 1, N_RESIDUES):
                H_dup += LAMBDA_DUP * x[i, s] * x[j, s]

    H_cov = 0
    for i in range(N_RESIDUES):
        H_cov += LAMBDA_UNCOVERED * (1 - sum(x[i, s] for s in range(N_SITES)))**2

    H_int = 0
    for i in range(N_RESIDUES):
        for j in range(i + 2, N_RESIDUES):
            for s1, s2 in itertools.product(range(N_SITES), repeat=2):
                if is_neighbor(COORDS[s1], COORDS[s2]):
                    if SEQUENCE[i] == 'H' and SEQUENCE[j] == 'H':
                        H_int -= x[i, s1] * x[j, s2]
                    elif SEQUENCE[i] == 'C' and SEQUENCE[j] == 'C':
                        H_int += x[i, s1] * x[j, s2]

    H_total = H_conn + H_dup + H_cov + H_int
    model = H_total.compile()


    def energy_from_sample(sample_dict):
        return model.energy(sample_dict, vartype="BINARY")
else:
    def energy_from_sample(sample_dict):
        raise RuntimeError(
            "pyqubo is not installed. Install it with `pip install pyqubo` to use hybrid_annealing.`"
        )


def hybrid_annealing(max_iterations=50, n_candidates=10):
    ensure_plots_dir()

    perm_sites = np.random.permutation(N_SITES)
    xclassical = perm_sites[:N_RESIDUES]
    best_solution = xclassical.copy()
    best_energy = float('inf')
    energy_trace = []

    for iteration in range(1, max_iterations + 1):
        candidates = []
        for _ in range(n_candidates):
            perm_sites = np.random.permutation(N_SITES)
            candidate = perm_sites[:N_RESIDUES]
            candidates.append(candidate)

        energies = []
        for cand in candidates:
            sample_dict = {f'x[{i}][{s}]': 1 if cand[i] == s else 0
                           for i in range(N_RESIDUES)
                           for s in range(N_SITES)}
            e = energy_from_sample(sample_dict)
            energies.append(e)

        min_idx = np.argmin(energies)
        xclassical = candidates[min_idx]
        E_current = energies[min_idx]
        energy_trace.append(E_current)

        if E_current < best_energy:
            best_energy = E_current
            best_solution = xclassical.copy()

        if iteration % 10 == 0 or iteration == max_iterations:
            residue_coords = [COORDS[i] for i in xclassical]
            plot_structure(residue_coords, SEQUENCE, f"HA Iter {iteration} (E={E_current})",
                           f"ha_iter_{iteration}")

    return best_solution, energy_trace, best_energy


if __name__ == "__main__":
    ensure_plots_dir()
    print("=== Starting Hybrid Annealing ===")
    solution, energy_trace, final_energy = hybrid_annealing(max_iterations=50, n_candidates=20)
    residue_coords = [COORDS[i] for i in solution]
    print(f"Best HA Energy: {final_energy}")
    plot_structure(residue_coords, SEQUENCE, f"HA Final (E={final_energy})", "ha_final")

    plt.figure()
    plt.plot(range(len(energy_trace)), energy_trace, color='green', linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Energy")
    plt.title("Energy vs Iteration (Hybrid Annealing)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_ROOT, "energy_trace.png"))
    plt.close()
    print(f"\nHA plots saved to {PLOTS_ROOT}")
