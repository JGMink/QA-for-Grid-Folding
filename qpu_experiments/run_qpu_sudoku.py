"""
QPU Experiments: 4×4 Sudoku on D-Wave Advantage2 (forward annealing, fixed givens)

Three difficulty levels based on number of givens (fixed cells):
  easy   — 12 givens,  4 free cells (16 free vars)
  medium —  8 givens,  8 free cells (32 free vars)
  hard   —  4 givens, 12 free cells (48 free vars)  ← original poster puzzle

The 9×9 puzzles (459–576 free vars) are too large for pure QPU; included for reference.

SA baseline (SimulatedAnnealingSampler) is computed locally so we can compare
QPU valid rate vs SA valid rate in the same script.

Run on Sycamore:
    source ~/quantum_folding_2d/venv/bin/activate
    cd ~/QA_for_Sudoku && git pull

    python qpu_experiments/run_qpu_sudoku.py --phase embed
    nohup python qpu_experiments/run_qpu_sudoku.py --phase solve \\
        > qpu_experiments/logs/sudoku_solve.log 2>&1 &
    python qpu_experiments/run_qpu_sudoku.py --phase analyze
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from dwave.system import DWaveSampler, FixedEmbeddingComposite
from dwave.embedding.chain_strength import uniform_torque_compensation
from minorminer import find_embedding
import dimod
from dimod import BinaryQuadraticModel, SimulatedAnnealingSampler

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SOLVER_ID      = "Advantage2_system1.13"
NUM_READS      = 1000
ANNEALING_TIME = 200   # µs
EMBED_TIMEOUT  = 600
EMBED_TRIES    = 50
EMBED_SEED     = 42
LAM            = 1.0   # penalty weight for all Sudoku constraints
SA_READS       = 200   # SA reads for baseline

CHAIN_STRENGTH_MULTIPLIER = 1  # × max_coupling (CS1x)

HERE           = Path(__file__).parent
RESULTS_DIR    = HERE / "results_sudoku"
EMBEDDINGS_DIR = HERE / "embeddings_sudoku"
LOGS_DIR       = HERE / "logs"

# ---------------------------------------------------------------------------
# Puzzles  (0 = blank)
# Solution: [[3,2,1,4],[4,1,2,3],[2,4,3,1],[1,3,4,2]]
# ---------------------------------------------------------------------------

PUZZLES_4x4 = {
    "easy": np.array([      # 12 givens, 4 free cells → 16 free vars
        [3, 2, 1, 0],
        [4, 1, 2, 0],
        [2, 4, 0, 1],
        [0, 3, 4, 2],
    ]),
    "medium": np.array([    # 8 givens, 8 free cells → 32 free vars
        [3, 2, 0, 0],
        [4, 0, 0, 3],
        [2, 0, 3, 0],
        [0, 0, 4, 2],
    ]),
    "hard": np.array([      # 4 givens, 12 free cells → 48 free vars (poster puzzle)
        [0, 2, 0, 0],
        [0, 0, 0, 3],
        [2, 0, 0, 0],
        [0, 0, 4, 0],
    ]),
}

# Known solutions (for ground-truth comparison)
SOLUTION_4x4 = np.array([
    [3, 2, 1, 4],
    [4, 1, 2, 3],
    [2, 4, 3, 1],
    [1, 3, 4, 2],
])

# 9×9 puzzles — included for SA/hybrid reference but NOT run on pure QPU by default
PUZZLES_9x9 = {
    "easy": np.array([         # 30 givens → 459 free vars
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9],
    ]),
    "hard": np.array([         # 17 givens → 576 free vars (minimum-clue puzzle)
        [8, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 3, 6, 0, 0, 0, 0, 0],
        [0, 7, 0, 0, 9, 0, 2, 0, 0],
        [0, 5, 0, 0, 0, 7, 0, 0, 0],
        [0, 0, 0, 0, 4, 5, 7, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 3, 0],
        [0, 0, 1, 0, 0, 0, 0, 6, 8],
        [0, 0, 8, 5, 0, 0, 0, 1, 0],
        [0, 9, 0, 0, 0, 0, 4, 0, 0],
    ]),
}

# ---------------------------------------------------------------------------
# QUBO construction
# ---------------------------------------------------------------------------

def var_idx(i: int, j: int, k: int, N: int) -> int:
    """Variable index for cell (i,j) having digit k+1 (k is 0-indexed)."""
    return (i * N + j) * N + k


def build_sudoku_bqm(puzzle: np.ndarray, lam: float = LAM) -> BinaryQuadraticModel:
    """
    Build a reduced BQM for a Sudoku puzzle with given cells fixed.

    Given cells are removed from the BQM (their quadratic interactions are
    folded into the linear biases of remaining variables), reducing the
    effective problem size and chain lengths on QPU.
    """
    N = puzzle.shape[0]
    box = int(round(N ** 0.5))
    linear: dict = defaultdict(float)
    quadratic: dict = defaultdict(float)
    offset = 0.0

    def add_one_hot(group: list[int]) -> None:
        """Add (Σ x_i - 1)^2 penalty for the given group of variable indices."""
        nonlocal offset
        for v in group:
            linear[v] += -lam
        for a in range(len(group)):
            for b in range(a + 1, len(group)):
                key = (min(group[a], group[b]), max(group[a], group[b]))
                quadratic[key] += 2 * lam
        offset += lam

    # E1: one digit per cell
    for i in range(N):
        for j in range(N):
            add_one_hot([var_idx(i, j, k, N) for k in range(N)])

    # E2: each digit appears once per row
    for i in range(N):
        for k in range(N):
            add_one_hot([var_idx(i, j, k, N) for j in range(N)])

    # E3: each digit appears once per column
    for j in range(N):
        for k in range(N):
            add_one_hot([var_idx(i, j, k, N) for i in range(N)])

    # E4: each digit appears once per box
    for br in range(box):
        for bc in range(box):
            cells = [(br * box + r, bc * box + c)
                     for r in range(box) for c in range(box)]
            for k in range(N):
                add_one_hot([var_idx(i, j, k, N) for (i, j) in cells])

    bqm = BinaryQuadraticModel(dict(linear), dict(quadratic), offset,
                               vartype="BINARY")

    # Fix given cells: remove those variables, folding their contributions in
    for i in range(N):
        for j in range(N):
            if puzzle[i, j] != 0:
                v = puzzle[i, j] - 1  # 0-indexed digit
                for k in range(N):
                    val = 1 if k == v else 0
                    bqm.fix_variable(var_idx(i, j, k, N), val)

    return bqm


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_sudoku(sample: dict, puzzle: np.ndarray) -> tuple[bool, dict]:
    """
    Validate a (partial) sample against the puzzle.

    sample: dict of {var_idx: 0|1} for the FREE variables only.
    puzzle: full grid with 0 for blanks.

    Returns (is_valid, breakdown) where breakdown has violation counts per constraint type.
    """
    N = puzzle.shape[0]
    box = int(round(N ** 0.5))

    # Reconstruct full grid (givens + sample)
    grid = puzzle.copy().astype(int)
    for i in range(N):
        for j in range(N):
            if puzzle[i, j] == 0:
                for k in range(N):
                    v = var_idx(i, j, k, N)
                    if sample.get(v, 0) == 1:
                        grid[i, j] = k + 1
                        break

    def count_violations(groups):
        violations = 0
        for group_cells in groups:
            digits = [grid[r][c] for (r, c) in group_cells if grid[r][c] != 0]
            if len(digits) != N or len(set(digits)) != N:
                violations += 1
        return violations

    rows  = [[(i, j) for j in range(N)] for i in range(N)]
    cols  = [[(i, j) for i in range(N)] for j in range(N)]
    boxes = [[(br * box + r, bc * box + c) for r in range(box) for c in range(box)]
             for br in range(box) for bc in range(box)]

    e1 = sum(1 for i in range(N) for j in range(N)
             if len([k for k in range(N) if sample.get(var_idx(i, j, k, N), 0) == 1
                     if puzzle[i, j] == 0]) != 1
             if puzzle[i, j] == 0)
    e2 = count_violations(rows)
    e3 = count_violations(cols)
    e4 = count_violations(boxes)

    is_valid = (e1 == 0 and e2 == 0 and e3 == 0 and e4 == 0)
    return is_valid, {"e1_cell": e1, "e2_row": e2, "e3_col": e3, "e4_box": e4}


# ---------------------------------------------------------------------------
# Token / helpers (shared pattern with run_qpu.py)
# ---------------------------------------------------------------------------

def load_token() -> str:
    token = os.environ.get("DWAVE_API_TOKEN")
    if token:
        return token
    for candidate in [
        REPO_ROOT / ".env",
        Path.home() / "quantum_folding_2d" / ".env",
    ]:
        if candidate.exists():
            for line in candidate.read_text().splitlines():
                if line.startswith("DWAVE_API_TOKEN="):
                    return line.split("=", 1)[1].strip()
    raise RuntimeError(
        "DWAVE_API_TOKEN not found. Set it in the environment or repo_root/.env"
    )


def load_or_compute_embedding(bqm, sampler, task_id, cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{task_id}.json"
    if cache_file.exists():
        with open(cache_file) as f:
            data = json.load(f)
        emb = {int(k): v for k, v in data["embedding"].items()}
        s = data["stats"]
        print(f"  [emb] cached  {s['physical_qubits']} qubits  "
              f"max_chain={s['max_chain_length']}  "
              f"chain_strength={s['chain_strength']:.2f}", flush=True)
        return emb

    print(f"  [emb] computing  {len(bqm.variables)} vars  "
          f"(timeout={EMBED_TIMEOUT}s, tries={EMBED_TRIES})...", flush=True)
    source_edges = list(bqm.quadratic.keys())
    t0 = time.time()
    emb = find_embedding(
        source_edges, sampler.edgelist,
        verbose=0, timeout=EMBED_TIMEOUT,
        tries=EMBED_TRIES, random_seed=EMBED_SEED,
    )
    elapsed = time.time() - t0

    if not emb:
        raise RuntimeError(f"Embedding failed for {task_id} after {elapsed:.0f}s")

    chain_lengths = [len(c) for c in emb.values()]
    stats = {
        "physical_qubits":  len({q for c in emb.values() for q in c}),
        "max_chain_length": max(chain_lengths),
        "avg_chain_length": round(float(np.mean(chain_lengths)), 2),
        "chain_strength":   round(float(uniform_torque_compensation(bqm, emb)), 3),
        "embed_time_s":     round(elapsed, 1),
    }
    print(f"  [emb] found in {elapsed:.1f}s  "
          f"{stats['physical_qubits']} qubits  "
          f"max_chain={stats['max_chain_length']}  "
          f"chain_strength={stats['chain_strength']}", flush=True)

    with open(cache_file, "w") as f:
        json.dump({"task_id": task_id,
                   "embedding": {k: list(v) for k, v in emb.items()},
                   "stats": stats,
                   "created_at": datetime.now().isoformat()}, f, indent=2)
    return emb


# ---------------------------------------------------------------------------
# SA baseline
# ---------------------------------------------------------------------------

def run_sa_baseline(bqm: BinaryQuadraticModel, puzzle: np.ndarray,
                    label: str) -> dict:
    """Run SA on the reduced BQM and report valid rate."""
    sampler = SimulatedAnnealingSampler()
    response = sampler.sample(bqm, num_reads=SA_READS, num_sweeps=10_000,
                              beta_range=(0.1, 10.0))

    n_valid = 0
    for datum in response.data():
        is_valid, _ = validate_sudoku(datum.sample, puzzle)
        if is_valid:
            n_valid += datum.num_occurrences

    n_total = sum(d.num_occurrences for d in response.data())
    valid_rate = round(100.0 * n_valid / n_total, 1) if n_total else 0.0
    print(f"  [sa]  valid={valid_rate:.1f}%  ({n_valid}/{n_total} reads)",
          flush=True)
    return {"sa_valid_rate": valid_rate, "sa_n_reads": int(n_total)}


# ---------------------------------------------------------------------------
# QPU solve
# ---------------------------------------------------------------------------

def solve_task_sudoku(puzzle: np.ndarray, sampler: DWaveSampler,
                      task_id: str) -> dict:
    bqm = build_sudoku_bqm(puzzle)
    n_free = len(bqm.variables)
    n_given = int(np.sum(puzzle != 0))
    print(f"  [bqm] {n_free} free vars  ({n_given} givens fixed)", flush=True)

    # SA baseline (free — no QPU cost)
    sa_stats = run_sa_baseline(bqm, puzzle, task_id)

    emb = load_or_compute_embedding(bqm, sampler, task_id, EMBEDDINGS_DIR)

    max_coupling = max(abs(v) for v in bqm.quadratic.values()) if bqm.quadratic else 1.0
    chain_strength = round(max_coupling * CHAIN_STRENGTH_MULTIPLIER, 2)

    composite = FixedEmbeddingComposite(sampler, emb)
    print(f"  [qpu] submitting  num_reads={NUM_READS}  "
          f"annealing_time={ANNEALING_TIME}us  chain_strength={chain_strength}",
          flush=True)
    t0 = time.time()
    response = composite.sample(bqm, num_reads=NUM_READS,
                                annealing_time=ANNEALING_TIME,
                                chain_strength=chain_strength)
    wall_s = round(time.time() - t0, 2)

    cbf_arr = getattr(response.record, "chain_break_fraction", None)
    mean_cbf = float(np.mean(cbf_arr)) if cbf_arr is not None else None

    n_valid = 0
    for datum in response.data():
        is_valid, _ = validate_sudoku(datum.sample, puzzle)
        if is_valid:
            n_valid += datum.num_occurrences

    n_total = int(sum(d.num_occurrences for d in response.data()))
    valid_rate = round(100.0 * n_valid / n_total, 3) if n_total else 0.0
    timing = response.info.get("timing", {})

    cbf_str = f"  chain_breaks={mean_cbf*100:.1f}%" if mean_cbf is not None else ""
    print(f"  [res] wall={wall_s}s  valid={valid_rate:.1f}%{cbf_str}", flush=True)

    return {
        "task_id":                   task_id,
        "n_givens":                  n_given,
        "n_free_vars":               n_free,
        "n_reads":                   n_total,
        "n_valid":                   n_valid,
        "qpu_valid_rate":            valid_rate,
        "sa_valid_rate":             sa_stats["sa_valid_rate"],
        "mean_chain_break_fraction": mean_cbf,
        "qpu_access_us":             timing.get("qpu_access_time"),
        "wall_time_s":               wall_s,
        "timing":                    timing,
        "solved_at":                 datetime.now().isoformat(),
    }


# ---------------------------------------------------------------------------
# Task list
# ---------------------------------------------------------------------------

def build_tasks(include_9x9: bool = False) -> list:
    tasks = []
    for diff, puzzle in PUZZLES_4x4.items():
        n_free = int((puzzle == 0).sum()) * puzzle.shape[0]
        tid = f"sudoku_4x4_{diff}_CS{CHAIN_STRENGTH_MULTIPLIER}x_AT{ANNEALING_TIME}"
        tasks.append(("4x4", diff, puzzle, tid, n_free))

    if include_9x9:
        for diff, puzzle in PUZZLES_9x9.items():
            n_given = int((puzzle != 0).sum())
            n_free  = int((puzzle == 0).sum()) * puzzle.shape[0]
            tid = f"sudoku_9x9_{diff}_CS{CHAIN_STRENGTH_MULTIPLIER}x_AT{ANNEALING_TIME}"
            tasks.append(("9x9", diff, puzzle, tid, n_free))

    return tasks


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------

def phase_embed(tasks: list, sampler: DWaveSampler) -> None:
    print(f"\n{'='*60}")
    print(f"PHASE: EMBED  ({len(tasks)} tasks)")
    print(f"{'='*60}\n")
    for i, (size, diff, puzzle, tid, n_free) in enumerate(tasks):
        print(f"[{i+1}/{len(tasks)}] {tid}  ({n_free} free vars)", flush=True)
        bqm = build_sudoku_bqm(puzzle)
        load_or_compute_embedding(bqm, sampler, tid, EMBEDDINGS_DIR)
    print(f"\nAll embeddings cached -> {EMBEDDINGS_DIR}", flush=True)


def phase_solve(tasks: list, sampler: DWaveSampler) -> None:
    print(f"\n{'='*60}")
    print(f"PHASE: SOLVE  ({len(tasks)} tasks)")
    print(f"{'='*60}\n")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    total_qpu_us = 0

    for i, (size, diff, puzzle, tid, n_free) in enumerate(tasks):
        result_file = RESULTS_DIR / f"{tid}.json"
        if result_file.exists():
            print(f"[{i+1}/{len(tasks)}] {tid}  SKIP", flush=True)
            with open(result_file) as f:
                total_qpu_us += json.load(f).get("qpu_access_us") or 0
            continue

        print(f"[{i+1}/{len(tasks)}] {tid}", flush=True)
        try:
            result = solve_task_sudoku(puzzle, sampler, tid)
            with open(result_file, "w") as f:
                json.dump(result, f, indent=2)
            total_qpu_us += result.get("qpu_access_us") or 0
        except Exception as exc:
            import traceback
            print(f"  ERROR: {exc}", flush=True)
            traceback.print_exc()

    print(f"\nTotal QPU time: {total_qpu_us/1e6:.3f} s", flush=True)


def phase_analyze(tasks: list) -> None:
    print(f"\n{'='*60}")
    print(f"PHASE: ANALYZE")
    print(f"{'='*60}\n")

    rows_out = []
    for size, diff, puzzle, tid, n_free in tasks:
        result_file = RESULTS_DIR / f"{tid}.json"
        if not result_file.exists():
            print(f"  MISSING: {result_file.name}")
            continue
        with open(result_file) as f:
            r = json.load(f)
        rows_out.append({
            "puzzle":        f"{size}_{diff}",
            "n_givens":      r["n_givens"],
            "n_free_vars":   r["n_free_vars"],
            "sa_valid_pct":  r["sa_valid_rate"],
            "qpu_valid_pct": r["qpu_valid_rate"],
            "chain_break_pct": round((r["mean_chain_break_fraction"] or 0) * 100, 1),
            "qpu_access_us": r.get("qpu_access_us"),
        })

    if not rows_out:
        print("No results.")
        return

    hdr = f"{'Puzzle':<16} {'Givens':>6} {'FreeVars':>9} {'SA%':>6} {'QPU%':>6} {'CB%':>6}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows_out:
        print(f"{r['puzzle']:<16} {r['n_givens']:>6} {r['n_free_vars']:>9} "
              f"{r['sa_valid_pct']:>6.1f} {r['qpu_valid_pct']:>6.1f} "
              f"{r['chain_break_pct']:>6.1f}")

    csv_path = HERE / "sudoku_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows_out[0].keys())
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"\nCSV saved -> {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="QPU Sudoku Experiments")
    parser.add_argument("--phase", required=True,
                        choices=["embed", "solve", "analyze", "all"])
    parser.add_argument("--include-9x9", action="store_true",
                        help="Also embed/solve 9×9 puzzles (large — may fail embedding)")
    args = parser.parse_args()

    tasks = build_tasks(include_9x9=args.include_9x9)
    print(f"{len(tasks)} tasks  |  phase: {args.phase}")
    for _, diff, puzzle, tid, n_free in tasks:
        n_given = int(np.sum(puzzle != 0))
        print(f"  {tid}  ({n_given} givens, {n_free} free vars)")

    phases = ["embed", "solve", "analyze"] if args.phase == "all" else [args.phase]
    sampler = None
    if "embed" in phases or "solve" in phases:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        token   = load_token()
        sampler = DWaveSampler(token=token, solver=SOLVER_ID)
        print(f"Connected to {sampler.solver.name}", flush=True)

    if "embed"   in phases: phase_embed(tasks, sampler)
    if "solve"   in phases: phase_solve(tasks, sampler)
    if "analyze" in phases: phase_analyze(tasks)


if __name__ == "__main__":
    main()
