"""
QPU Experiments: MJ Protein Folding on D-Wave Advantage2

Run 1 — Weight validation:  8-res, 12 sequences x 3 lambda configs  = 36 tasks (~12.6 s QPU)
Run 2 — Scaling study:      8/12/16-res, 4 seqs x 3 sizes, best lambda = 12 tasks (~4.2 s QPU)

Metrics per task: valid rate, ground-state rate, mean chain break fraction.

Workflow on Sycamore (from repo root):
    source ~/quantum_folding_2d/venv/bin/activate
    cd ~/QA_for_Sudoku && git pull

    # Once:
    python qpu_experiments/compute_ground_truths.py

    # Run 1:
    python qpu_experiments/run_qpu.py --run 1 --phase embed
    nohup python qpu_experiments/run_qpu.py --run 1 --phase solve \\
        > qpu_experiments/logs/run1_solve.log 2>&1 &
    tail -f qpu_experiments/logs/run1_solve.log
    python qpu_experiments/run_qpu.py --run 1 --phase analyze

    # Run 2 (pick best lambda from run 1 analyze output):
    python qpu_experiments/run_qpu.py --run 2 --best-lambda 3.0,4.0,4.0 --phase embed
    nohup python qpu_experiments/run_qpu.py --run 2 --best-lambda 3.0,4.0,4.0 --phase solve \\
        > qpu_experiments/logs/run2_solve.log 2>&1 &
    python qpu_experiments/run_qpu.py --run 2 --best-lambda 3.0,4.0,4.0 --phase analyze
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

# D-Wave Ocean SDK
from dwave.system import DWaveSampler, FixedEmbeddingComposite
from dwave.embedding.chain_strength import uniform_torque_compensation
from minorminer import find_embedding
import dimod
from dimod import BinaryQuadraticModel, SimulatedAnnealingSampler

# Allow imports from repo root (gs_strats lives there)
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SOLVER_ID       = "Advantage2_system1.13"
NUM_READS       = 1000
ANNEALING_TIME  = 200   # microseconds (20 gave 0% valid; 10x increase)
EMBED_TIMEOUT   = 600   # seconds (Zephyr may take longer on first attempt)
EMBED_TRIES     = 50
EMBED_SEED      = 42

HERE            = Path(__file__).parent
RESULTS_DIR     = HERE / "results"
EMBEDDINGS_DIR  = HERE / "embeddings"
LOGS_DIR        = HERE / "logs"
GT_FILE         = HERE / "ground_truths.json"

# ---------------------------------------------------------------------------
# Amino acid data (from benchmark_8res_complete notebook)
# ---------------------------------------------------------------------------

AMINO_ACIDS = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
               'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

MJ_MATRIX = np.array([
    [-0.20,-0.34,-0.44,-0.42,-0.62,-0.40,-0.42,-0.29,-0.51,-0.73,-0.65,-0.36,-0.64,-0.69,-0.31,-0.34,-0.40,-0.61,-0.53,-0.68],
    [-0.34,-0.15,-0.44,-0.90,-0.55,-0.50,-0.93,-0.40,-0.33,-0.42,-0.45,-0.35,-0.34,-0.41,-0.24,-0.37,-0.32,-0.26,-0.29,-0.37],
    [-0.44,-0.44,-0.28,-0.46,-0.58,-0.44,-0.51,-0.46,-0.54,-0.51,-0.47,-0.46,-0.41,-0.47,-0.34,-0.48,-0.50,-0.36,-0.42,-0.46],
    [-0.42,-0.90,-0.46,-0.19,-0.52,-0.53,-0.49,-0.44,-0.49,-0.42,-0.35,-0.62,-0.33,-0.36,-0.31,-0.48,-0.44,-0.23,-0.30,-0.36],
    [-0.62,-0.55,-0.58,-0.52,-1.54,-0.65,-0.55,-0.54,-0.73,-0.92,-0.88,-0.47,-0.92,-1.01,-0.49,-0.56,-0.61,-0.84,-0.79,-0.91],
    [-0.40,-0.50,-0.44,-0.53,-0.65,-0.29,-0.55,-0.40,-0.49,-0.56,-0.51,-0.46,-0.47,-0.54,-0.31,-0.43,-0.41,-0.39,-0.42,-0.51],
    [-0.42,-0.93,-0.51,-0.49,-0.55,-0.55,-0.22,-0.43,-0.42,-0.44,-0.37,-0.64,-0.35,-0.40,-0.30,-0.47,-0.43,-0.21,-0.27,-0.38],
    [-0.29,-0.40,-0.46,-0.44,-0.54,-0.40,-0.43,-0.14,-0.46,-0.52,-0.46,-0.41,-0.47,-0.51,-0.26,-0.37,-0.39,-0.44,-0.44,-0.47],
    [-0.51,-0.33,-0.54,-0.49,-0.73,-0.49,-0.42,-0.46,-0.34,-0.66,-0.62,-0.35,-0.64,-0.72,-0.37,-0.46,-0.47,-0.62,-0.57,-0.60],
    [-0.73,-0.42,-0.51,-0.42,-0.92,-0.56,-0.44,-0.52,-0.66,-1.00,-0.95,-0.41,-0.96,-1.04,-0.49,-0.52,-0.59,-0.90,-0.83,-0.98],
    [-0.65,-0.45,-0.47,-0.35,-0.88,-0.51,-0.37,-0.46,-0.62,-0.95,-0.91,-0.37,-0.93,-1.01,-0.46,-0.47,-0.53,-0.89,-0.80,-0.93],
    [-0.36,-0.35,-0.46,-0.62,-0.47,-0.46,-0.64,-0.41,-0.35,-0.41,-0.37,-0.18,-0.32,-0.39,-0.23,-0.39,-0.36,-0.19,-0.24,-0.35],
    [-0.64,-0.34,-0.41,-0.33,-0.92,-0.47,-0.35,-0.47,-0.64,-0.96,-0.93,-0.32,-0.89,-0.98,-0.45,-0.47,-0.52,-0.86,-0.78,-0.92],
    [-0.69,-0.41,-0.47,-0.36,-1.01,-0.54,-0.40,-0.51,-0.72,-1.04,-1.01,-0.39,-0.98,-1.10,-0.50,-0.51,-0.57,-0.98,-0.89,-1.02],
    [-0.31,-0.24,-0.34,-0.31,-0.49,-0.31,-0.30,-0.26,-0.37,-0.49,-0.46,-0.23,-0.45,-0.50,-0.16,-0.30,-0.32,-0.42,-0.38,-0.45],
    [-0.34,-0.37,-0.48,-0.48,-0.56,-0.43,-0.47,-0.37,-0.46,-0.52,-0.47,-0.39,-0.47,-0.51,-0.30,-0.33,-0.41,-0.41,-0.41,-0.48],
    [-0.40,-0.32,-0.50,-0.44,-0.61,-0.41,-0.43,-0.39,-0.47,-0.59,-0.53,-0.36,-0.52,-0.57,-0.32,-0.41,-0.38,-0.45,-0.45,-0.55],
    [-0.61,-0.26,-0.36,-0.23,-0.84,-0.39,-0.21,-0.44,-0.62,-0.90,-0.89,-0.19,-0.86,-0.98,-0.42,-0.41,-0.45,-0.86,-0.76,-0.84],
    [-0.53,-0.29,-0.42,-0.30,-0.79,-0.42,-0.27,-0.44,-0.57,-0.83,-0.80,-0.24,-0.78,-0.89,-0.38,-0.41,-0.45,-0.76,-0.69,-0.78],
    [-0.68,-0.37,-0.46,-0.36,-0.91,-0.51,-0.38,-0.47,-0.60,-0.98,-0.93,-0.35,-0.92,-1.02,-0.45,-0.48,-0.55,-0.84,-0.78,-0.96],
])

# ---------------------------------------------------------------------------
# Sequences
# ---------------------------------------------------------------------------

SEQ_8 = [                   # Run 1: all 12 eight-residue benchmark sequences
    "CFIKDLWV", "ILFVWMCY", "CFWVILMY", "WFYCMILV",
    "ACDEFGHI", "KLMNPQRS", "STVWYACF", "RKDECFWI",
    "RKDENQST", "HHKKRRDD", "SSTTNNQQ", "GPAGPSAG",
]

SEQ_8_RUN2  = ["CFIKDLWV", "ILFVWMCY", "WFYCMILV", "RKDECFWI"]

SEQ_12_RUN2 = [
    "CFIKDLWVRKDE",   # hydrophobic core + charged
    "ILFVWMCYHHKK",   # hydrophobic + basic
    "WFYCMILVSSTT",   # aromatic + polar
    "ACDEFGHIKLMN",   # alphabetically diverse
]

SEQ_16_RUN2 = [
    "CFIKDLWVRKDENQST",
    "ILFVWMCYHHKKRRDD",
    "WFYCMILVSSTTNNQQ",
    "ACDEFGHIKLMNPQRS",
]

LATTICES = {8: (2, 4), 12: (3, 4), 16: (4, 4)}

# Lambda configs per proposal: SA-direct, QPU-1.33x, Irback-style
LAMBDA_CONFIGS = [
    (5.2, 6.12, 6.9),   # SA-tuned optimum  (99.3% valid on SA)
    (6.9, 8.14, 9.18),  # QPU-scaled 1.33x  (harder constraints for noisy QPU landscape)
    (3.4, 4.52, 5.1),   # Conservative      (99.2% valid, lower absolute values = lower chain strength)
]
LAMBDA_NAMES = {
    (5.2, 6.12, 6.9):   "SA_tuned",
    (6.9, 8.14, 9.18):  "QPU_133x",
    (3.4, 4.52, 5.1):   "Conservative",
}

# ---------------------------------------------------------------------------
# QUBO construction (from benchmark_8res_complete notebook)
# ---------------------------------------------------------------------------

def build_2d_lattice(rows: int, cols: int) -> np.ndarray:
    n = rows * cols
    adj = np.zeros((n, n), dtype=int)
    for i in range(n):
        r, c = i // cols, i % cols
        if c < cols - 1:
            adj[i, i + 1] = adj[i + 1, i] = 1
        if r < rows - 1:
            adj[i, i + cols] = adj[i + cols, i] = 1
    return adj


def build_protein_qubo(
    sequence: str,
    adj: np.ndarray,
    lambda1: float,
    lambda2: float,
    lambda3: float,
) -> tuple[dict, dict, float]:
    """Return (linear, quadratic, offset) dicts for use with dimod BQM."""
    N, M = len(sequence), adj.shape[0]
    assert N == M, f"len(sequence)={N} must equal lattice size={M}"

    linear:    dict = defaultdict(float)
    quadratic: dict = defaultdict(float)
    offset = 0.0

    def bit(r: int, p: int) -> int:
        return r * M + p

    # E_MJ: Miyazawa-Jernigan interactions (non-consecutive residue pairs)
    for i in range(N):
        for j in range(i + 2, N):
            C_ij = MJ_MATRIX[AA_TO_IDX[sequence[i]], AA_TO_IDX[sequence[j]]]
            for n in range(M):
                for m in range(M):
                    if adj[n, m] == 1:
                        b_i, b_j = bit(i, n), bit(j, m)
                        quadratic[(min(b_i, b_j), max(b_i, b_j))] += C_ij

    # E1: one position per residue  (lambda1)
    for i in range(N):
        for n in range(M):
            linear[bit(i, n)] += lambda1 * (-1)
        for n in range(M):
            for m in range(n + 1, M):
                quadratic[(bit(i, n), bit(i, m))] += lambda1 * 2
        offset += lambda1

    # E2: self-avoidance — one residue per position  (lambda2)
    for n in range(M):
        for i in range(N):
            for j in range(i + 1, N):
                quadratic[(bit(i, n), bit(j, n))] += lambda2

    # E3: chain connectivity — penalise non-adjacent consecutive placement  (lambda3)
    non_adj = 1 - adj - np.eye(M)
    for i in range(N - 1):
        for n in range(M):
            for m in range(M):
                if non_adj[n, m] == 1:
                    b_i, b_j = bit(i, n), bit(i + 1, m)
                    quadratic[(min(b_i, b_j), max(b_i, b_j))] += lambda3

    return dict(linear), dict(quadratic), offset


def validate_solution(
    sample: dict,
    sequence: str,
    adj: np.ndarray,
    lambda1: float,
    lambda2: float,
    lambda3: float,
) -> tuple[bool, float, dict, list | None]:
    """
    Validate a binary sample against all QUBO constraints.

    Returns:
        is_valid, E_total, breakdown dict, path (list of positions) or None.
    """
    N, M = len(sequence), adj.shape[0]
    b = np.zeros((N, M))
    for bit_idx, val in sample.items():
        if val == 1:
            i, n = int(bit_idx) // M, int(bit_idx) % M
            if i < N and n < M:
                b[i, n] = 1

    E1 = int(np.sum((np.sum(b, axis=1) - 1) ** 2))
    E2 = sum(
        int(np.sum(b[:, n]) * (np.sum(b[:, n]) - 1) / 2)
        for n in range(M)
    )
    non_adj = 1 - adj - np.eye(M)
    E3 = int(sum(b[i, :] @ non_adj @ b[i + 1, :] for i in range(N - 1)))

    E_MJ = sum(
        MJ_MATRIX[AA_TO_IDX[sequence[i]], AA_TO_IDX[sequence[j]]]
        for i in range(N)
        for j in range(i + 2, N)
        for n in range(M)
        for m in range(M)
        if b[i, n] == 1 and b[j, m] == 1 and adj[n, m] == 1
    )

    is_valid = (E1 == 0 and E2 == 0 and E3 == 0)
    path = None
    if is_valid:
        path = [int(np.argmax(b[i, :])) for i in range(N)]

    E_total = E_MJ + lambda1 * E1 + lambda2 * E2 + lambda3 * E3
    return is_valid, E_total, {"E_MJ": E_MJ, "E1": E1, "E2": E2, "E3": E3}, path


# ---------------------------------------------------------------------------
# D-Wave helpers
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
        "DWAVE_API_TOKEN not found. Set it in the environment or in "
        "repo_root/.env as DWAVE_API_TOKEN=DEV-..."
    )


def load_or_compute_embedding(
    bqm: BinaryQuadraticModel,
    sampler: DWaveSampler,
    task_id: str,
    cache_dir: Path,
) -> dict:
    """Return cached embedding or compute and cache a new one (free — no QPU)."""
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

    print(f"  [emb] computing (timeout={EMBED_TIMEOUT}s, tries={EMBED_TRIES})...",
          flush=True)
    source_edges = list(bqm.quadratic.keys())
    t0 = time.time()
    emb = find_embedding(
        source_edges, sampler.edgelist,
        verbose=0,
        timeout=EMBED_TIMEOUT,
        tries=EMBED_TRIES,
        random_seed=EMBED_SEED,
    )
    elapsed = time.time() - t0

    if not emb:
        raise RuntimeError(f"Embedding failed for {task_id} after {elapsed:.0f}s")

    chain_lengths = [len(c) for c in emb.values()]
    stats = {
        "physical_qubits": len({q for c in emb.values() for q in c}),
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
        json.dump({
            "task_id":    task_id,
            "embedding":  {k: list(v) for k, v in emb.items()},
            "stats":      stats,
            "created_at": datetime.now().isoformat(),
        }, f, indent=2)

    return emb


def solve_task(
    sequence: str,
    adj: np.ndarray,
    lambdas: tuple,
    sampler: DWaveSampler,
    task_id: str,
) -> dict:
    """Build BQM, load embedding, solve on QPU, return full result dict."""
    l1, l2, l3 = lambdas
    linear, quadratic, offset = build_protein_qubo(sequence, adj, l1, l2, l3)
    bqm = BinaryQuadraticModel(linear, quadratic, offset, vartype="BINARY")
    print(f"  [bqm] {len(bqm.variables)} vars  {len(bqm.quadratic)} quadratic terms",
          flush=True)

    emb = load_or_compute_embedding(bqm, sampler, task_id, EMBEDDINGS_DIR)

    # Chain strength: set to 1× max QUBO coupling.
    # D-Wave normalizes all couplings by chain_strength, so hardware QUBO values
    # max out at 1.0 (same as chain coupling). This maximizes constraint signal
    # (~10× kT* vs ~5× at 2×) at the cost of slightly higher chain breaks.
    # uniform_torque_compensation gives ~5×, which buries constraints near kT*.
    max_coupling = max(abs(v) for v in bqm.quadratic.values())
    chain_strength = round(max_coupling * CHAIN_STRENGTH_MULTIPLIER, 2)

    composite = FixedEmbeddingComposite(sampler, emb)
    print(f"  [qpu] submitting  num_reads={NUM_READS}  "
          f"annealing_time={ANNEALING_TIME}us  chain_strength={chain_strength}", flush=True)
    t0 = time.time()
    response = composite.sample(bqm, num_reads=NUM_READS, annealing_time=ANNEALING_TIME,
                                chain_strength=chain_strength)
    wall_s = round(time.time() - t0, 2)

    # Chain break fraction (field added by FixedEmbeddingComposite)
    cbf_arr = getattr(response.record, "chain_break_fraction", None)
    mean_cbf = float(np.mean(cbf_arr)) if cbf_arr is not None else None

    # Validate each unique sample, weighted by num_occurrences
    valid_samples = []
    for datum in response.data():
        is_valid, E_total, breakdown, path = validate_solution(
            datum.sample, sequence, adj, l1, l2, l3
        )
        if is_valid:
            valid_samples.append({
                "E_MJ":            breakdown["E_MJ"],
                "num_occurrences": datum.num_occurrences,
                "path":            path,
            })

    n_total = int(sum(d.num_occurrences for d in response.data()))
    n_valid = int(sum(s["num_occurrences"] for s in valid_samples))
    valid_rate = round(100.0 * n_valid / n_total, 3) if n_total else 0.0
    best_E_MJ  = min((s["E_MJ"] for s in valid_samples), default=None)

    timing = response.info.get("timing", {})
    print(f"  [res] wall={wall_s}s  "
          f"valid={valid_rate:.1f}%  "
          f"chain_breaks={mean_cbf*100:.1f}%  " if mean_cbf is not None
          else f"  [res] wall={wall_s}s  valid={valid_rate:.1f}%  ",
          flush=True)

    return {
        "task_id":                  task_id,
        "sequence":                 sequence,
        "n_residues":               len(sequence),
        "lambdas":                  list(lambdas),
        "n_reads":                  n_total,
        "n_valid":                  n_valid,
        "valid_rate":               valid_rate,
        "best_E_MJ":                best_E_MJ,
        "mean_chain_break_fraction": mean_cbf,
        "qpu_access_us":            timing.get("qpu_access_time"),
        "wall_time_s":              wall_s,
        "timing":                   timing,
        "solved_at":                datetime.now().isoformat(),
    }


# ---------------------------------------------------------------------------
# Reverse annealing helpers
# ---------------------------------------------------------------------------

def find_valid_sa_state(
    sequence: str,
    adj: np.ndarray,
    lambdas: tuple,
    max_reads: int = 200,
) -> dict | None:
    """
    Run SA to find a valid protein-fold starting state for reverse annealing.
    Returns a {var_idx: 0|1} dict, or None if no valid state found.
    """
    l1, l2, l3 = lambdas
    linear, quadratic, offset = build_protein_qubo(sequence, adj, l1, l2, l3)
    bqm = BinaryQuadraticModel(linear, quadratic, offset, vartype="BINARY")
    sa = SimulatedAnnealingSampler()
    response = sa.sample(bqm, num_reads=max_reads, num_sweeps=10_000,
                         beta_range=(0.1, 5.0))
    for datum in response.data():
        is_valid, _, _, _ = validate_solution(datum.sample, sequence, adj, l1, l2, l3)
        if is_valid:
            return dict(datum.sample)
    return None


def solve_task_reverse(
    sequence: str,
    adj: np.ndarray,
    lambdas: tuple,
    sampler: DWaveSampler,
    task_id: str,
) -> dict:
    """
    Reverse annealing: start from a SA-valid conformation, pause at s=0.45,
    let QPU search nearby for lower-energy valid states, re-anneal.

    Valid rates should be much higher than forward annealing since we start
    in the valid manifold and search locally via quantum tunneling.
    """
    l1, l2, l3 = lambdas
    linear, quadratic, offset = build_protein_qubo(sequence, adj, l1, l2, l3)
    bqm = BinaryQuadraticModel(linear, quadratic, offset, vartype="BINARY")
    print(f"  [bqm] {len(bqm.variables)} vars  {len(bqm.quadratic)} quadratic terms",
          flush=True)

    # Find valid starting state via SA
    print("  [sa]  finding valid initial state...", flush=True)
    initial_state = find_valid_sa_state(sequence, adj, lambdas)
    if initial_state is None:
        raise RuntimeError(
            f"SA could not find a valid state for {sequence} — "
            "try increasing max_reads or adjusting lambdas"
        )
    print("  [sa]  valid initial state found", flush=True)

    emb = load_or_compute_embedding(bqm, sampler, task_id, EMBEDDINGS_DIR)

    max_coupling = max(abs(v) for v in bqm.quadratic.values())
    chain_strength = round(max_coupling * CHAIN_STRENGTH_MULTIPLIER, 2)

    composite = FixedEmbeddingComposite(sampler, emb)
    print(f"  [qpu] reverse anneal  num_reads={NUM_READS}  "
          f"schedule={REVERSE_ANNEAL_SCHEDULE}  chain_strength={chain_strength}",
          flush=True)
    t0 = time.time()
    response = composite.sample(
        bqm,
        num_reads=NUM_READS,
        anneal_schedule=REVERSE_ANNEAL_SCHEDULE,
        initial_state=initial_state,
        reinitialize_state=True,
        chain_strength=chain_strength,
    )
    wall_s = round(time.time() - t0, 2)

    cbf_arr = getattr(response.record, "chain_break_fraction", None)
    mean_cbf = float(np.mean(cbf_arr)) if cbf_arr is not None else None

    valid_samples = []
    for datum in response.data():
        is_valid, E_total, breakdown, path = validate_solution(
            datum.sample, sequence, adj, l1, l2, l3
        )
        if is_valid:
            valid_samples.append({
                "E_MJ":            breakdown["E_MJ"],
                "num_occurrences": datum.num_occurrences,
                "path":            path,
            })

    n_total = int(sum(d.num_occurrences for d in response.data()))
    n_valid = int(sum(s["num_occurrences"] for s in valid_samples))
    valid_rate = round(100.0 * n_valid / n_total, 3) if n_total else 0.0
    best_E_MJ  = min((s["E_MJ"] for s in valid_samples), default=None)

    timing = response.info.get("timing", {})
    cbf_str = f"  chain_breaks={mean_cbf*100:.1f}%" if mean_cbf is not None else ""
    print(f"  [res] wall={wall_s}s  valid={valid_rate:.1f}%{cbf_str}", flush=True)

    return {
        "task_id":                   task_id,
        "mode":                      "reverse",
        "sequence":                  sequence,
        "n_residues":                len(sequence),
        "lambdas":                   list(lambdas),
        "anneal_schedule":           REVERSE_ANNEAL_SCHEDULE,
        "n_reads":                   n_total,
        "n_valid":                   n_valid,
        "valid_rate":                valid_rate,
        "best_E_MJ":                 best_E_MJ,
        "mean_chain_break_fraction": mean_cbf,
        "qpu_access_us":             timing.get("qpu_access_time"),
        "wall_time_s":               wall_s,
        "timing":                    timing,
        "solved_at":                 datetime.now().isoformat(),
    }


# ---------------------------------------------------------------------------
# Ground truth loader
# ---------------------------------------------------------------------------

def load_ground_truths() -> dict:
    """Load precomputed ground truths. Key: '{n_residues}_{sequence}'."""
    if not GT_FILE.exists():
        raise FileNotFoundError(
            f"{GT_FILE} not found — run compute_ground_truths.py first"
        )
    with open(GT_FILE) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Task builders
# ---------------------------------------------------------------------------

CHAIN_STRENGTH_MULTIPLIER = 1  # × max_coupling; baked into task_id to prevent collisions

# Reverse annealing: pause at s=0.45 for 100µs then re-anneal
# Total schedule time ~120µs; annealing_time is ignored when anneal_schedule is set
REVERSE_ANNEAL_SCHEDULE = [
    [0.0,   1.0],   # start fully annealed (at initial_state)
    [10.0,  0.45],  # reverse to s=0.45 in 10µs
    [110.0, 0.45],  # pause 100µs at s=0.45 (thermal fluctuations search nearby)
    [120.0, 1.0],   # re-anneal to s=1 in 10µs
]


def _task_id(prefix: str, seq: str, lambdas: tuple, mode: str = "forward") -> str:
    lname = LAMBDA_NAMES.get(lambdas, f"{lambdas[0]}_{lambdas[1]}_{lambdas[2]}")
    suffix = "_RA" if mode == "reverse" else f"_AT{ANNEALING_TIME}"
    return f"{prefix}_{seq}_{lname}_CS{CHAIN_STRENGTH_MULTIPLIER}x{suffix}"


def build_run1_tasks(mode: str = "forward") -> list:
    rows, cols = LATTICES[8]
    adj = build_2d_lattice(rows, cols)
    return [
        (seq, adj, lam, _task_id("r1", seq, lam, mode))
        for seq in SEQ_8
        for lam in LAMBDA_CONFIGS
    ]


def build_run2_tasks(best_lambdas: tuple, mode: str = "forward") -> list:
    tasks = []
    for size, seqs in [(8, SEQ_8_RUN2), (12, SEQ_12_RUN2), (16, SEQ_16_RUN2)]:
        rows, cols = LATTICES[size]
        adj = build_2d_lattice(rows, cols)
        for seq in seqs:
            tid = _task_id(f"r2_{size}res", seq, best_lambdas, mode)
            tasks.append((seq, adj, best_lambdas, tid))
    return tasks


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------

def phase_embed(tasks: list, sampler: DWaveSampler) -> None:
    print(f"\n{'='*60}")
    print(f"PHASE: EMBED  ({len(tasks)} tasks)")
    print(f"{'='*60}\n")
    for i, (seq, adj, lam, tid) in enumerate(tasks):
        print(f"[{i+1}/{len(tasks)}] {tid}", flush=True)
        l1, l2, l3 = lam
        linear, quadratic, offset = build_protein_qubo(seq, adj, l1, l2, l3)
        bqm = BinaryQuadraticModel(linear, quadratic, offset, vartype="BINARY")
        load_or_compute_embedding(bqm, sampler, tid, EMBEDDINGS_DIR)
    print(f"\nAll embeddings cached -> {EMBEDDINGS_DIR}", flush=True)


def phase_solve(tasks: list, sampler: DWaveSampler, mode: str = "forward") -> None:
    print(f"\n{'='*60}")
    print(f"PHASE: SOLVE  ({len(tasks)} tasks)  mode={mode}")
    print(f"{'='*60}\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    total_qpu_us = 0
    done = 0

    for i, (seq, adj, lam, tid) in enumerate(tasks):
        result_file = RESULTS_DIR / f"{tid}.json"
        if result_file.exists():
            print(f"[{i+1}/{len(tasks)}] {tid}  SKIP (already done)", flush=True)
            with open(result_file) as f:
                cached = json.load(f)
            total_qpu_us += cached.get("qpu_access_us") or 0
            done += 1
            continue

        print(f"[{i+1}/{len(tasks)}] {tid}", flush=True)
        try:
            if mode == "reverse":
                result = solve_task_reverse(seq, adj, lam, sampler, tid)
            else:
                result = solve_task(seq, adj, lam, sampler, tid)
            with open(result_file, "w") as f:
                json.dump(result, f, indent=2)
            total_qpu_us += result.get("qpu_access_us") or 0
            done += 1
        except Exception as exc:
            import traceback
            print(f"  ERROR: {exc}", flush=True)
            traceback.print_exc()

    print(f"\nCompleted: {done}/{len(tasks)}", flush=True)
    print(f"Total QPU time: {total_qpu_us/1e6:.3f} s", flush=True)


def phase_analyze(tasks: list, run_label: str) -> None:
    print(f"\n{'='*60}")
    print(f"PHASE: ANALYZE  ({run_label})")
    print(f"{'='*60}\n")

    gts = load_ground_truths()
    rows_out = []

    for seq, adj, lam, tid in tasks:
        result_file = RESULTS_DIR / f"{tid}.json"
        if not result_file.exists():
            print(f"  MISSING: {result_file.name}")
            continue
        with open(result_file) as f:
            r = json.load(f)

        gt_key = f"{len(seq)}_{seq}"
        gt = gts.get(gt_key, {})
        E_min = gt.get("E_min")

        gs_rate = None
        if E_min is not None and r["best_E_MJ"] is not None:
            gs_rate = 100.0 if abs(r["best_E_MJ"] - E_min) < 0.005 else 0.0
        gs_flag = "" if (gt.get("completed", True)) else "*"  # * = partial enum

        cbf_pct = (r["mean_chain_break_fraction"] or 0.0) * 100
        rows_out.append({
            "task_id":        tid,
            "sequence":       seq,
            "n_residues":     len(seq),
            "lambda_name":    LAMBDA_NAMES.get(tuple(lam), str(lam)),
            "valid_rate":     r["valid_rate"],
            "gs_rate":        gs_rate,
            "gs_exact":       gt.get("completed", True),
            "chain_break_pct": round(cbf_pct, 2),
            "best_E_MJ":      r["best_E_MJ"],
            "E_min":          E_min,
            "qpu_access_us":  r.get("qpu_access_us"),
        })

    if not rows_out:
        print("No results found.")
        return

    # Print table
    hdr = f"{'Task':<38} {'Valid%':>7} {'GS%':>6} {'CB%':>6} {'E_best':>9} {'E_min':>9}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows_out:
        gs_str = f"{r['gs_rate']:5.0f}{gs_flag}" if r["gs_rate"] is not None else "   n/a"
        eb = f"{r['best_E_MJ']:9.4f}" if r["best_E_MJ"] is not None else "      n/a"
        em = f"{r['E_min']:9.4f}"     if r["E_min"]     is not None else "      n/a"
        print(f"{r['task_id']:<38} {r['valid_rate']:7.1f} {gs_str} "
              f"{r['chain_break_pct']:6.1f} {eb} {em}")

    # Aggregate by lambda config (for Run 1 comparison)
    if rows_out[0]["n_residues"] == 8 and len(set(r["lambda_name"] for r in rows_out)) > 1:
        print(f"\n--- Lambda config summary (mean over {len(SEQ_8)} sequences) ---")
        for lname in [LAMBDA_NAMES[lam] for lam in LAMBDA_CONFIGS]:
            subset = [r for r in rows_out if r["lambda_name"] == lname]
            if not subset:
                continue
            avg_valid = np.mean([r["valid_rate"] for r in subset])
            gs_vals   = [r["gs_rate"] for r in subset if r["gs_rate"] is not None]
            avg_gs    = np.mean(gs_vals) if gs_vals else float("nan")
            avg_cb    = np.mean([r["chain_break_pct"] for r in subset])
            print(f"  {lname:<14}  valid={avg_valid:5.1f}%  gs={avg_gs:5.1f}%  cb={avg_cb:5.1f}%")
        print("(* = ground truth is a lower bound — enumeration hit budget cap)")

    # Aggregate by size (for Run 2 scaling figure)
    if len(set(r["n_residues"] for r in rows_out)) > 1:
        print(f"\n--- Scaling summary ---")
        for size in [8, 12, 16]:
            subset = [r for r in rows_out if r["n_residues"] == size]
            if not subset:
                continue
            avg_valid = np.mean([r["valid_rate"] for r in subset])
            avg_cb    = np.mean([r["chain_break_pct"] for r in subset])
            gs_vals   = [r["gs_rate"] for r in subset if r["gs_rate"] is not None]
            avg_gs    = np.mean(gs_vals) if gs_vals else float("nan")
            print(f"  {size:>2}-res: valid={avg_valid:5.1f}%  gs={avg_gs:5.1f}%  cb={avg_cb:5.1f}%")

    # Save CSV
    csv_path = HERE / f"{run_label}_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows_out[0].keys())
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"\nCSV saved -> {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="QPU MJ Protein Folding Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run",  type=int, required=True, choices=[1, 2])
    parser.add_argument("--phase", required=True,
                        choices=["embed", "solve", "analyze", "all"])
    parser.add_argument("--mode", default="forward", choices=["forward", "reverse"],
                        help="forward: standard annealing; reverse: start from SA-valid state")
    parser.add_argument("--best-lambda", default="3.0,4.0,4.0",
                        help="For --run 2: comma-separated l1,l2,l3 (e.g. 3.0,4.0,4.0)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only run the first N tasks (for smoke-testing)")
    args = parser.parse_args()

    best_lambdas = tuple(float(x) for x in args.best_lambda.split(","))
    if len(best_lambdas) != 3:
        parser.error("--best-lambda must be three comma-separated floats")

    if args.run == 1:
        tasks      = build_run1_tasks(mode=args.mode)
        run_label  = f"run1_weight_validation_{args.mode}"
    else:
        if best_lambdas not in LAMBDA_NAMES:
            LAMBDA_NAMES[best_lambdas] = "_".join(str(x) for x in best_lambdas)
        tasks      = build_run2_tasks(best_lambdas, mode=args.mode)
        run_label  = f"run2_scaling_{args.best_lambda.replace(',', '_')}_{args.mode}"

    if args.limit:
        tasks = tasks[:args.limit]
    print(f"Run {args.run}: {len(tasks)} tasks  |  phase: {args.phase}  |  mode: {args.mode}")

    phases = ["embed", "solve", "analyze"] if args.phase == "all" else [args.phase]

    sampler = None
    if "embed" in phases or "solve" in phases:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        token   = load_token()
        sampler = DWaveSampler(token=token, solver=SOLVER_ID)
        print(f"Connected to {sampler.solver.name}", flush=True)

    if "embed"   in phases: phase_embed(tasks, sampler)
    if "solve"   in phases: phase_solve(tasks, sampler, mode=args.mode)
    if "analyze" in phases: phase_analyze(tasks, run_label)


if __name__ == "__main__":
    main()
