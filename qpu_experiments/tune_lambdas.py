"""
Lambda Tuning for MJ Protein Folding QUBO
==========================================
Inspired by quantum_folding_2d/experiments/focal_tuning.py.

Three phases:
  1. Diagnostic  — isolate each constraint (E1, E2, E3) by cranking one weight
                   while holding others low; reveals which constraint is hardest.
  2. Ratio sweep — fix overall scale at a few levels, vary λ2/λ1 and λ3/λ1 ratios;
                   ratio balance matters more than absolute scale.
  3. Refinement  — fine-grained multipliers around the Phase-2 winner.

Per-constraint violation rates are tracked (not just overall valid rate) so you
can see *where* SA fails, exactly like focal_tuning's conditional_collision_rate.

Results saved to qpu_experiments/tuning_results/ as CSV + JSON summary.

Usage:
    python qpu_experiments/tune_lambdas.py
    python qpu_experiments/tune_lambdas.py --phases 1,2   # skip refinement
    python qpu_experiments/tune_lambdas.py --top 15
"""

import argparse
import csv
import json
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

try:
    from dwave.samplers import SimulatedAnnealingSampler
except ImportError:
    from dimod.reference.samplers import SimulatedAnnealingSampler

from dimod import BinaryQuadraticModel

import sys
sys.path.insert(0, str(Path(__file__).parent))
from run_qpu import (
    MJ_MATRIX, AA_TO_IDX, SEQ_8,
    build_2d_lattice, build_protein_qubo, validate_solution,
    LATTICES,
)

HERE        = Path(__file__).parent
GT_FILE     = HERE / "ground_truths.json"
OUT_DIR     = HERE / "tuning_results"

SA_SWEEPS   = 10_000
SA_READS    = 100
SA_BETA     = (1.0, 50.0)

# ---------------------------------------------------------------------------
# Phase definitions (inspired by focal_tuning.py)
# ---------------------------------------------------------------------------

def phase1_diagnostic():
    """
    Crank each constraint weight to 50 while holding others at 1.
    Reveals which constraint (E1/E2/E3) is hardest to satisfy at baseline.
    """
    configs = []
    for name, (l1, l2, l3) in [
        ("E1_dominant", (50.0,  1.0,  1.0)),
        ("E2_dominant", ( 1.0, 50.0,  1.0)),
        ("E3_dominant", ( 1.0,  1.0, 50.0)),
        ("balanced_lo", ( 1.5,  2.0,  2.0)),
        ("balanced_md", ( 3.0,  4.0,  4.0)),
        ("balanced_hi", ( 6.0,  8.0,  8.0)),
        ("equal_5",     ( 5.0,  5.0,  5.0)),
        ("equal_10",    (10.0, 10.0, 10.0)),
    ]:
        configs.append({"name": name, "l1": l1, "l2": l2, "l3": l3})
    return configs


def phase2_ratio_sweep(best_p1=None):
    """
    Fix overall scale s ∈ {2, 4, 6, 10}, vary λ2/λ1 and λ3/λ1 ratios.
    Ratio balance matters more than absolute scale (focal_tuning insight).
    """
    scales      = [2.0, 4.0, 6.0, 10.0]
    r2_values   = [0.8, 1.0, 1.33, 1.5, 2.0]   # λ2/λ1
    r3_values   = [0.8, 1.0, 1.33, 1.5, 2.0]   # λ3/λ1

    configs = []
    for s in scales:
        for r2 in r2_values:
            for r3 in r3_values:
                l1 = round(s, 2)
                l2 = round(s * r2, 2)
                l3 = round(s * r3, 2)
                configs.append({
                    "name": f"s{s}_r2{r2}_r3{r3}",
                    "l1": l1, "l2": l2, "l3": l3,
                })

    # Also seed with best from phase 1 if provided
    if best_p1:
        for mult in [0.8, 1.0, 1.2]:
            configs.append({
                "name": f"p1best_x{mult}",
                "l1": round(best_p1["l1"] * mult, 2),
                "l2": round(best_p1["l2"] * mult, 2),
                "l3": round(best_p1["l3"] * mult, 2),
            })

    # Deduplicate
    seen = set()
    unique = []
    for c in configs:
        key = (c["l1"], c["l2"], c["l3"])
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def phase3_refinement(best):
    """Fine multipliers around the Phase-2 winner."""
    multipliers = [0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 2.0]
    configs = []
    for m1 in multipliers:
        for m23 in multipliers:
            configs.append({
                "name": f"ref_m1{m1}_m23{m23}",
                "l1": round(best["l1"] * m1,  2),
                "l2": round(best["l2"] * m23, 2),
                "l3": round(best["l3"] * m23, 2),
            })
    return configs


# ---------------------------------------------------------------------------
# SA evaluation
# ---------------------------------------------------------------------------

def _eval_config(args):
    """Worker function — evaluable by ProcessPoolExecutor."""
    l1, l2, l3, seq, adj_list, gt_E_min = args
    adj = np.array(adj_list)

    linear, quadratic, offset = build_protein_qubo(seq, adj, l1, l2, l3)
    bqm = BinaryQuadraticModel(linear, quadratic, offset, vartype='BINARY')
    result = SimulatedAnnealingSampler().sample(
        bqm, num_reads=SA_READS, num_sweeps=SA_SWEEPS,
        beta_range=SA_BETA, seed=42,
    )

    n_total = sum(d.num_occurrences for d in result.data())
    n_valid = 0
    e1_violations = e2_violations = e3_violations = 0
    best_E_MJ = None

    for datum in result.data():
        _, _, breakdown, _ = validate_solution(datum.sample, seq, adj, l1, l2, l3)
        occ = datum.num_occurrences
        if breakdown['E1'] == 0 and breakdown['E2'] == 0 and breakdown['E3'] == 0:
            n_valid += occ
            e_mj = breakdown['E_MJ']
            if best_E_MJ is None or e_mj < best_E_MJ:
                best_E_MJ = e_mj
        else:
            e1_violations += occ * (breakdown['E1'] > 0)
            e2_violations += occ * (breakdown['E2'] > 0)
            e3_violations += occ * (breakdown['E3'] > 0)

    valid_rate  = 100.0 * n_valid / n_total
    e1_viol_rate = 100.0 * e1_violations / n_total
    e2_viol_rate = 100.0 * e2_violations / n_total
    e3_viol_rate = 100.0 * e3_violations / n_total
    gs_rate = 0.0
    if gt_E_min is not None and best_E_MJ is not None:
        gs_rate = 100.0 if abs(best_E_MJ - gt_E_min) < 0.005 else 0.0

    return {
        "valid_rate": valid_rate, "gs_rate": gs_rate,
        "e1_viol": e1_viol_rate, "e2_viol": e2_viol_rate, "e3_viol": e3_viol_rate,
    }


def evaluate_configs(configs, ground_truths, adj, label, cpus=4):
    """Run SA on every (config, sequence) pair. Returns list of result rows."""
    adj_list = adj.tolist()
    tasks = [
        (c["l1"], c["l2"], c["l3"], seq, adj_list, ground_truths.get(seq, {}).get("E_min"))
        for c in configs
        for seq in SEQ_8
    ]

    raw = {}  # (config_idx, seq) -> metrics
    print(f"  {len(configs)} configs × {len(SEQ_8)} seqs = {len(tasks)} SA runs")

    with ProcessPoolExecutor(max_workers=cpus) as pool:
        futures = {pool.submit(_eval_config, t): i for i, t in enumerate(tasks)}
        done = 0
        for fut in as_completed(futures):
            raw[futures[fut]] = fut.result()
            done += 1
            if done % 20 == 0 or done == len(tasks):
                print(f"    {done}/{len(tasks)}", end="\r", flush=True)

    print()

    rows = []
    for ci, cfg in enumerate(configs):
        seq_results = [raw[ci * len(SEQ_8) + si] for si in range(len(SEQ_8))]
        rows.append({
            "phase":    label,
            "name":     cfg["name"],
            "l1":       cfg["l1"],
            "l2":       cfg["l2"],
            "l3":       cfg["l3"],
            "valid":    float(np.mean([r["valid_rate"] for r in seq_results])),
            "gs":       float(np.mean([r["gs_rate"]    for r in seq_results])),
            "e1_viol":  float(np.mean([r["e1_viol"]    for r in seq_results])),
            "e2_viol":  float(np.mean([r["e2_viol"]    for r in seq_results])),
            "e3_viol":  float(np.mean([r["e3_viol"]    for r in seq_results])),
        })
    return rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_top(rows, n=10, label=""):
    rows_s = sorted(rows, key=lambda r: (-r["valid"], -r["gs"]))
    print(f"\n  {'λ1':>5} {'λ2':>5} {'λ3':>5}  {'valid%':>7}  {'GS%':>6}  "
          f"{'E1v%':>6} {'E2v%':>6} {'E3v%':>6}  name")
    print(f"  {'-'*75}")
    for r in rows_s[:n]:
        print(f"  {r['l1']:5.1f} {r['l2']:5.1f} {r['l3']:5.1f}  "
              f"{r['valid']:7.1f}  {r['gs']:6.1f}  "
              f"{r['e1_viol']:6.1f} {r['e2_viol']:6.1f} {r['e3_viol']:6.1f}  "
              f"{r['name']}")
    return rows_s[0]


def save_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phases", default="1,2,3",
                        help="Comma-separated phases to run (default: 1,2,3)")
    parser.add_argument("--top",    type=int, default=10)
    parser.add_argument("--cpus",   type=int, default=4)
    args = parser.parse_args()

    phases = [int(p) for p in args.phases.split(",")]

    if not GT_FILE.exists():
        print(f"Ground truths not found — run compute_ground_truths.py first.")
        return

    with open(GT_FILE) as f:
        ground_truths = json.load(f)

    rows8, cols8 = LATTICES[8]
    adj = build_2d_lattice(rows8, cols8)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    best_p1 = best_p2 = None
    t0 = time.time()

    # ---- Phase 1: Diagnostic -----------------------------------------------
    if 1 in phases:
        print(f"\n{'='*60}")
        print("PHASE 1 — Diagnostic (which constraint is hardest?)")
        print(f"{'='*60}")
        cfgs = phase1_diagnostic()
        rows = evaluate_configs(cfgs, ground_truths, adj, "phase1", args.cpus)
        all_rows.extend(rows)
        best_p1 = print_top(rows, args.top, "Phase 1")
        save_csv(rows, OUT_DIR / "phase1.csv")
        print(f"\n  → hardest constraint identified by highest violation rate above")

    # ---- Phase 2: Ratio sweep ----------------------------------------------
    if 2 in phases:
        print(f"\n{'='*60}")
        print("PHASE 2 — Ratio sweep (λ2/λ1, λ3/λ1 at multiple scales)")
        print(f"{'='*60}")
        cfgs = phase2_ratio_sweep(best_p1)
        rows = evaluate_configs(cfgs, ground_truths, adj, "phase2", args.cpus)
        all_rows.extend(rows)
        best_p2 = print_top(rows, args.top, "Phase 2")
        save_csv(rows, OUT_DIR / "phase2.csv")

    # ---- Phase 3: Refinement -----------------------------------------------
    if 3 in phases:
        if best_p2 is None:
            print("Phase 2 must run before Phase 3 (needed for best config seed).")
        else:
            print(f"\n{'='*60}")
            print("PHASE 3 — Refinement (fine multipliers around Phase-2 winner)")
            print(f"{'='*60}")
            cfgs = phase3_refinement(best_p2)
            rows = evaluate_configs(cfgs, ground_truths, adj, "phase3", args.cpus)
            all_rows.extend(rows)
            best_final = print_top(rows, args.top, "Phase 3")
            save_csv(rows, OUT_DIR / "phase3.csv")

            print(f"\n{'='*60}")
            print("FINAL RECOMMENDATION")
            print(f"{'='*60}")
            print(f"  SA-optimal:    λ1={best_final['l1']}, "
                  f"λ2={best_final['l2']}, λ3={best_final['l3']}")
            print(f"  QPU-scaled (×1.33):  "
                  f"λ1={best_final['l1']*1.33:.2f}, "
                  f"λ2={best_final['l2']*1.33:.2f}, "
                  f"λ3={best_final['l3']*1.33:.2f}")
            print(f"  valid={best_final['valid']:.1f}%  GS={best_final['gs']:.1f}%")

    save_csv(all_rows, OUT_DIR / "all_phases.csv")
    summary = {"phases_run": phases, "n_configs": len(all_rows),
               "elapsed_min": round((time.time() - t0) / 60, 1)}
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll results → {OUT_DIR}/")
    print(f"Total time: {summary['elapsed_min']} min")


if __name__ == "__main__":
    main()
