# 🧬 Protein Folding via QUBO Formulation

A comprehensive demonstration of how to formulate the **lattice protein folding problem** as a **Quadratic Unconstrained Binary Optimization (QUBO)** problem, suitable for quantum annealing, QAOA, or classical optimization methods.

---

## 🌱 What This Project Does

This project systematically transforms the protein folding problem into a QUBO formulation through:

1. **Problem Definition**: Simplified 2D lattice model with 3 amino acid types (H, P, C)
2. **Mathematical Formulation**: Rigorous derivation of energy terms and constraints
3. **QUBO Construction**: Step-by-step conversion of formulas into matrices
4. **Validation**: Verification that the QUBO correctly encodes the problem
5. **Scalability Analysis**: Demonstration from small (4 residues) to larger (6 residues) chains

The core innovation is showing how **biological constraints** (chain connectivity, self-avoidance) and **optimization objectives** (Miyazawa-Jernigan interaction energy) can be unified into a single matrix suitable for quantum or classical solvers.

---

## 📊 Project Structure
```
protein-folding-qubo/
├── problem_formation_and_evaluation/
│   ├── energy_calc/
│   │   └── calc_mods.py              # Direct energy calculation functions
│   └── QUBO_construction/
│       ├── qubo_generation.py        # QUBO matrix building functions
│       └── construction_test.py      # Test suite (validates correctness)
├── notebooks/
│   └── protein_folding_qubo.ipynb    # Main interactive demonstration
└── README.md
```

---

## 🔍 Key Features

### Energy Formulation

The total energy combines:

$$E_{\text{total}} = E_{\text{MJ}} + \lambda_1 E_1 + \lambda_2 E_2 + \lambda_3 E_3$$

Where:
- **E_MJ**: Miyazawa-Jernigan interaction energy (biological objective)
- **E₁**: One position per residue constraint
- **E₂**: Self-avoidance constraint (no overlaps)
- **E₃**: Chain connectivity constraint

### Binary Encoding

Each configuration is encoded as binary variables:
- $b_{i,n} = 1$ if residue $i$ occupies position $n$
- $b_{i,n} = 0$ otherwise

For N residues and M positions: **N × M binary variables**

### QUBO Matrix

All energy terms are combined into a single matrix **Q** such that:

$$E(x) = \text{offset} + \sum_i Q_{ii} x_i + \sum_{i<j} Q_{ij} x_i x_j$$

---

## ⚙️ Getting Started

### Requirements
```bash
pip install numpy matplotlib seaborn jupyter
```

### Running the Notebook

#### Option 1: Local Jupyter
```bash
# Clone the repository
git clone <your-repo-url>
cd protein-folding-qubo

# Launch Jupyter
jupyter notebook

# Open notebooks/protein_folding_qubo.ipynb
```

#### Option 2: Google Colab

1. Upload the following files to Colab:
   - `notebooks/protein_folding_qubo.ipynb`
   - `problem_formation_and_evaluation/QUBO_construction/qubo_generation.py`
   - `problem_formation_and_evaluation/energy_calc/calc_mods.py`

2. In Colab, make sure the `.py` files are in the same directory or adjust the import paths:
```python
   # At the top of the notebook, add if needed:
   from google.colab import files
   uploaded = files.upload()  # Upload qubo_generation.py and calc_mods.py
```

3. Run all cells sequentially

---

## 🧪 What's in the Notebook

### Section 1: Problem Definition (3-4 min)
- Introduction to lattice protein folding
- Amino acid types (H, P, C) and their interactions
- Physical constraints explained

### Section 2: Mathematical Formulation (5-6 min)
- Binary variable encoding
- Derivation of all four energy terms (E_MJ, E₁, E₂, E₃)
- Full LaTeX mathematics with explanations

### Section 3: QUBO Construction (8-10 min)
- Step-by-step matrix building for each energy term
- Visualization of QUBO matrices as heatmaps
- Polynomial representation of energy functions
- Combined QUBO matrix

### Section 4: Small Example - HPCH Chain (4-5 min)
- 4 residues on a 2×2 lattice (16 binary variables)
- Complete walkthrough of matrix construction
- Visualization of lattice configurations

### Section 5: Validation and Testing (5-6 min)
- Test Case 1: Valid configuration (all constraints satisfied)
- Test Case 2: Invalid configuration (multiple violations)
- Verification that QUBO evaluation matches direct calculation
- Color-coded lattice visualizations

### Section 6: Larger Example - HHCHPC Chain (3-4 min)
- 6 residues on a 6-position lattice (36 binary variables)
- Demonstrates scalability of the approach
- Complexity analysis

### Section 7: Analysis and Discussion (3-4 min)
- Complexity scaling (how problem size grows)
- Potential solvers (quantum annealing, QAOA, classical methods)
- Future extensions (3D, larger chains, sophisticated energy models)

---

## 🧩 Example Test Cases

The notebook includes three validated test cases:

| Test | Chain | Bitstring | Expected Energy (E_MJ, E₁, E₂, E₃) |
|------|-------|-----------|-------------------------------------|
| 1 | HPCH | `0010 1000 0100 0001` | (-1, 0, 0, 0) |
| 2 | HCHP | `1100 0101 0100 1010` | (-1, 3, 4, 2) |
| 3 | HHCHPC | `100000 000010 010000 001000 000100 000001` | (-2, 0, 0, 3) |

All tests pass with exact agreement between QUBO evaluation and direct calculation! ✓

---

## 🔧 Customization

### Interaction Matrix
Modify the interaction energies in `calc_mods.py`:
```python
C = {
    ('H', 'H'): 1,   # Hydrophobic attraction
    ('C', 'C'): -1,  # Charge repulsion
    ('H', 'C'): 0,   # Neutral
    # ... etc
}
```

### Lagrange Multipliers
Adjust constraint penalties:
```python
L1, L2, L3 = 1.0, 1.0, 1.0  # Default: equal weighting
```

### Lattice Structure
Define custom adjacency matrices for different lattice topologies in your notebook cells.

---

## 📊 Visualization Features

The notebook includes:
- **Heatmaps**: QUBO matrix visualizations
- **Lattice diagrams**: 2D grid with residue placements
- **Color coding**:
  - 🔵 Blue nodes: Valid placements
  - 🔴 Red nodes: E₂ violations (overlaps)
  - 🟢 Green edges: Connected chain
  - 🔴 Dashed red edges: E₃ violations (broken connectivity)
- **Energy breakdown**: Individual contributions from each term

---

## 🎯 Use Cases

This formulation can be used with:

1. **Quantum Annealers** (e.g., D-Wave): Direct QUBO input
2. **QAOA**: Convert QUBO to Ising Hamiltonian
3. **Simulated Annealing**: Classical probabilistic optimization
4. **Branch & Bound**: Exact classical solvers
5. **Genetic Algorithms**: Heuristic approaches

---

## 🧠 Key Insights

### Why QUBO?
- **Unconstrained**: Converts hard constraints into soft penalties
- **Quantum-friendly**: Natural encoding for quantum annealing and QAOA
- **Modular**: Easy to add/modify constraints independently
- **Verified**: Dual calculation methods ensure correctness

### Complexity
- **Variables**: N × M (linear in chain length and lattice size)
- **Matrix size**: (N×M)² ≈ O(N²M²)
- **Sparsity**: Typically 30-50% non-zero entries
- **Scalability**: Tractable for N ≤ 20-30 with modern solvers

---

## 📚 Further Reading

For background on:
- **Protein folding models**: Search for "HP model protein folding" or "Miyazawa-Jernigan potential"
- **QUBO formulations**: Andrew Lucas's 2014 paper "Ising formulations of many NP problems" is excellent
- **Quantum optimization**: The original QAOA paper by Farhi et al. (2014) on arXiv

---

## 🏡 Future Enhancements

- [ ] 3D lattice implementation
- [ ] Integration with D-Wave Ocean SDK
- [ ] QAOA implementation with Qiskit
- [ ] Comparative benchmarks (SA vs QA vs QAOA)
- [ ] Larger protein chains (N > 10)
- [ ] More sophisticated energy models (solvation, electrostatics)
- [ ] Interactive lattice visualization (plotly/dash)

---

## 🧾 Testing

To verify the QUBO construction:
```bash
cd problem_formation_and_evaluation/QUBO_construction
python construction_test.py
```

You should see:
```
✓ ALL TESTS PASSED!
```

---

## 📧 Contact

Questions, suggestions, or bug reports?  
Email: **jonah@planetminkoff.com**

---

**Thanks for checking it out!** 🚀  
If you find this useful, please star the repo and share with others interested in quantum optimization or computational biology!