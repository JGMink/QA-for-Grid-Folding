# 🧬 Protein Folding via QUBO Formulation

An **educational implementation** and explanation of the QUBO-based protein folding approach from:

> **"Protein Folding on Quantum Computers: A Hybrid Quantum-Classical Approach"**  
> *Aaranya et al. (2024)*  
> arXiv:2510.01890  
> [View Paper](https://arxiv.org/pdf/2510.01890)

This project provides a **pedagogical walkthrough** of their methodology using **toy examples** to make the concepts accessible and verifiable.

---

## 🎯 Purpose of This Project

This repository is designed to:

1. **Explain** the QUBO formulation from Aaranya et al. (2024) with detailed derivations
2. **Demonstrate** the approach using small, tractable examples (4-6 residues)
3. **Validate** that the implementation correctly encodes the problem
4. **Provide** tested, reusable code for educational purposes

**This is NOT a production protein folding solver** - it's an educational tool to understand how to formulate biological optimization problems for quantum computing.

---

## 📄 Original Paper Summary

The paper by Aaranya et al. presents a hybrid quantum-classical approach that:
- Formulates lattice protein folding as a QUBO problem
- Uses Miyazawa-Jernigan interaction potentials
- Encodes physical constraints (connectivity, self-avoidance) as penalty terms
- Can be solved using quantum annealers, QAOA, or classical methods

**Our contribution**: We provide a clean, well-documented implementation with toy examples that make the mathematics accessible to students and researchers new to quantum optimization.

---

## 🌱 What This Implementation Does

This project systematically demonstrates the QUBO formulation through:

1. **Problem Definition**: Simplified 2D lattice model with 3 amino acid types (H, P, C)
2. **Mathematical Formulation**: Complete derivation of all energy terms from the paper
3. **QUBO Construction**: Step-by-step matrix building with visualization
4. **Validation**: Dual calculation methods verify correctness
5. **Toy Examples**: Small chains (4-6 residues) that can be verified by hand

The implementation shows how **biological constraints** (chain connectivity, self-avoidance) and **optimization objectives** (Miyazawa-Jernigan interaction energy) are unified into a single QUBO matrix.

---

## 📊 Project Structure
```
protein-folding-qubo/
├── problem_formation_and_evaluation/
│   ├── energy_calc/
│   │   └── calc_mods.py              # Direct energy calculation
│   ├── QUBO_construction/
│   │   ├── qubo_generation.py        # QUBO matrix building
│   │   └── construction_test.py      # Validation tests
│   └── claude_eval/
│       └── verify_qubo.py            # Comprehensive verification
├── notebooks/
│   └── protein_folding_qubo.ipynb    # Main tutorial notebook
├── papers/
│   └── protein_folding.pdf           # Original paper (Aaranya et al.)
└── README.md
```

---

## 🔍 Energy Formulation (Following Aaranya et al.)

The total energy combines four terms:

$$E_{\text{total}} = E_{\text{MJ}} + \lambda_1 E_1 + \lambda_2 E_2 + \lambda_3 E_3$$

Where:
- **E_MJ**: Miyazawa-Jernigan interaction energy (biological objective)
- **E₁**: One position per residue constraint
- **E₂**: Self-avoidance constraint (no overlaps)
- **E₃**: Chain connectivity constraint

### Binary Encoding

Each configuration uses binary variables:
- $b_{i,n} = 1$ if residue $i$ occupies position $n$
- $b_{i,n} = 0$ otherwise

For N residues and M positions: **N × M binary variables**

### QUBO Matrix Representation

All terms combine into a single matrix **Q**:

$$E(x) = x^T Q x + \text{offset}$$

This formulation is compatible with quantum annealers (D-Wave), QAOA, and classical solvers.

---

## ⚙️ Getting Started

### Requirements
```bash
pip install numpy matplotlib seaborn jupyter
```

### Running the Tutorial Notebook

#### Option 1: Local Jupyter
```bash
git clone <your-repo-url>
cd protein-folding-qubo
jupyter notebook notebooks/protein_folding_qubo.ipynb
```

#### Option 2: Google Colab
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19EnI3qWr388mPwWXmZY4o0jzo4_UETY3?usp=sharing)

---

## 🧪 What's in the Tutorial Notebook

The notebook provides a **complete walkthrough** of the QUBO formulation:

### Section 1: Problem Definition (3-4 min)
- Introduction to the paper's approach
- Lattice protein folding basics
- Amino acid types and interactions

### Section 2: Mathematical Formulation (5-6 min)
- Binary variable encoding
- **Detailed derivation** of all four energy terms
- Step-by-step expansion of constraint penalties

### Section 3: QUBO Construction (8-10 min)
- Matrix building for each energy component
- Visualization of QUBO structure
- How constraints become penalty terms

### Section 4: Toy Example - HPCH Chain (4-5 min)
- **4 residues** on a 2×2 lattice (16 variables)
- Small enough to verify calculations by hand
- Complete matrix walkthrough

### Section 5: Validation (5-6 min)
- Valid vs. invalid configurations
- Verification: QUBO method = Direct calculation
- Constraint violation detection

### Section 6: Slightly Larger Example - HHCHPC (3-4 min)
- **6 residues** demonstrating scalability
- Still tractable for educational purposes
- Complexity analysis

### Section 7: Discussion (3-4 min)
- Scaling behavior
- Connection to quantum/classical solvers
- Extensions and future work

**Total time: ~30-35 minutes**

---

## 🧩 Validated Test Cases

All examples verified with dual calculation methods:

| Test | Chain | Variables | Status | Valid Config? |
|------|-------|-----------|--------|---------------|
| 1 | HPCH | 16 | ✓ Pass | Yes (E=−1) |
| 2 | HCHP | 16 | ✓ Pass | No (violations) |
| 3 | HHCHPC | 36 | ✓ Pass | No (E₃ violation) |

**All tests show exact agreement** between QUBO evaluation and direct energy calculation.

---

## 🔧 Running Verification Tests

To verify the implementation:

```bash
cd problem_formation_and_evaluation/QUBO_construction
python construction_test.py
```

For comprehensive verification:
```bash
cd problem_formation_and_evaluation/claude_eval
python verify_qubo.py
```

Expected output:
```
✓ ALL TESTS PASSED - IMPLEMENTATION IS CORRECT
```

---

## 📊 Visualization Features

The notebook includes:
- **QUBO matrix heatmaps**: See the structure of Q
- **Lattice diagrams**: 2D grid with residue placements
- **Energy breakdowns**: Contribution from each term
- **Color-coded violations**: Visual constraint checking

---

## 🎯 Potential Use Cases (Following the Paper)

This formulation can be used with:

1. **Quantum Annealers** (D-Wave): Direct QUBO implementation
2. **QAOA**: Convert to Ising Hamiltonian for gate-based quantum computers
3. **Simulated Annealing**: Classical probabilistic optimization
4. **Hybrid Solvers**: Quantum-classical approaches (as in the original paper)

---

## 🧠 Educational Goals

### What You'll Learn:
- How to formulate constrained optimization as QUBO
- Converting biological constraints to penalty terms
- Practical QUBO matrix construction
- Verification techniques for quantum-compatible formulations

### Why Small Examples?
- **Verifiable**: Can check calculations by hand
- **Transparent**: Every matrix entry has clear meaning
- **Debuggable**: Easy to trace errors
- **Educational**: Focus on concepts, not scale

**For production protein folding**, see the original paper's full implementation and benchmarks.

---

## 📚 Related Resources

### Original Paper
- Aaranya et al. (2024). "Protein Folding on Quantum Computers: A Hybrid Quantum-Classical Approach"
- arXiv:2510.01890
- [Paper PDF](papers/protein_folding.pdf)

### Background Reading
- **QUBO formulations**: Andrew Lucas (2014) - "Ising formulations of many NP problems"
- **HP model**: Ken Dill's work on simplified protein models
- **Miyazawa-Jernigan**: Original 1996 paper on residue interaction potentials
- **QAOA**: Farhi et al. (2014) - Original quantum approximate optimization paper

---

## 🔄 Differences from Original Paper

| Aspect | Original Paper | This Implementation |
|--------|----------------|---------------------|
| Scale | Production-scale chains | Toy examples (4-6 residues) |
| Purpose | Research results | Educational tool |
| Focus | Performance benchmarks | Mathematical clarity |
| Code | Research codebase | Pedagogical, documented |
| Validation | Quantum hardware results | Dual calculation verification |

**This is complementary to the paper**: We provide the educational foundation to understand their methodology.

---

## 🏡 Future Extensions

Potential enhancements (maintaining educational focus):

- [ ] Integration with D-Wave Ocean SDK (live quantum annealing)
- [ ] QAOA implementation with Qiskit
- [ ] Step-by-step solver comparison (SA vs exact vs quantum)
- [ ] Interactive visualization (plotly/dash)
- [ ] Additional small examples with detailed walkthroughs
- [ ] Jupyter widgets for parameter exploration

---

## 📧 Contact & Attribution

**Implementation by:** Jonah Minkoff  
**Email:** jonah@planetminkoff.com

**Based on the work of:**  
Aaranya et al. (2024) - "Protein Folding on Quantum Computers"  
arXiv:2510.01890

---

## 📜 Citation

If you use this educational implementation, please cite both:

**This implementation:**
```bibtex
@misc{minkoff2025qubo,
  author = {Minkoff, Jonah},
  title = {Protein Folding via QUBO Formulation: An Educational Implementation},
  year = {2025},
  url = {<your-repo-url>}
}
```

**Original paper:**
```bibtex
@article{aaranya2024protein,
  title={Protein Folding on Quantum Computers: A Hybrid Quantum-Classical Approach},
  author={Aaranya, et al.},
  journal={arXiv preprint arXiv:2510.01890},
  year={2024}
}
```

---

## ⚖️ License

MIT License.

Please respect the original paper's work when using or extending this implementation.

---

**Thanks for exploring quantum optimization for biology!** 🚀  

If this helps your understanding, please:
- ⭐ Star the repo
- 📖 Read the original paper
- 🔗 Share with others learning quantum computing or computational biology