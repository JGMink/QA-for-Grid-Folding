# 🧬 Lattice Protein Folding — Simulated & Hybrid Annealing Demo

A small-scale demo showing how **Simulated Annealing (SA)** and **Hybrid Annealing (HA)** can solve simplified **protein folding problems** on a 3D lattice — inspired by the paper:

> **Folding lattice proteins confined on minimal surfaces**  
> *Anders Irbäck, Lucas Knuthson, Sandipan Mohanty (2025)*  
> [📄 arXiv:2510.01890](https://arxiv.org/abs/2510.01890)

---

## 🌱 What This Project Does

Proteins naturally fold into 3D shapes that minimize their internal energy — a process that’s incredibly complex to simulate.  
This project recreates that process *in miniature*, using a **tiny 3D lattice** and a simplified **3-letter amino acid alphabet**:

| Letter | Type | Effect on Energy |
|:------:|:------|:----------------|
| **H** | Hydrophobic | Favors contact with other H’s |
| **P** | Polar | Neutral interactions |
| **C** | Charged | Penalized for adjacent charged pairs |

Each “protein” chain has 6 residues placed in a **2×3×1 grid**, forming a linear chain that folds according to energy rules.  
We then use **Simulated Annealing** (and optionally **Hybrid Annealing**) to find the lowest-energy configuration.

---

## 🔍 Why It’s Cool

- 🧊 Demonstrates **optimization in high-dimensional search spaces**  
- 🧠 Bridges classical and quantum-inspired approaches (SA ↔ HA)  
- 🔬 Makes real protein folding ideas understandable on a toy scale  
- ⏱️ Tracks energy changes, temperature schedules, and time per run  

---

## ⚙️ How It Works

Each run:
1. Randomly generates a sequence (e.g. `HPCPHC`)
2. Places residues in a 2×3×1 lattice
3. Computes total energy  
   \[
   E = E_{MJ} + λ_1E_1 + λ_2E_2 + λ_3E_3
   \]
4. Iteratively improves it using **Simulated Annealing**:
   - Randomly swaps or moves residues  
   - Accepts worse states probabilistically (temperature-dependent)
5. Optionally validates against **brute-force enumeration (256 configs)**

---


Graph output (via Matplotlib) shows energy vs temperature and the best configuration found.

---

## 💡 Methods and Background

This implementation borrows the general energy formulation from **Irbäck et al. (2025)**, which introduces lattice folding on minimal surfaces, and adapts it to a simpler 3-letter system for clarity.

For background on Simulated Annealing, see Georgia Tech’s excellent summary:
> [Simulated Annealing: Methods and Real-World Applications](https://sites.gatech.edu/omscs7641/2024/02/19/simulated-annealing-methods-and-real-world-applications/)

---

## 🧪 Getting Started

### Requirements
```bash
  pip install numpy matplotlib
```
### Run
```bash
  python main.py
```
### 🔧 Configuration

You can tweak parameters in config.py or at the top of main.py:

TEMPERATURE_STEPS: number of temperature levels (default: 25)

SWEEPS_PER_TEMP: number of iterations per temperature

ALPHA: cooling rate for geometric schedule

SEQUENCE: choose your protein sequence (e.g., HPCPHC, HHPPCC, etc.)

These allow quick experimentation with annealing depth, chain type, and cooling behavior.

### 📊 Visualization

After each run, the program automatically generates:

- Energy vs. Temperature plots

- Lattice configuration snapshots (final folded state)

- You can enable live plotting during execution by setting:
```{python}
  LIVE_PLOT = True
```

### 🧩 Example Experiments

Try adjusting the parameters to see how annealing affects convergence:

| Experiment | Description | Expected Behavior |
|-------------|--------------|-------------------|
| `ALPHA = 0.95` | Slow cooling | More thorough exploration, slower runtime |
| `ALPHA = 0.70` | Fast cooling | Quicker but more likely to get stuck |
| `SWEEPS_PER_TEMP = 1000` | Longer per step | Better convergence for small systems |
| `SEQUENCE = 'HCHCHC'` | Alternating charges | Higher penalties, more complex landscape |

---

## 🧠 What’s Next

- Add true Hybrid Annealing integration (mixing classical + quantum subroutines)

- Visualize 3D lattice states with matplotlib or PyVista

- Extend to larger lattice (3×3×3) and more complex alphabets (HP or HP+C variants)

---

## 📄 References

Irbäck, Knuthson, Mohanty (2025).
Folding lattice proteins confined on minimal surfaces.
arXiv:2510.01890

Georgia Tech OMSCS 7641.
Simulated Annealing: Methods and Real-World Applications.
Course site

---

## 🧾 Paper Link

The accompanying Overleaf paper (in /paper/) explains:

- The full energy formulation

- The academic algorithm explained

- Background on HA vs SA

---

## Thanks for checking it out!
Feel free to email me with any concerns/noticeable bugs: <jonah@planetminkoff.com>


