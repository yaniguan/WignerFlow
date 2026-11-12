# WignerFlow Studio

`WignerFlow Studio` (`equitraj/studio.py` + `equitraj/compliance.py`) is a browser app that connects to a local Python + PyTorch server. It lets you try every feature in WignerFlow and test 10 basic physics rules in real time, with no extra web frameworks required.

---

## 1. Quick Start

```bash
# Open the interactive web app on http://127.0.0.1:8788
python scripts/wignerflow_studio.py --port 8788

# Or run the 10-rule physics check in the terminal and save a JSON report
python scripts/wignerflow_studio.py --audit-only --target aspirin_hetero_o3 --precision float64 --json-out audit.json
```

---

## 2. Five Tabs Inside the App

| Tab | What You Can Do | What You See |
| :--- | :--- | :--- |
| **1. Physics Rules Check (`G1-G10`)** | Test trained models (`argon_flow_v1`) or multi-element molecules (`aspirin_hetero_o3`) under any 3D rotation, or click **`F1-F10` Break-a-Rule switches** to see how bugs get caught. | 3D rotated-vs-unrotated arrows, 5-axis accuracy radar, and a 10-rule pass/fail list. |
| **2. Molecule Simulator (`3D & .xyz`)** | Run fast CPU simulations on built-in molecules (`aspirin`, `ala2`, `ethanol`, `li_ec_pf6`) or paste your own `.xyz` coordinates. | Live 3D moving molecule, temperature curve, bond-stretch check, and `.xyz` download. |
| **3. Drug Pocket Check** | Wiggle a candidate drug pose inside a protein pocket for a short CPU run to see if it stays locked in (`GO`) or clashes and pops out (`REJECT`). | 3D pocket + drug view, distance drift (`RMSD`), hydrogen bonds, and `GO / HOLD / REJECT` badge. |
| **4. Battery Electrolytes & CALiSol-23** | Simulate a small lithium electrolyte box and separate free ions from ion pairs. Shows the saved CALiSol-23 aligner metrics and the top 15 recipes from the 1,000,000-recipe screen (toy model output, not lab-tested). | 3D periodic box, `Li+-O` neighbor curve, CALiSol-23 metrics, and the top 15 table. |
| **5. All Benchmark Numbers** | Browse saved results: the MD17/MD22 large-time-step test, CALiSol-23 metrics, measured CPU speed, and one-line dataset import commands. | Summary tables and `.json` report download. |

---

## 3. The 10 Physics Rules (`G1-G10`) in Plain English

1. **`G1` — 3D Rotation (`SO(3)` & Pole Check)**: If you rotate all input atoms by an angle $R$, the predicted atom moves must rotate by the exact same $R$ — even when a bond points straight along the $+y$ or $-y$ pole (`F1`, `F2`).
2. **`G2` — Mirror Flip (`O(3)` Parity)**: If you reflect a molecule in a mirror, predicted moves must reflect cleanly too. Adding a right-hand cross product $\mathbf{r} \times \mathbf{v}$ (`F3`) passes normal 3D rotation (`G1`), but immediately fails the mirror check (`G2`).
3. **`G3` — Space Shift & Center-of-Mass Lock**: Moving the whole molecule in space must not change internal motion, and total momentum $\sum_i m_i \Delta \mathbf{v}_i$ must stay zero so the molecule never drifts on its own (`F4`).
4. **`G4` — Atom Re-ordering**: Swapping row #1 and row #5 in the atom table must give the exact same motion for each physical atom.
5. **`G5` — Repeating Box Wrap & Smooth Cutoff**: Atoms crossing a periodic box wall or the neighbor cutoff radius must wrap smoothly without sudden jumps (`F5`).
6. **`G6` — No Fake Loop Forces**: Moving an atom around a tiny closed loop must not create or destroy free energy (`F6`).
7. **`G7` — Proper Heat Balance (No "Flying Ice Cube")**: Kinetic energy must stay in internal bond vibrations instead of draining into rigid whole-molecule drift (`F7`).
8. **`G8` — No Atom Overlap & Intact Bonds**: Non-bonded atoms must not crash inside each other's size radius, and covalent bonds must stay near their natural lengths (`F8`).
9. **`G9` — 100% Survival & Target Temperature**: A multi-step trajectory must finish without blowing up and stay near the target temperature (`F9`).
10. **`G10` — Smooth Flow Path (`Euler` vs `Heun`)**: 1-step Euler and 2-step Heun solvers agree closely when the learned flow path is smooth and straight (`F10`).

---

## 4. Where the numbers come from

The benchmark tables are read at start-up from the committed result files:

- `results/pure_md_md17/md17_md22_benchmark.json` (MD17/MD22 large-time-step test; a pair-model baseline, not the WignerFlow network)
- `results/calisol/align_eval_seed42.json` and `results/calisol/box_scaling_cpu.json` (CALiSol-23 metrics and measured CPU speed)
- `results/solvation_top15/top15_solvation_report.json` (top 15 recipes; toy-model output)

If a file is missing, its table is left empty. No benchmark number is hard-coded in the app.
WignerFlow's public generation results are for argon only.

