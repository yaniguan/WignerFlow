# How WignerFlow Works (Plain-English Technical Guide)

This note explains how `WignerFlow` (`equitraj/`) works, why normal molecular dynamics (MD) is slow, and what physics rules keep neural trajectories stable.

---

## 1. Why Normal MD Needs Tiny Steps

In classical MD, we move atoms using Newton's law ($F = ma$).
- Light hydrogen atoms vibrate back and forth roughly once every **10 femtoseconds (`10 fs`)**.
- Because of that fast vibration, normal MD must take tiny **`0.5–1.0 fs` steps**.
- If you try jumping `10 fs` at once with standard Velocity-Verlet, atoms overshoot their bonds and fly apart immediately.

## 2. How WignerFlow Takes Bigger Steps

Instead of computing tiny instantaneous forces every `0.5 fs`, WignerFlow learns **where atoms end up after a larger jump** ($\Delta t = 10\text{–}50\text{ fs}$), given their current positions and velocities:

1. **Coast First, Predict the Bend**: Most of an atom's motion over $\Delta t$ is straight-line coasting ($\mathbf{v} \cdot \Delta t$). WignerFlow only predicts the leftover curve away from straight-line coasting. That keeps the flow path close to straight, so a few neural steps (for example 2 Heun steps) are usually enough.
2. **Rotate Every Pair to the Y-Axis (`equitraj/so3.py`)**: 3D equivariant networks are usually slow because 3D tensor products take heavy math. WignerFlow rotates each neighbor pair onto the $+y$ axis (`EdgeRotation`), applies fast block-diagonal matrix multiplies (`SO2Linear`), and rotates back.
3. **Keep Center of Mass Still (`remove_com`)**: Total momentum $\sum_i m_i \Delta \mathbf{v}_i$ is forced to zero at every step so the molecule never drifts on its own.
4. **Short-Range Overlap Guard (`RepulsionPrior`)**: If two non-bonded atoms get dangerously close during a rollout, a simple analytic push keeps them from crashing into each other.

## 3. Where WignerFlow Wins vs. Where OpenMM-GPU Wins

- **Use OpenMM-GPU** when you have a cheap classical force field (AMBER/OPLS) and want one long `100 ns` trajectory. Compiled C++/CUDA kernels have almost zero launch overhead.
- **Use WignerFlow** when:
  1. Each force step is expensive (quantum `CCSD(T)`/`DFT` or polarizable force fields), so taking fewer, larger steps saves time.
  2. You pack `64–512` candidate molecules or battery boxes into one GPU batch.
  3. You want a quick check on a laptop CPU before spending GPU time on a long run.
