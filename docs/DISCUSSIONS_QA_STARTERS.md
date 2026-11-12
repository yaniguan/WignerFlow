# Starter questions for GitHub Discussions (Q&A)

Short answers to common questions, to seed the
[WignerFlow Q&A board](https://github.com/yaniguan/WignerFlow/discussions/categories/q-a).
Public generation results in this repo are for argon only.

1. **How can WignerFlow take `50 fs` steps when bond vibrations need `1–2 fs` steps in normal MD?**
   Normal integrators such as Velocity Verlet follow the instantaneous force and blow up when the
   step is too large for the fastest vibration. WignerFlow does not integrate forces. It learns
   where atoms end up after one large step, given the last few frames of positions and velocities,
   so the fast vibrations are averaged out by the learned model. This only works inside the range of
   states seen in training.

2. **Why use SO(2) edge-aligned message passing instead of full SO(3) tensor products?**
   If each edge is rotated so it points along one axis, the equivariant layers become
   block-diagonal and much cheaper (Passaro & Zitnick, ICML 2023 show the cost drops from about
   `O(L^6)` to `O(L^3)` in the maximum degree `L`). The rotation is undone afterwards, so the output
   is still equivariant. The tests check this in `float64` to about `1e-10`.

3. **How can we tell that a rollout is going bad before atoms collide?**
   Two cheap signals, no reference MD needed: (a) compare a 1-step Euler and a 2-step Heun solve of
   the same flow step; a big gap means the learned flow is curving sharply, often a sign of an
   unfamiliar state; (b) draw several noise samples from the same history and measure their spread.

4. **Why can single-step training under-predict diffusion, and what does unrolled training change?**
   Training on one step at a time rewards the average next state, which can slightly shrink
   velocities. Over many steps this adds up and slows diffusion. Unrolled training runs several model
   steps with gradients (`EquiTraj.predict_with_grad`) and can add loss terms for temperature, the velocity autocorrelation, and
   mean-squared displacement, which is meant to reduce this damping.
