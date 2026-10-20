"""Where does a one-step prediction gain or lose kinetic energy? Decompose it from many samples of the same context.

    python scripts/kinetic_diagnostics.py --run results/argon/B_S3a_seed1 --ckpt snapshots/best_step10000.pt --contexts 64 --samples 32

For a context c (the last k true frames) and K samples v'_s of the model's next velocities, with v the last true velocity:
    E[K(v')] - K(v) = sum_i m_i v_i.(E[v'_i] - v_i)  +  1/2 sum_i m_i |E[v'_i] - v_i|^2  +  1/2 sum_i m_i tr Cov(v'_i | c)
                       cross term                         squared term                        variance term
The variance term is zero for a deterministic head and is what a regression head cannot supply. The same split of the true
step (c -> true next frame) is reported; in an NVE ground truth the mean energy change over a stride is about zero.
Energies are kJ/mol per atom; "T per step" is the temperature change per model step that this energy change would cause.
Writes <run>/kinetic_diagnostics.json (or --out).
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import load_model  # noqa: E402

from equitraj.units import AMU_A2_PER_FS2_TO_KJ_PER_MOL, KB  # noqa: E402


def kinetic(v, m):
    """v: (..., N, 3) -> (...,) kJ/mol."""
    return 0.5 * torch.einsum("n,...nd->...", m.double(), v.double() ** 2) * AMU_A2_PER_FS2_TO_KJ_PER_MOL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--contexts", type=int, default=64)
    ap.add_argument("--samples", type=int, default=32, help="flow samples per context (a regression head gives one)")
    ap.add_argument("--sampling_steps", type=int, default=10)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device)
    model, cfg, ds, step = load_model(args.run, args.ckpt, device, args.split)
    k, n = cfg["k"], len(ds.types)
    deterministic = model.head_type == "regression"
    n_samples = 1 if deterministic else args.samples
    masses = ds.masses.to(device)
    dof = 3 * n - 3 - ds.n_constraints
    idx = np.linspace(0, len(ds) - 1, args.contexts).astype(int)
    torch.manual_seed(0)
    rec = {key: [] for key in ["cross", "square", "variance", "dK_model", "true_cross", "true_square", "dK_true", "cos_dv"]}
    for i in idx:
        pos, vel, box = ds[i]  # k context frames and one true next frame
        p, v = pos[None, :k].to(device), vel[None, :k].to(device)
        b = box[None].to(device) if ds.periodic else None
        types = ds.types.to(device)
        v_last, v_true = v[0, -1], vel[k].to(device)
        samples = torch.stack([model.predict(p, v, types, b, masses, n_steps=args.sampling_steps)[1][0] for _ in range(n_samples)])  # (S, N, 3)
        v_mean = samples.mean(0)
        dv, dv_true = v_mean - v_last, v_true - v_last
        cross = (masses * (v_last * dv).sum(-1)).sum() * AMU_A2_PER_FS2_TO_KJ_PER_MOL
        square = 0.5 * (masses * (dv**2).sum(-1)).sum() * AMU_A2_PER_FS2_TO_KJ_PER_MOL
        var = kinetic(samples, masses).mean() - kinetic(v_mean, masses)  # E[K(v')] - K(E[v']) = 1/2 sum m tr Cov
        tc = (masses * (v_last * dv_true).sum(-1)).sum() * AMU_A2_PER_FS2_TO_KJ_PER_MOL
        ts = 0.5 * (masses * (dv_true**2).sum(-1)).sum() * AMU_A2_PER_FS2_TO_KJ_PER_MOL
        for key, val in [("cross", cross), ("square", square), ("variance", var), ("dK_model", cross + square + var),
                         ("true_cross", tc), ("true_square", ts), ("dK_true", tc + ts),
                         ("cos_dv", torch.nn.functional.cosine_similarity(dv.flatten(), dv_true.flatten(), dim=0))]:
            rec[key].append(float(val))
    mean = {key: float(np.mean(val)) for key, val in rec.items()}
    sem = {key: float(np.std(val, ddof=1) / np.sqrt(len(val))) for key, val in rec.items()}
    per_atom = {key: mean[key] / n for key in mean if key != "cos_dv"}
    t_per_step = lambda e: 2 * e / (dof * KB)  # noqa: E731
    out = {"run": args.run, "checkpoint": args.ckpt, "train_step": int(step), "head": model.head_type, "target": model.target,
           "contexts": len(idx), "samples_per_context": n_samples, "stride_fs": ds.stride_fs,
           "kJ_mol_per_atom_mean": per_atom, "kJ_mol_total_sem": {key: sem[key] for key in sem if key != "cos_dv"},
           "cos_similarity_of_mean_dv_with_true_dv": mean["cos_dv"],
           "T_change_per_step_K": {"model": t_per_step(mean["dK_model"]), "true": t_per_step(mean["dK_true"]),
                                   "model_minus_true": t_per_step(mean["dK_model"] - mean["dK_true"])}}
    path = args.out or os.path.join(args.run, "kinetic_diagnostics.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
