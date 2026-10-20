"""Run a v0 checkpoint: generate a trajectory with the model alone (no integrator, no OpenMM, no data files).

    python scripts/demo.py --ckpt checkpoints/argon_flow_v0.pt --out demo_out

Starts from the bundled 3-frame context, rolls out --steps steps, and compares temperature, RDF and VACF
with the bundled ground-truth reference. Writes figures and demo_out/summary.json.
"""

import argparse
import json
import os

import numpy as np
import torch

from equitraj import analysis, plots
from equitraj.model import EquiTraj
from equitraj.rollout import rollout, temperature


def load_release(path, device):
    ck = torch.load(path, map_location=device, weights_only=True)
    assert ck.get("format") == "wignerflow-v0", "not a WignerFlow v0 release checkpoint"
    model = EquiTraj(n_types=len(ck["elements"]), k=ck["k"], stride_fs=ck["stride_fs"], **ck["model_cfg"])
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck


def first_zero(t, c):
    i = np.argmax(c < 0)
    if c[i] >= 0:
        return None
    return float(t[i - 1] + (t[i] - t[i - 1]) * c[i - 1] / (c[i - 1] - c[i]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/argon_flow_v0.pt")
    ap.add_argument("--steps", type=int, default=None, help="default: 1000 on GPU, 200 on CPU (~1 s/step)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sampling_steps", type=int, default=10, help="flow-matching ODE steps per frame")
    ap.add_argument("--out", default="demo_out")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    if args.steps is None:
        args.steps = 1000 if device.type == "cuda" else 200
    os.makedirs(args.out, exist_ok=True)
    model, ck = load_release(args.ckpt, device)
    ref, dt = ck["reference"], ck["stride_fs"]
    demo = ck["demo"]
    box = None if demo["box"] is None else demo["box"][None]
    n_atoms = len(ck["types"])

    torch.manual_seed(args.seed)
    out = rollout(model, demo["pos"][None], demo["vel"][None], ck["types"], box, ck["masses"], args.steps,
                  sampling_steps=args.sampling_steps, t_max=3 * ref["T_K"], n_constraints=ck.get("n_constraints", 0))
    crashed = int(out["crash_step"][0])
    x, v = out["positions"][:, 0].numpy(), out["velocities"][:, 0].numpy()
    n_ok = args.steps if crashed < 0 else crashed
    half = n_ok // 2
    temp = temperature(torch.from_numpy(v[:n_ok]), ck["masses"], ck.get("n_constraints", 0)).numpy()
    summary = {"checkpoint": os.path.basename(args.ckpt), "train_step": ck["train_step"], "device": str(device),
               "steps": args.steps, "stride_fs": dt, "simulated_ps": args.steps * dt / 1000,
               "crashed_at_step": None if crashed < 0 else crashed, "wall_s": out["wall_s"],
               "T_ref_K": ref["T_K"], "T_model_second_half_K": float(temp[half:].mean()) if n_ok > 1 else None}

    idx = np.arange(n_atoms)
    r_ref, g_ref = np.array(ref["rdf_r"]), np.array(ref["rdf_g"])
    if n_ok >= 20 and box is not None:
        frames = x[half:n_ok:max(1, (n_ok - half) // 200)]
        r, g, _ = analysis.rdf(frames, np.tile(box[0].numpy(), (len(frames), 1, 1)), idx, idx, float(r_ref[-1]) + 1e-6, len(r_ref))
        pm, hm = analysis.first_peak(r, g)
        pg, hg = analysis.first_peak(r_ref, g_ref)
        summary["rdf_peak_pos_err_A"] = float(abs(pm - pg))
        summary["rdf_peak_height_rel_err"] = float(abs(hm - hg) / hg)
        plots.line_plot(os.path.join(args.out, "rdf.png"), [("ground truth", r_ref, g_ref), ("model", r, g)],
                        "r (Å)", "g(r)", "RDF (second half of the rollout)")
    lag = np.array(ref["vacf_lag_fs"])
    max_lag = min(len(lag), n_ok // 2)
    if max_lag > 5:
        c = analysis.vacf(v[half:n_ok] if n_ok - half > max_lag else v[:n_ok], max_lag)
        c = c / c[0]
        z_model, z_ref = first_zero(lag[:max_lag], c), first_zero(lag, np.array(ref["vacf"]))
        summary["vacf_first_zero_fs"] = {"model": z_model, "reference": z_ref}
        plots.line_plot(os.path.join(args.out, "vacf.png"),
                        [("ground truth", lag / 1000, ref["vacf"]), ("model", lag[:max_lag] / 1000, c)],
                        "t (ps)", "C(t) / C(0)", "Velocity autocorrelation", hline=0.0)
    t_axis = np.arange(1, n_ok + 1) * dt / 1000
    plots.line_plot(os.path.join(args.out, "temperature.png"),
                    [("ground truth", t_axis, np.full(n_ok, ref["T_K"])), ("model", t_axis, temp)],
                    "rollout time (ps)", "T (K)", "Temperature (model only)")
    np.save(os.path.join(args.out, "positions.npy"), x)
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
