"""Stability timeline for long rollouts (e.g. 20 ns targets).

Splits a trajectory into consecutive time blocks and reports, per block:
    mean temperature (K), mean total energy (kJ/mol, if potential energy given),
    RDF L1 error vs a reference g(r) (reference trajectory, or the first block),
    smallest pair distance (Å).
Then finds the first block that violates any threshold and reports the
"stable time" (ns) up to that block. Plus whole-run temperature drift (K/ns)
and total-energy drift (kJ/mol/ns/dof).

Units: positions Å, velocities Å/fs, masses amu, time fs, boxes (T,3,3) Å orthorhombic.
"""
import numpy as np

from equitraj import analysis
from equitraj.units import AMU_A2_PER_FS2_TO_KJ_PER_MOL


def kinetic_energy(velocities, masses):
    """(T,) kJ/mol from (T,N,3) Å/fs and (N,) amu."""
    return 0.5 * np.einsum("n,tnd->t", masses, velocities ** 2) * AMU_A2_PER_FS2_TO_KJ_PER_MOL


def block_slices(n_frames, n_blocks):
    if n_blocks < 1 or n_frames < n_blocks:
        raise ValueError("need 1 <= n_blocks <= n_frames")
    edges = np.linspace(0, n_frames, n_blocks + 1).astype(int)
    return [slice(a, b) for a, b in zip(edges[:-1], edges[1:])]


def _min_dist(pos, box):
    best = np.inf
    for t in range(len(pos)):
        L = None if box is None else analysis.box_lengths(box[t])
        d = np.linalg.norm(analysis.minimum_image(pos[t][:, None] - pos[t][None], L), axis=-1)
        d[np.diag_indices_from(d)] = np.inf
        best = min(best, float(d.min()))
    return best


def stability_timeline(positions, velocities, masses, time_fs, box, *, n_blocks=10,
                       potential_energy=None, idx_a=None, idx_b=None, r_max=None, n_bins=100,
                       reference_positions=None, reference_box=None, n_constraints=0,
                       max_temp_dev=50.0, max_rdf_l1=0.5, min_distance=0.5, target_temperature=None):
    """Return a JSON-serializable dict with per-block stats and the stable time.

    target_temperature: thresholds on temperature are |T_block - target|; defaults to block-0 mean.
    idx_a/idx_b default to all atoms. r_max defaults to 0.49 * min box length.
    reference_*: reference trajectory for g(r); defaults to block 0 of this trajectory.
    """
    positions = np.asarray(positions, float)
    velocities = np.asarray(velocities, float)
    masses = np.asarray(masses, float)
    time_fs = np.asarray(time_fs, float)
    box = None if box is None else np.asarray(box, float)
    n_frames, n_atoms = positions.shape[:2]
    idx = np.arange(n_atoms)
    idx_a = idx if idx_a is None else np.asarray(idx_a)
    idx_b = idx if idx_b is None else np.asarray(idx_b)
    if r_max is None:
        if box is None:
            raise ValueError("r_max is required without a periodic box")
        r_max = 0.49 * float(min(analysis.box_lengths(box[0])))
    dof = analysis.n_dof(n_atoms, n_constraints)
    ke = kinetic_energy(velocities, masses)
    temp = analysis.temperature(ke, dof)
    etot = None if potential_energy is None else ke + np.asarray(potential_energy, float)
    blocks = block_slices(n_frames, n_blocks)
    if reference_positions is None:
        ref_pos, ref_box = positions[blocks[0]], None if box is None else box[blocks[0]]
    else:
        ref_pos = np.asarray(reference_positions, float)
        ref_box = None if reference_box is None else np.asarray(reference_box, float)
    r, g_ref, _ = analysis.rdf(ref_pos, ref_box, idx_a, idx_b, r_max, n_bins)
    t_target = float(temp[blocks[0]].mean()) if target_temperature is None else float(target_temperature)

    rows, first_bad = [], None
    for i, s in enumerate(blocks):
        pos_b = positions[s]
        box_b = None if box is None else box[s]
        _, g, _ = analysis.rdf(pos_b, box_b, idx_a, idx_b, r_max, n_bins)
        row = dict(block=i, t_start_ns=float(time_fs[s][0] * 1e-6), t_end_ns=float(time_fs[s][-1] * 1e-6),
                   temperature_K=float(temp[s].mean()), rdf_l1=float(analysis.rdf_l1(r, g, g_ref)),
                   min_distance_A=_min_dist(pos_b, box_b),
                   finite=bool(np.isfinite(pos_b).all() and np.isfinite(velocities[s]).all()))
        if etot is not None:
            row["total_energy_kJmol"] = float(etot[s].mean())
        reasons = []
        if not row["finite"]:
            reasons.append("nan")
        if abs(row["temperature_K"] - t_target) > max_temp_dev:
            reasons.append("temperature")
        if row["rdf_l1"] > max_rdf_l1:
            reasons.append("rdf")
        if row["min_distance_A"] < min_distance:
            reasons.append("overlap")
        row["violations"] = reasons
        if reasons and first_bad is None:
            first_bad = i
        rows.append(row)

    t_ns = time_fs * 1e-6
    ok = np.isfinite(temp)
    out = dict(n_frames=int(n_frames), n_blocks=len(blocks), target_temperature_K=t_target,
               thresholds=dict(max_temp_dev=max_temp_dev, max_rdf_l1=max_rdf_l1, min_distance=min_distance),
               temperature_drift_K_per_ns=float(np.polyfit(t_ns[ok], temp[ok], 1)[0]) if ok.sum() > 1 else None,
               blocks=rows, first_unstable_block=first_bad,
               stable_time_ns=float(t_ns[-1] - t_ns[0]) if first_bad is None
               else float(rows[first_bad]["t_start_ns"] - t_ns[0]))
    if etot is not None and np.isfinite(etot).sum() > 1:
        m = np.isfinite(etot)
        out["energy_drift_kJmol_per_ns_per_dof"] = float(analysis.energy_drift(time_fs[m] * 1e-3, etot[m], dof))
    return out


def timeline_markdown(report):
    lines = [f"Stable time: {report['stable_time_ns']:.4g} ns "
             f"(first unstable block: {report['first_unstable_block']})",
             f"Temperature drift: {report['temperature_drift_K_per_ns']} K/ns", "",
             "| block | t_start (ns) | T (K) | RDF L1 | min d (Å) | violations |", "|---|---|---|---|---|---|"]
    for b in report["blocks"]:
        lines.append(f"| {b['block']} | {b['t_start_ns']:.4g} | {b['temperature_K']:.1f} | {b['rdf_l1']:.3f} | "
                     f"{b['min_distance_A']:.2f} | {','.join(b['violations']) or '-'} |")
    return "\n".join(lines)
