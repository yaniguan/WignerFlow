"""Stability monitoring for autoregressive rollouts.

StabilityMonitor is called by rollout() at every check step. It applies the crash criteria
(same thresholds and order as before) and additionally records, per system:
    temperature, kinetic energy, smallest pair distance, largest |v|, total momentum, and optionally
    the potential energy from a user callable energy_fn(pos (B, N, 3), box) -> (B,) kJ/mol
When a system crashes it stores why (reason in nan / overlap / temperature / bond, in that priority),
which atoms (indices and types), and the last `history` checks of the statistics.
diagnostics() summarizes everything as JSON-serializable lists, including the temperature drift (K/ns)
and kinetic-energy drift (kJ/mol/ns), fit by least squares over the checks before the crash.

Units: Å, Å/fs, amu, K, kJ/mol; time for the drifts in ns (step * stride_fs).
"""

from collections import deque

import numpy as np
import torch

from equitraj.neighbors import minimum_image
from equitraj.rollout import max_bond_deviation, min_pair_distance, temperature
from equitraj.units import AMU_A2_PER_FS2_TO_KJ_PER_MOL

REASONS = ("nan", "overlap", "temperature", "bond")


def kinetic_energy(vel, masses):
    """Kinetic energy (kJ/mol) per system. vel: (B, N, 3) Å/fs."""
    return 0.5 * torch.einsum("n,bnd->b", masses.double(), vel.double() ** 2) * AMU_A2_PER_FS2_TO_KJ_PER_MOL


def _slope(t, y):
    ok = np.isfinite(y)
    if ok.sum() < 2:
        return float("nan")
    return float(np.polyfit(t[ok], y[ok], 1)[0])


class StabilityMonitor:
    def __init__(self, types, masses, box=None, min_dist=0.5, t_max=None, bonds=None, bond_r0=None,
                 max_bond_dev=0.5, energy_fn=None, history=20, stride_fs=1.0, n_report=20):
        self.types, self.masses, self.box = types, masses, box
        self.min_dist, self.t_max = min_dist, t_max
        self.bonds, self.bond_r0, self.max_bond_dev = bonds, bond_r0, max_bond_dev
        self.energy_fn = energy_fn
        self.stride_fs = float(stride_fs)
        self.n_report = n_report
        self.crash_step = None  # (B,) long on CPU, -1 = not crashed
        self.crashes = None  # per system: None or a dict describing the crash
        self._hist = deque(maxlen=history)
        self._steps, self._series = [], {}

    def stats(self, x, v):
        """Per-system statistics, each (B,) float64."""
        masses = self.masses.to(x.device)
        box = None if self.box is None else self.box.to(x.device)
        vd = v.double()
        s = {"T": temperature(v, masses),
             "KE": kinetic_energy(v, masses),
             "min_dist": min_pair_distance(x, box).double(),
             "max_speed": torch.linalg.norm(vd, dim=-1).amax(dim=1),
             "momentum": torch.linalg.norm(torch.einsum("n,bnd->bd", masses.double(), vd), dim=-1)}
        if self.bonds is not None:
            s["bond_dev"] = max_bond_deviation(x, self.bonds.to(x.device), self.bond_r0.to(x.device)).double()
        if self.energy_fn is not None:
            s["PE"] = torch.as_tensor(self.energy_fn(x, box), dtype=torch.float64, device=x.device).reshape(-1)
        return s

    def check(self, step, x, v):
        """Record statistics for positions x and velocities v (B, N, 3); returns the crash mask (B,) bool."""
        b = x.shape[0]
        if self.crash_step is None:
            self.crash_step = torch.full((b,), -1, dtype=torch.long)
            self.crashes = [None] * b
        s = self.stats(x, v)
        nan = ~torch.isfinite(x).all(dim=(1, 2)) | ~torch.isfinite(v).all(dim=(1, 2))
        none = torch.zeros_like(nan)
        flags = {"nan": nan,
                 "overlap": s["min_dist"] < self.min_dist,
                 "temperature": s["T"] > self.t_max if self.t_max is not None else none,
                 "bond": s["bond_dev"] > self.max_bond_dev if self.bonds is not None else none}
        bad = flags["nan"] | flags["overlap"] | flags["temperature"] | flags["bond"]

        s_cpu = {key: val.detach().cpu() for key, val in s.items()}
        self._hist.append((step, s_cpu))
        self._steps.append(step)
        for key, val in s_cpu.items():
            self._series.setdefault(key, []).append(val)

        new = (self.crash_step < 0) & bad.cpu()
        if new.any():
            flags_cpu = {key: val.cpu() for key, val in flags.items()}
            for bi in torch.nonzero(new).flatten().tolist():
                self.crash_step[bi] = step
                self.crashes[bi] = self._describe(bi, step, x[bi], v[bi], flags_cpu)
        return bad

    def _describe(self, bi, step, x, v, flags):
        fired = [r for r in REASONS if bool(flags[r][bi])]
        reason = fired[0]
        info = {"system": bi, "step": step, "reason": reason, "reasons": fired}
        if reason == "nan":
            atoms = torch.nonzero(~torch.isfinite(x).all(-1) | ~torch.isfinite(v).all(-1)).flatten()
            info["value"] = float("nan")
        elif reason == "overlap":
            dr = x[:, None, :] - x[None, :, :]
            if self.box is not None:
                dr = minimum_image(dr, self.box[bi].to(x.device, dr.dtype))
            d = torch.linalg.norm(dr, dim=-1)
            upper = torch.ones_like(d, dtype=torch.bool).triu(1)
            i, j = torch.nonzero(upper & (d < self.min_dist), as_tuple=True)
            order = torch.argsort(d[i, j])[:self.n_report]
            i, j = i[order], j[order]
            info["pairs"] = [[int(a), int(c), float(d[a, c])] for a, c in zip(i.tolist(), j.tolist())]
            atoms = torch.unique(torch.cat([i, j]))
            info["value"] = float(d[i[0], j[0]]) if len(i) else float("nan")
        elif reason == "temperature":
            ke_atom = 0.5 * self.masses.to(v.device).double() * (v.double() ** 2).sum(-1)
            atoms = torch.argsort(ke_atom, descending=True)[:min(5, len(ke_atom))]
            info["value"] = float(temperature(v[None], self.masses.to(v.device))[0])
        else:  # bond
            bonds = self.bonds.to(x.device)
            r = torch.linalg.norm(x[bonds[:, 0]] - x[bonds[:, 1]], dim=-1)
            dev = ((r - self.bond_r0.to(x.device)) / self.bond_r0.to(x.device)).abs()
            idx = torch.nonzero(dev > self.max_bond_dev).flatten()[:self.n_report]
            info["bonds"] = [[int(bonds[k, 0]), int(bonds[k, 1]), float(dev[k])] for k in idx.tolist()]
            atoms = torch.unique(bonds[idx].flatten())
            info["value"] = float(dev.max())
        atoms = atoms.cpu()
        info["atoms"] = atoms.tolist()
        info["atom_types"] = self.types.cpu()[atoms].tolist()
        info["history"] = [{"step": st, **{key: float(val[bi]) for key, val in stats.items()}} for st, stats in self._hist]
        return info

    def diagnostics(self):
        """JSON-serializable summary; per-system lists exclude checks at or after a system's crash."""
        if not self._steps:
            return {"crash_step": [], "crash_reason": [], "crashes": [], "series": {"step": []}}
        steps = np.asarray(self._steps, dtype=np.float64)
        t_ns = steps * self.stride_fs * 1e-6
        series = {key: torch.stack(val).numpy() for key, val in self._series.items()}  # (n_checks, B)
        b = len(self.crashes)
        out = {"crash_step": self.crash_step.tolist(),
               "crash_reason": [c["reason"] if c else None for c in self.crashes],
               "crashes": self.crashes}
        drift = {"T_drift_K_per_ns": [], "KE_drift_kJ_mol_per_ns": [], "KE_rel_drift": [],
                 "min_pair_distance": [], "max_speed": [], "max_momentum": []}
        if "PE" in series:
            drift.update({"PE_drift_kJ_mol_per_ns": [], "E_drift_kJ_mol_per_ns": []})
        for bi in range(b):
            c = int(self.crash_step[bi])
            keep = steps < c if c >= 0 else np.ones_like(steps, dtype=bool)
            t = t_ns[keep]
            ke = series["KE"][keep, bi]
            drift["T_drift_K_per_ns"].append(_slope(t, series["T"][keep, bi]))
            drift["KE_drift_kJ_mol_per_ns"].append(_slope(t, ke))
            drift["KE_rel_drift"].append(float((ke[-1] - ke[0]) / ke[0]) if len(ke) >= 2 and ke[0] > 0 else float("nan"))
            for key, col, fn in [("min_pair_distance", "min_dist", np.min), ("max_speed", "max_speed", np.max),
                                 ("max_momentum", "momentum", np.max)]:
                y = series[col][keep, bi]
                drift[key].append(float(fn(y)) if len(y) else float("nan"))
            if "PE" in series:
                pe = series["PE"][keep, bi]
                drift["PE_drift_kJ_mol_per_ns"].append(_slope(t, pe))
                drift["E_drift_kJ_mol_per_ns"].append(_slope(t, pe + ke))
        out.update(drift)
        out["series"] = {"step": self._steps, **{key: val.T.tolist() for key, val in series.items()}}
        return out
