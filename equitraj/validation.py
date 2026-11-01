"""Species-resolved MD validation. Inputs: Å, Å/fs, fs, amu; row-vector cells.

Diffusion and conductivities are finite-trajectory estimators, not convergence
claims. No center-of-mass subtraction is performed; choose a common rest frame
before comparing. Einstein-Helfand uses the collective charge displacement,
including interparticle correlations. Charges are in elementary-charge units.
"""
from itertools import combinations_with_replacement

import numpy as np

from equitraj import analysis
from equitraj.adapters import boxes_array, minimum_image, unwrap_positions
from equitraj.units import AMU_A2_PER_FS2_TO_KJ_PER_MOL, KB

_E = 1.602176634e-19
_KB_SI = 1.380649e-23


def _fit(t, y, window):
    mask = (t >= window[0]) & (t <= window[1])
    if mask.sum() < 2:
        raise ValueError("fit window must include at least two sampled lags")
    return float(np.polyfit(t[mask], np.asarray(y)[mask], 1)[0])


def _rdf(x, cells, a, b, edges):
    hist = np.zeros(len(edges) - 1)
    same = np.array_equal(a, b)
    pairs = len(a) * (len(b) - int(same))
    shell = 4 * np.pi / 3 * np.diff(edges ** 3)
    norm = np.zeros_like(hist)
    for t, frame in enumerate(x):
        dr = frame[a, None] - frame[None, b]
        if cells is not None:
            dr = minimum_image(dr, cells[t])
        dist = np.linalg.norm(dr, axis=-1)
        if same:
            dist = dist[~np.eye(len(a), dtype=bool)]
        hist += np.histogram(dist, edges)[0]
        if cells is not None:
            norm += pairs * shell / abs(np.linalg.det(cells[t]))
    if cells is None:
        norm = np.full_like(hist, hist.sum()) * np.diff(edges)
    return np.divide(hist, norm, out=np.zeros_like(hist), where=norm > 0)


def _prepare(data, elements, masses, box, dt_fs, wrapped, min_distance, max_temperature, dof):
    x = np.asarray(data["positions"], dtype=float)
    v = np.asarray(data["velocities"], dtype=float)
    if x.ndim != 3 or x.shape != v.shape or x.shape[-1] != 3 or len(x) < 2:
        raise ValueError("positions/velocities must be matching (T,N,3) with T>=2")
    elem = np.asarray(data.get("elements", elements)).astype(str)
    m = np.asarray(data.get("masses", masses), dtype=float)
    if elem.shape != (x.shape[1],) or m.shape != (x.shape[1],) or np.any(m < 0) or not np.isfinite(m).all():
        raise ValueError("per-atom elements and finite nonnegative masses required")
    interval = data.get("dt_fs", dt_fs)
    if interval is None:
        raise ValueError("positive frame interval dt_fs required")
    dt = float(interval)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("positive frame interval dt_fs required")
    cell = data.get("box", box)
    cells = None if cell is None else boxes_array(cell, len(x))
    ndof = int(dof if dof is not None else 3 * len(m))
    if ndof <= 0:
        raise ValueError("dof must be positive")
    temp = np.sum(m[None, :, None] * v ** 2, axis=(1, 2)) * AMU_A2_PER_FS2_TO_KJ_PER_MOL / (ndof * KB)
    valid = (np.isfinite(x).all(axis=(1, 2)) & np.isfinite(v).all(axis=(1, 2))
             & np.isfinite(temp))
    if max_temperature is not None:
        valid &= temp <= max_temperature
    if min_distance is not None:
        for t in np.flatnonzero(valid):
            dr = x[t, :, None] - x[t, None, :]
            if cells is not None:
                dr = minimum_image(dr, cells[t])
            distances = np.linalg.norm(dr, axis=-1)
            np.fill_diagonal(distances, np.inf)
            valid[t] &= distances.min() >= min_distance
    bad = np.flatnonzero(~valid)
    stop = int(bad[0]) if len(bad) else len(x)
    # Optional failure metadata uses frame indices, not integrator steps.
    failure = data.get("stable_frames")
    if failure is not None:
        stop = min(stop, max(0, int(failure)))
    info = dict(input_frames=len(x), stable_frames=stop,
                stable_length_ns=max(0, stop - 1) * dt / 1e6,
                first_unstable_frame=stop if stop < len(x) else None)
    if stop < 2:
        return None, info
    x, v, temp = x[:stop], v[:stop], temp[:stop]
    cells = cells[:stop] if cells is not None else None
    if wrapped:
        x = unwrap_positions(x, cells)
    return dict(x=x, v=v, elements=elem, masses=m, dt=dt, box=cells, temperature=temp), info


def _metrics(d, edges, max_lag, fit_window_fs, charges, temperature_K, energy):
    x, v, cells, dt = d["x"], d["v"], d["box"], d["dt"]
    lag = min(len(x) - 1, max_lag if max_lag is not None else max(1, len(x) // 2))
    if lag < 1:
        raise ValueError("max_lag must be >=1")
    times = np.arange(lag + 1) * dt
    window = (fit_window_fs if fit_window_fs is not None else
              (times[max(0, lag // 2)], times[-1]))
    temp = d["temperature"]
    temperature = float(temp.mean())
    species = sorted(set(d["elements"]))
    indices = {s: np.flatnonzero(d["elements"] == s) for s in species}
    result = dict(lag_fs=times.tolist(), fit_window_fs=list(window), species={}, rdf={},
                  temperature_mean_K=temperature,
                  temperature_drift_K_per_ns=_fit(np.arange(len(x)) * dt / 1e6, temp, (0, np.inf)))
    for s, idx in indices.items():
        msd = analysis.msd(x[:, idx], np.arange(lag + 1))
        diffusion = _fit(times, msd, window) / 6
        vacf = analysis.vacf(v[:, idx], lag + 1)
        integral = np.concatenate(([0.0], np.cumsum((vacf[:-1] + vacf[1:]) * dt / 2)))
        result["species"][s] = dict(msd_A2=msd.tolist(), diffusion_A2_per_fs=diffusion,
                                     diffusion_cm2_per_s=diffusion * 0.1,
                                     vacf_A2_per_fs2=vacf.tolist(), vacf_integral_A2_per_fs=integral.tolist(),
                                     vacf_diffusion_A2_per_fs=float(integral[-1] / 3))
    result["rdf_r_A"] = ((edges[1:] + edges[:-1]) / 2).tolist()
    result["rdf_kind"] = "g(r)" if cells is not None else "pair_distance_density_per_A"
    for a, b in combinations_with_replacement(species, 2):
        result["rdf"][f"{a}-{b}"] = _rdf(x, cells, indices[a], indices[b], edges).tolist()
    if charges is not None:
        if cells is None:
            raise ValueError("conductivity requires a periodic volume")
        if not all(s in charges for s in species):
            raise ValueError("charges must specify every species, including neutral species")
        t_k = float(temperature_K if temperature_K is not None else temperature)
        if not np.isfinite(t_k) or t_k <= 0:
            raise ValueError("conductivity requires positive temperature_K or kinetic temperature")
        volume = float(np.mean(np.abs(np.linalg.det(cells)))) * 1e-30
        # D [Å²/fs] -> m²/s = 1e-5. q is supplied in units of e.
        ne = sum(len(indices[s]) * charges[s] ** 2 * result["species"][s]["diffusion_A2_per_fs"]
                 for s in species) * _E ** 2 * 1e-5 / (volume * _KB_SI * t_k)
        q = np.array([charges[s] for s in d["elements"]], dtype=float)
        if not np.isfinite(q).all():
            raise ValueError("charges must be finite")
        collective = np.einsum("n,tnc->tc", q, x)[:, None, :]
        charge_msd = analysis.msd(collective, np.arange(lag + 1))
        eh = _fit(times, charge_msd, window) * _E ** 2 * 1e-5 / (6 * volume * _KB_SI * t_k)
        result["conductivity"] = dict(nernst_einstein_S_per_m=float(ne), einstein_helfand_S_per_m=float(eh),
                                        collective_charge_msd_e2_A2=charge_msd.tolist(),
                                        temperature_K=t_k, mean_volume_m3=volume,
                                        net_charge_e=float(q.sum()))
    if energy is not None:
        # Callable receives one frame (Å, Å/fs, amu, Å cell or None), returns TOTAL kJ/mol.
        values = np.array([energy(x[t], v[t], d["masses"], None if cells is None else cells[t])
                           for t in range(len(x))], dtype=float)
        if values.shape != (len(x),) or not np.isfinite(values).all():
            raise ValueError("energy callable must return a finite scalar total energy")
        result["energy_mean_kJmol"] = float(values.mean())
        result["energy_drift_kJmol_per_ns"] = _fit(np.arange(len(x)) * dt / 1e6, values, (0, np.inf))
    return result


def validate(rollout, reference, *, elements=None, masses=None, box=None, dt_fs=None,
             wrapped=False, r_max=None, n_bins=100, max_lag=None, fit_window_fs=None,
             charges=None, temperature_K=None, dof=None, energy=None,
             min_distance=None, max_temperature=None):
    """Compare trajectory mappings with positions and velocities in Å and Å/fs.

    Mappings may supply elements, masses, box and dt_fs (saved-frame spacing),
    overriding shared arguments. Inputs are assumed unwrapped unless wrapped=True.
    Each series uses its own time grid; diffusion fits use the same fs window.
    Nonfinite frames and optional distance/temperature failures truncate analysis
    at the first failure. stable_frames can explicitly mark a valid prefix.
    Default kinetic DOF is 3N; pass dof for constraints/COM removal.
    RDF radius is limited to half the minimum cell height of both series.
    """
    prepared, report = {}, {"units": "Å, fs, amu; conductivity S/m", "errors": {}}
    for name, data in (("rollout", rollout), ("reference", reference)):
        prepared[name], report[name] = _prepare(data, elements, masses, box, dt_fs, wrapped,
                                                 min_distance, max_temperature, dof)
    if any(d is None for d in prepared.values()):
        report["status"] = "insufficient stable frames"
        return report
    a, b = prepared.values()
    if not np.array_equal(a["elements"], b["elements"]):
        raise ValueError("rollout and reference must have the same per-atom species ordering")
    if (a["box"] is None) != (b["box"] is None):
        raise ValueError("both trajectories must have matching periodicity")
    if max_lag is not None and (max_lag < 1 or int(max_lag) != max_lag):
        raise ValueError("max_lag must be a positive integer")
    if fit_window_fs is None:
        end = min(min(len(d["x"]) - 1,
                      max_lag if max_lag is not None else max(1, len(d["x"]) // 2)) * d["dt"]
                  for d in prepared.values())
        start = end / 2 if end > 2 * max(a["dt"], b["dt"]) else 0
        fit_window_fs = (start, end)
    radii = []
    for d in prepared.values():
        if d["box"] is not None:
            # Reciprocal-vector norms give perpendicular cell heights.
            heights = 1 / np.linalg.norm(np.linalg.inv(d["box"]), axis=1)
            radii.append(float(heights.min() / 2))
    if r_max is None:
        if not radii:
            raise ValueError("r_max required for nonperiodic data")
        r_max = min(radii)
    if (not np.isfinite(r_max) or r_max <= 0 or n_bins < 1 or int(n_bins) != n_bins
            or (radii and r_max > min(radii) * (1 + 1e-7))):
        raise ValueError("invalid RDF grid or radius exceeds half the minimum cell height")
    edges = np.linspace(0, r_max, n_bins + 1)
    for name, d in prepared.items():
        report[name].update(_metrics(d, edges, max_lag, fit_window_fs, charges, temperature_K, energy))
    for key in report["reference"]["rdf"]:
        g = np.array(report["rollout"]["rdf"][key])
        ref = np.array(report["reference"]["rdf"][key])
        l1 = float(np.sum(np.abs(g - ref) * np.diff(edges)))
        area = float(np.sum(np.abs(ref) * np.diff(edges)))
        report["errors"][key] = dict(rdf_L1_A=l1, rdf_relative_area_error=l1 / area if area else None,
                                      rdf_signed_area_error_A=float(np.sum((g - ref) * np.diff(edges))))
    report["status"] = "ok"
    return report


def report_markdown(report):
    """Render compact, unit-labeled metrics; full curves remain in the dict/JSON."""
    lines = ["# MD validation", "", f"Status: {report['status']}", ""]
    for name in ("rollout", "reference"):
        d = report[name]
        lines += [f"## {name.capitalize()}", "", f"Stable length: {d['stable_length_ns']:.6g} ns ({d['stable_frames']} frames).", ""]
        if "species" not in d:
            continue
        lines += [f"Temperature: {d['temperature_mean_K']:.6g} K; drift {d['temperature_drift_K_per_ns']:.6g} K/ns.", "",
                  "| Species | D (cm²/s) | VACF integral / 3 (Å²/fs) |", "|---|---:|---:|"]
        for s, m in d["species"].items():
            lines.append(f"| {s} | {m['diffusion_cm2_per_s']:.6g} | {m['vacf_diffusion_A2_per_fs']:.6g} |")
        lines.append("")
        if "conductivity" in d:
            c = d["conductivity"]
            lines += [f"Conductivity: Nernst–Einstein {c['nernst_einstein_S_per_m']:.6g} S/m; "
                      f"Einstein–Helfand {c['einstein_helfand_S_per_m']:.6g} S/m.", ""]
        if "energy_drift_kJmol_per_ns" in d:
            lines += [f"Total energy drift: {d['energy_drift_kJmol_per_ns']:.6g} kJ/mol/ns.", ""]
    if report["errors"]:
        lines += ["| Pair | RDF L1 (Å) | Relative area error |", "|---|---:|---:|"]
        for pair, e in report["errors"].items():
            relative = e["rdf_relative_area_error"]
            lines.append(f"| {pair} | {e['rdf_L1_A']:.6g} | {relative if relative is not None else 'undefined'} |")
    return "\n".join(lines) + "\n"
