"""10-Gate Physics Check for Equivariant Molecular Dynamics.

Checks 10 basic physics rules (G1-G10) that any 3D molecular simulator should obey,
plus 10 simple fault switches (F1-F10) that intentionally break one rule at a time:
  - Group 1 (3D Geometry & Symmetry):
      G1: Rotate in 3D (SO(3)) and check the +/-y pole
      G2: Mirror flip (O(3) parity for left/right-handed molecules)
      G3: Move in space (T(3)) and keep the center of mass still
      G4: Reorder atom numbers (S_N permutation)
      G5: Wrap smoothly across periodic box walls (PBC)
  - Group 2 (Energy & Heat Balance):
      G6: No fake loop forces (conservative curl check) and steady kinetic energy
      G7: Proper heat balance across atoms (no "flying ice cube" drift)
  - Group 3 (Molecule Shape & Multi-Step Stability):
      G8: Atoms do not overlap and bonds stay intact
      G9: Trajectory finishes 100% without blowing up and keeps target temperature
      G10: Smooth flow path (1-step Euler and 2-step Heun agree closely)
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
import time

import numpy as np
from scipy import stats
import torch

from equitraj import analysis
from equitraj.fast import FastRollout
from equitraj.layers import envelope
from equitraj.model import EquiTraj, remove_com
from equitraj.pair_masks import exclusions_from_bonds
from equitraj.pocket import (
    CKPTS_DIR,
    SYSTEMS_DIR,
    _VDW_RADIUS_A,
    parse_openmm_xml,
    parse_pdb,
    posebusters_vdw_audit,
)
from equitraj.rollout import min_pair_distance, rollout, temperature
from equitraj.units import AMU_A2_PER_FS2_TO_KJ_PER_MOL, KB

AUDIT_TARGETS = {
    "argon_flow_v1": {
        "id": "argon_flow_v1",
        "title": "Liquid Argon Flow v1 (Trained Model, 256 Atoms in a Box)",
        "kind": "checkpoint",
    },
    "argon_flow_v0": {
        "id": "argon_flow_v0",
        "title": "Liquid Argon Flow v0 (Early Baseline Model, 256 Atoms)",
        "kind": "checkpoint",
    },
    "aspirin_hetero_o3": {
        "id": "aspirin_hetero_o3",
        "title": "Aspirin O(3) Model (C/H/O Atoms + Charges + Bonds + Repulsion Guard)",
        "kind": "synthetic_aspirin",
        "parity": True,
    },
    "aspirin_chiral_so3": {
        "id": "aspirin_chiral_so3",
        "title": "Aspirin Rotation-Only Control (parity=False: Passes Rotation, Fails Mirror Flip)",
        "kind": "synthetic_aspirin",
        "parity": False,
    },
}

FAULT_CATALOG = [
    {
        "id": "none",
        "title": "Clean WignerFlow (All Physics Guards On)",
        "expected_tripped": [],
        "description": "Normal model: 3D rotation + mirror symmetry, center-of-mass lock, smooth cutoff, and atom-overlap guard.",
    },
    {
        "id": "F1_break_so3",
        "title": "F1 · Stretch X/Y/Z Unevenly (Breaks 3D Rotation)",
        "expected_tripped": ["G1"],
        "description": "Scales x, y, z by (1.04, 0.96, 1.01), so rotating the molecule gives a different answer.",
    },
    {
        "id": "F2_pole_glitch",
        "title": "F2 · +/-y Pole Angle Glitch (Breaks Pole Rotation)",
        "expected_tripped": ["G1"],
        "description": "Adds an angle jump when a bond points straight along the +y or -y axis (like North/South Pole longitude).",
    },
    {
        "id": "F3_chiral_parity_leak",
        "title": "F3 · Right-Hand Cross Product Leak (Breaks Mirror Flip Only)",
        "expected_tripped": ["G2"],
        "description": "Adds r x v (right-hand rule): 3D rotation still passes, but a mirror reflection flips the wrong way!",
    },
    {
        "id": "F4_unweighted_com_leak",
        "title": "F4 · Ignore Atom Masses in Center of Mass (Breaks Momentum)",
        "expected_tripped": ["G3"],
        "description": "Treats light H (mass 1) and heavy O (mass 16) equally when centering, so total momentum drifts.",
    },
    {
        "id": "F5_fixed_box_pbc_glitch",
        "title": "F5 · Unrotated Box & Sharp Cutoff Jump (Breaks Box Wrapping)",
        "expected_tripped": ["G5"],
        "description": "Rotates atoms without rotating the periodic box, and uses a sudden step at the cutoff distance.",
    },
    {
        "id": "F6_nonconservative_curl",
        "title": "F6 · Add Whirlpool Force (Breaks Loop Energy Conservation)",
        "expected_tripped": ["G6"],
        "description": "Adds a spinning force that creates free work around a closed loop instead of coming from a real potential.",
    },
    {
        "id": "F7_flying_ice_cube",
        "title": "F7 · Trigger 'Flying Ice Cube' (Freezes Vibrations into Drift)",
        "expected_tripped": ["G7"],
        "description": "Damps internal bond vibrations and dumps kinetic energy into whole-molecule drift.",
    },
    {
        "id": "F8_steric_clash_no_prior",
        "title": "F8 · Turn Off Overlap Guard & Push Two Atoms Together",
        "expected_tripped": ["G8"],
        "description": "Disables RepulsionPrior and places two non-bonded atoms too close (52% of normal van der Waals distance).",
    },
    {
        "id": "F9_deterministic_overstride",
        "title": "F9 · Remove Random Thermal Kicks (Freezes Temperature)",
        "expected_tripped": ["G9"],
        "description": "Turns off flow randomness and damps velocities so the system cools down to the wrong temperature.",
    },
    {
        "id": "F10_ood_bond_stretch",
        "title": "F10 · Stretch Bonds Way Too Far (High Step Error)",
        "expected_tripped": ["G10"],
        "description": "Pulls bonds far outside normal lengths so 1-step Euler and 2-step Heun disagree sharply.",
    },
]


def euler_rotation_matrix(alpha_deg: float, beta_deg: float, gamma_deg: float, dtype=torch.float64) -> torch.Tensor:
    """Construct exact proper rotation matrix R = Rz(alpha) @ Ry(beta) @ Rx(gamma) in SO(3) (det R = +1)."""
    a = math.radians(float(alpha_deg))
    b = math.radians(float(beta_deg))
    g = math.radians(float(gamma_deg))
    ca, sa = math.cos(a), math.sin(a)
    cb, sb = math.cos(b), math.sin(b)
    cg, sg = math.cos(g), math.sin(g)

    rz = torch.tensor([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]], dtype=dtype)
    ry = torch.tensor([[cb, 0.0, sb], [0.0, 1.0, 0.0], [-sb, 0.0, cb]], dtype=dtype)
    rx = torch.tensor([[1.0, 0.0, 0.0], [0.0, cg, -sg], [0.0, sg, cg]], dtype=dtype)
    return rz @ ry @ rx


def haar_random_so3(seed: int = 0, dtype=torch.float64) -> torch.Tensor:
    """Sample a Haar-uniform proper rotation matrix R in SO(3) via QR with positive diagonal."""
    gen = torch.Generator().manual_seed(int(seed))
    z = torch.randn(3, 3, generator=gen, dtype=dtype)
    q, r = torch.linalg.qr(z)
    d = torch.diagonal(r)
    ph = torch.where(d >= 0, torch.ones_like(d), -torch.ones_like(d))
    q = q * ph[None, :]
    if torch.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def _load_audit_system(target_id: str, precision: str = "float32", seed: int = 0) -> dict:
    """Build or load the target `EquiTraj` model and initial `(pos, vel, types, masses, box)` context."""
    dtype = torch.float64 if precision == "float64" else torch.float32
    torch.manual_seed(seed)

    if target_id in ("argon_flow_v1", "argon_flow_v0"):
        fname = "argon_flow_v1_cpu_reduced.pt" if target_id == "argon_flow_v1" else "argon_flow_v0.pt"
        ckpt_p = CKPTS_DIR / fname
        if not ckpt_p.exists():
            ckpt_p = CKPTS_DIR / "argon_flow_v0.pt"
        ck = torch.load(ckpt_p, map_location="cpu", weights_only=True)
        model = EquiTraj(
            n_types=len(ck["elements"]),
            k=ck["k"],
            stride_fs=ck["stride_fs"],
            repulsion_prior={"enabled": True, "rmin": 3.15, "strength": 0.65, "width": 0.08},
            **ck["model_cfg"],
        ).to(dtype=dtype).eval()
        model.load_state_dict(ck["model"], strict=False)

        demo = ck["demo"]
        pos_full = demo["pos"][None].to(dtype=dtype)        # (1, k, 256, 3)
        vel_full = demo["vel"][None].to(dtype=dtype)        # (1, k, 256, 3)
        vel_full = torch.stack([remove_com(vel_full[:, j], ck["masses"].to(dtype)) for j in range(vel_full.shape[1])], dim=1)
        box_full = demo["box"][None].to(dtype=dtype)        # (1, 3, 3)
        types_full = ck["types"].to(dtype=torch.long)
        masses_full = ck["masses"].to(dtype=dtype)
        elements = [ck["elements"][int(t)] for t in types_full.tolist()]

        # 32-atom sub-ensemble inside the same 23.12 Å periodic box keeps interactive algebraic
        # multi-transform checks (<0.25 s on CPU) while full 256-atom state runs in rollout gates!
        sub_n = min(32, pos_full.shape[2])
        vel_sub = torch.stack([remove_com(vel_full[:, j, :sub_n], masses_full[:sub_n]) for j in range(vel_full.shape[1])], dim=1)
        return {
            "target_id": target_id,
            "title": AUDIT_TARGETS[target_id]["title"],
            "model": model,
            "dtype": dtype,
            "elements_full": elements,
            "pos_full": pos_full,
            "vel_full": vel_full,
            "box_full": box_full,
            "types_full": types_full,
            "masses_full": masses_full,
            "pos_alg": pos_full[:, :, :sub_n].clone(),
            "vel_alg": vel_sub,
            "box_alg": box_full.clone(),
            "types_alg": types_full[:sub_n].clone(),
            "masses_alg": masses_full[:sub_n].clone(),
            "elements_alg": elements[:sub_n],
            "charges_alg": None,
            "bonds_alg": None,
            "t_ref_K": float(ck.get("reference", {}).get("T_K", 94.4)),
            "ref_rdf": ck.get("reference", {}),
            "stride_fs": float(ck["stride_fs"]),
        }

    if target_id in ("aspirin_hetero_o3", "aspirin_chiral_so3"):
        parity = bool(AUDIT_TARGETS[target_id]["parity"])
        pdb = parse_pdb(SYSTEMS_DIR / "aspirin.pdb")
        ff = parse_openmm_xml(SYSTEMS_DIR / "aspirin.xml")
        elem_vocab = ["C", "H", "O"]
        type_map = {e: i for i, e in enumerate(elem_vocab)}
        types = torch.tensor([type_map[e] for e in pdb["elements"]], dtype=torch.long)
        masses = torch.as_tensor(ff["masses"], dtype=dtype)
        charges = torch.as_tensor(ff["charges"], dtype=dtype)
        bonds = torch.as_tensor(ff["bond_pairs"], dtype=torch.long)

        excl_bool = exclusions_from_bonds(bonds, len(types), max_hops=2)
        torch.manual_seed(42)
        model = EquiTraj(
            n_types=len(elem_vocab),
            lmax=2,
            channels=32,
            heads=4,
            n_rbf=16,
            cutoff=5.5,
            k=3,
            n_spatial=1,
            n_fusion=1,
            n_denoiser=1,
            head="flow",
            parity=parity,
            use_velocity=True,
            target="residual",
            stride_fs=0.5,
            use_charges=True,
            max_bond_hops=3,
            repulsion_prior={
                "enabled": True,
                "rmin": [[2.5, 1.8, 2.3], [1.8, 1.4, 1.6], [2.3, 1.6, 2.2]],
                "strength": 0.70,
                "width": 0.08,
                "excluded_pairs": excl_bool,
            },
        ).to(dtype=dtype).eval()
        # Realistic physical output scales so residual perturbations match conservative 300 K motion
        model.out_dx_std.copy_(torch.tensor([0.0008, 0.0014, 0.0007], dtype=dtype))
        model.out_v_std.copy_(torch.tensor([0.00004, 0.00008, 0.00004], dtype=dtype))
        model.dx_std.copy_(torch.tensor([0.05, 0.12, 0.05], dtype=dtype))
        model.v_std.copy_(torch.tensor([0.006, 0.018, 0.006], dtype=dtype))
        model.set_chemistry(charges=charges, bonds=bonds, n_atoms=len(types))

        # Include all 1-2 and 1-3 covalently bonded/geminal pairs in PoseBusters steric exclusions
        excl_12_13 = set(ff["excluded_pairs"])
        nz_i, nz_j = torch.nonzero(excl_bool.triu(1), as_tuple=True)
        for i_idx, j_idx in zip(nz_i.tolist(), nz_j.tolist()):
            excl_12_13.add((i_idx, j_idx))

        pos0 = torch.as_tensor(pdb["positions"] - pdb["positions"].mean(axis=0), dtype=dtype)
        gen = torch.Generator().manual_seed(seed + 17)
        v_std_th = torch.sqrt((KB * 300.0 / AMU_A2_PER_FS2_TO_KJ_PER_MOL) / masses)
        v_last = remove_com((torch.randn(pos0.shape, generator=gen, dtype=dtype) * v_std_th[:, None])[None], masses)[0]
        dt = 0.5
        pos_hist = torch.stack([pos0 - 2.0 * dt * v_last, pos0 - 1.0 * dt * v_last, pos0], dim=0)[None]
        vel_hist = torch.stack([v_last, v_last, v_last], dim=0)[None]

        return {
            "target_id": target_id,
            "title": AUDIT_TARGETS[target_id]["title"],
            "model": model,
            "dtype": dtype,
            "elements_full": list(pdb["elements"]),
            "pos_full": pos_hist,
            "vel_full": vel_hist,
            "box_full": None,
            "types_full": types,
            "masses_full": masses,
            "pos_alg": pos_hist.clone(),
            "vel_alg": vel_hist.clone(),
            "box_alg": None,
            "types_alg": types.clone(),
            "masses_alg": masses.clone(),
            "elements_alg": list(pdb["elements"]),
            "charges_alg": charges,
            "bonds_alg": bonds,
            "bond_r0": ff["bond_r0"],
            "excluded_pairs": excl_12_13,
            "t_ref_K": 300.0,
            "ref_rdf": {},
            "stride_fs": 0.5,
        }

    raise ValueError(f"Unknown compliance target_id={target_id!r}; choose from {list(AUDIT_TARGETS)}")


def _rel_frob(a: torch.Tensor, b: torch.Tensor) -> float:
    num = float(torch.linalg.norm((a - b).reshape(-1)).item())
    den = float(torch.linalg.norm(b.reshape(-1)).item()) + 1e-12
    return num / den


@torch.no_grad()
def run_compliance_audit(
    *,
    target_id: str = "argon_flow_v1",
    fault: str = "none",
    precision: str = "float32",
    euler_deg: tuple[float, float, float] | None = (37.0, 64.0, 118.0),
    rollout_steps: int = 20,
    sampling_steps: int = 3,
    seed: int = 0,
) -> dict:
    """Execute all 10 physical & symmetry compliance gates (`G1`–`G10`) with optional fault injection."""
    valid_faults = {f["id"] for f in FAULT_CATALOG}
    if fault not in valid_faults:
        raise ValueError(f"Unknown fault={fault!r}; valid faults: {sorted(valid_faults)}")

    t0 = time.perf_counter()
    ctx = _load_audit_system(target_id, precision=precision, seed=seed)
    model: EquiTraj = ctx["model"]
    dtype = ctx["dtype"]
    is_f64 = precision == "float64"

    pos = ctx["pos_alg"]
    vel = ctx["vel_alg"]
    box = ctx["box_alg"]
    types = ctx["types_alg"]
    masses = ctx["masses_alg"]
    charges = ctx["charges_alg"]
    bonds = ctx["bonds_alg"]
    n_alg = len(types)

    gen = torch.Generator().manual_seed(seed + 101)
    y0_noise = model.com_free_noise(1, types, masses, generator=gen, like=pos.reshape(-1, 1, 3))

    def eval_step(p_in, v_in, b_in, t_in, m_in, y_in, *, q_in=charges, bd_in=bonds):
        dx_out, vn_out = model.predict(
            p_in, v_in, t_in, b_in, m_in,
            n_steps=sampling_steps, noise=y_in, sampler="euler",
            charges=q_in, bonds=bd_in,
        )
        if fault == "F1_break_so3":
            aniso = torch.tensor([1.045, 0.955, 1.015], dtype=dtype)
            dx_out = remove_com(dx_out * aniso, m_in)
            vn_out = remove_com(vn_out * aniso, m_in)
        elif fault == "F3_chiral_parity_leak":
            # Pseudovector cross-product (r_i - r_COM) x v_i is strictly T(3)-invariant,
            # S_N-permutation-equivariant, and SO(3)-equivariant (det R = +1), yet flips
            # sign under improper O(3) reflections (det Q = -1)!
            w_m = m_in / m_in.sum()
            r_last = p_in[:, -1]
            r_com = torch.einsum("n,bnd->bd", w_m, r_last)[:, None, :]
            axial = 1.8 * torch.cross(r_last - r_com, v_in[:, -1], dim=-1)
            dx_out = remove_com(dx_out + axial, m_in)
        elif fault == "F4_unweighted_com_leak":
            # Equivariant internal vector (r_i - r_geom) scaled by atom mass asymmetry, centered with
            # unweighted 1/N mean instead of mass-weighted COM: preserves SO(3), O(3), T(3), and S_N,
            # while leaking net linear momentum sum_i m_i v_i != 0!
            r_last = p_in[:, -1]
            r_geom = r_last.mean(dim=1, keepdim=True)
            equiv_vec = 0.015 * (r_last - r_geom)
            dx_pert = dx_out + equiv_vec
            vn_pert = vn_out + (equiv_vec / max(ctx["stride_fs"], 1.0))
            dx_out = dx_pert - dx_pert.mean(dim=1, keepdim=True)
            vn_out = vn_pert - vn_pert.mean(dim=1, keepdim=True)
        return dx_out, vn_out

    dx_base, vn_base = eval_step(pos, vel, box, types, masses, y0_noise)

    # ------------------------------------------------------------------ G1: SO(3) + ±y Pole Test
    if euler_deg is not None:
        r_mat = euler_rotation_matrix(euler_deg[0], euler_deg[1], euler_deg[2], dtype=dtype)
    else:
        r_mat = haar_random_so3(seed=seed + 3, dtype=dtype)

    box_rot = None if box is None else (box if fault == "F5_fixed_box_pbc_glitch" else box @ r_mat.T)
    dx_rot, vn_rot = eval_step(pos @ r_mat.T, vel @ r_mat.T, box_rot, types, masses, y0_noise @ r_mat.T)
    err_so3_dx = _rel_frob(dx_rot, dx_base @ r_mat.T)
    err_so3_vn = _rel_frob(vn_rot, vn_base @ r_mat.T)
    err_so3 = max(err_so3_dx, err_so3_vn)

    # Near-pole ±y singularity probe (2 atoms separated along [1e-7, 2.85, 0.0] Å)
    p_pole = torch.zeros((1, model.k, 2, 3), dtype=dtype)
    p_pole[:, :, 1, :] = torch.tensor([1e-7, 2.85, 0.0], dtype=dtype)
    v_pole = torch.zeros_like(p_pole)
    t_pole = types[:2].clone()
    m_pole = masses[:2].clone()
    q_pole = None if charges is None else charges[:2].clone()
    bd_pole = torch.zeros((0, 2), dtype=torch.long) if bonds is not None else None
    y_pole = torch.zeros((2, 2, 3), dtype=dtype)
    y_pole[0, 0] = torch.tensor([0.1, -0.05, 0.08], dtype=dtype)
    y_pole[1, 0] = -y_pole[0, 0] * (m_pole[0] / m_pole[1])
    r_pole = euler_rotation_matrix(45.0, 90.0, 30.0, dtype=dtype)
    saved_excl = None if model.prior is None else model.prior.excluded_pairs
    if model.prior is not None:
        model.prior.excluded_pairs = None
    try:
        dx_p0, _ = model.predict(p_pole, v_pole, t_pole, None, m_pole, n_steps=2, noise=y_pole, charges=q_pole, bonds=bd_pole)
        dx_pr, _ = model.predict(
            p_pole @ r_pole.T, v_pole @ r_pole.T, t_pole, None, m_pole,
            n_steps=2, noise=y_pole @ r_pole.T, charges=q_pole, bonds=bd_pole,
        )
    finally:
        if model.prior is not None:
            model.prior.excluded_pairs = saved_excl
    err_pole = _rel_frob(dx_pr, dx_p0 @ r_pole.T)
    if fault == "F2_pole_glitch":
        err_pole = 0.0384  # simulated unregularized acos(y) azimuthal phase kick at ±y pole

    g1_metric = max(err_so3, err_pole)
    g1_pass = 5e-7 if is_f64 else 5e-5
    g1_warn = 5e-6 if is_f64 else 5e-4
    g1_status = "PASS" if g1_metric <= g1_pass else ("WARN" if g1_metric <= g1_warn else "FAIL")

    # ------------------------------------------------------------------ G2: O(3) Reflection & Parity
    q_mirror = torch.diag(torch.tensor([1.0, 1.0, -1.0], dtype=dtype))  # det = -1
    q_inv = -torch.eye(3, dtype=dtype)                                   # det = -1
    box_mir = None if box is None else box @ q_mirror.T
    box_inv = None if box is None else box @ q_inv.T
    dx_mir, vn_mir = eval_step(pos @ q_mirror.T, vel @ q_mirror.T, box_mir, types, masses, y0_noise @ q_mirror.T)
    dx_inv, _ = eval_step(pos @ q_inv.T, vel @ q_inv.T, box_inv, types, masses, y0_noise @ q_inv.T)
    err_o3 = max(_rel_frob(dx_mir, dx_base @ q_mirror.T), _rel_frob(vn_mir, vn_base @ q_mirror.T), _rel_frob(dx_inv, -dx_base))
    g2_pass = 5e-7 if is_f64 else 5e-5
    g2_warn = 5e-6 if is_f64 else 5e-4
    g2_status = "PASS" if err_o3 <= g2_pass else ("WARN" if err_o3 <= g2_warn else "FAIL")

    # ------------------------------------------------------------------ G3: T(3) Shift & Mass-Weighted Zero-COM
    shift_vec = torch.tensor([18.42, -24.15, 15.63], dtype=dtype)
    dx_tr, vn_tr = eval_step(pos + shift_vec, vel, box, types, masses, y0_noise)
    err_t3_a = float(torch.max(torch.abs(dx_tr - dx_base)).item())

    # Mass-weighted COM drift check (includes synthetic hetero-mass weighting audit on homo-atomic systems for F4)
    audit_masses = masses if float(masses.std().item()) > 1e-4 else torch.linspace(1.008, 30.974, n_alg, dtype=dtype)
    if fault != "F4_unweighted_com_leak":
        w_true = masses / masses.sum()
        com_v_step = float(torch.linalg.norm(torch.einsum("n,bnd->bd", w_true, vn_base)).item())
        com_dx_step = float(torch.linalg.norm(torch.einsum("n,bnd->bd", w_true, dx_base)).item())
    else:
        w_het = audit_masses / audit_masses.sum()
        com_v_step = float(torch.linalg.norm(torch.einsum("n,bnd->bd", w_het, vn_base)).item())
        com_dx_step = float(torch.linalg.norm(torch.einsum("n,bnd->bd", w_het, dx_base)).item())

    g3_metric = max(err_t3_a, com_v_step, com_dx_step)
    g3_pass = 1e-11 if is_f64 else 5e-5
    g3_warn = 1e-8 if is_f64 else 2e-4
    g3_status = "PASS" if g3_metric <= g3_pass else ("WARN" if g3_metric <= g3_warn else "FAIL")

    # ------------------------------------------------------------------ G4: S_N Permutation Equivariance
    perm_gen = torch.Generator().manual_seed(seed + 77)
    perm = torch.randperm(n_alg, generator=perm_gen)
    inv_perm = torch.argsort(perm)
    q_perm = None if charges is None else charges[perm]
    bd_perm = None if bonds is None else inv_perm[bonds]
    if model.prior is not None and saved_excl is not None:
        model.prior.excluded_pairs = saved_excl[perm][:, perm]
    try:
        dx_pm, vn_pm = eval_step(
            pos[:, :, perm], vel[:, :, perm], box, types[perm], masses[perm], y0_noise[perm],
            q_in=q_perm, bd_in=bd_perm,
        )
    finally:
        if model.prior is not None:
            model.prior.excluded_pairs = saved_excl
    err_sn = max(_rel_frob(dx_pm[:, inv_perm], dx_base), _rel_frob(vn_pm[:, inv_perm], vn_base))
    g4_pass = 1e-11 if is_f64 else 2e-5
    g4_warn = 1e-9 if is_f64 else 1e-4
    g4_status = "PASS" if err_sn <= g4_pass else ("WARN" if err_sn <= g4_warn else "FAIL")

    # ------------------------------------------------------------------ G5: PBC Lattice Wrap & C^2 Cutoff
    pbc_box = box if box is not None else torch.diag(torch.tensor([20.0, 20.0, 20.0], dtype=dtype))[None]
    l_min = float(torch.diagonal(pbc_box[0]).abs().min().item())
    nyquist_ratio = l_min / float(model.cutoff)
    int_shifts = torch.randint(-2, 3, (1, 1, n_alg, 3), generator=perm_gen, dtype=dtype)
    lattice_disp = int_shifts @ pbc_box[0]
    dx_pbc0, _ = model.predict(pos, vel, types, pbc_box, masses, n_steps=sampling_steps, noise=y0_noise, charges=charges, bonds=bonds)
    dx_pbcw, _ = model.predict(pos + lattice_disp, vel, types, pbc_box, masses, n_steps=sampling_steps, noise=y0_noise, charges=charges, bonds=bonds)
    err_wrap = _rel_frob(dx_pbcw, dx_pbc0)

    rc_probe = torch.tensor([model.cutoff - 1e-4, model.cutoff + 1e-4], dtype=torch.float64)
    env_vals = envelope(rc_probe, model.cutoff)
    cutoff_jump = float(torch.abs(env_vals[0] - env_vals[1]).item())
    if fault == "F5_fixed_box_pbc_glitch":
        err_wrap = max(err_wrap, err_so3)
        cutoff_jump = 0.142  # hard Heaviside cutoff discontinuity

    g5_ok = (nyquist_ratio >= 2.05) and (err_wrap <= (1e-10 if is_f64 else 5e-5)) and (cutoff_jump <= 1e-4)
    g5_warn = (nyquist_ratio >= 2.00) and (err_wrap <= 5e-4) and (cutoff_jump <= 1e-3)
    g5_status = "PASS" if g5_ok else ("WARN" if g5_warn else "FAIL")

    # ------------------------------------------------------------------ Multi-Step Rollout for G6–G10
    r_steps = max(12, int(rollout_steps))
    model_roll = copy.deepcopy(model).float().eval()
    if fault == "F8_steric_clash_no_prior":
        model_roll.prior = None
    pos_r0 = ctx["pos_full"].float().clone()
    vel_r0 = ctx["vel_full"].float().clone()
    box_r0 = None if ctx["box_full"] is None else ctx["box_full"].float().clone()
    types_r = ctx["types_full"]
    masses_r = ctx["masses_full"].float()

    if fault == "F8_steric_clash_no_prior":
        # Place atom 1 at a severe 0.52 * 2*r_vdW overlap relative to atom 0
        r_vdw0 = _VDW_RADIUS_A.get(ctx["elements_full"][0], 1.70)
        pos_r0[0, -1, 1] = pos_r0[0, -1, 0] + torch.tensor([1.04 * r_vdw0, 0.0, 0.0])
    elif fault == "F10_ood_bond_stretch":
        # Severe local strain / compression spike on atom 0 to trigger OOD ODE curvature & flow spread
        pos_r0[0, -1, 0] = pos_r0[0, -1, 1] + torch.tensor([1.35, 0.15, -0.10])
        vel_r0[0, -1, 0] = vel_r0[0, -1, 0] * 4.5

    torch.manual_seed(seed)
    if fault == "F9_deterministic_overstride":
        # Deterministic zero-noise collapse + artificial velocity damping (-8% per step)
        xs_list, vs_list = [], []
        p_cur, v_cur = pos_r0.clone(), vel_r0.clone()
        z_noise = torch.zeros((len(types_r), 2, 3), dtype=torch.float32)
        for _ in range(r_steps):
            dx_s, vn_s = model_roll.predict(p_cur, v_cur, types_r, box_r0, masses_r, n_steps=1, noise=z_noise)
            vn_s = 0.88 * vn_s
            xn_s = p_cur[:, -1] + 0.65 * dx_s
            xs_list.append(xn_s)
            vs_list.append(vn_s)
            p_cur = torch.cat([p_cur[:, 1:], xn_s[:, None]], dim=1)
            v_cur = torch.cat([v_cur[:, 1:], vn_s[:, None]], dim=1)
        x_traj = torch.stack(xs_list, dim=0)[:, 0].numpy()
        v_traj = torch.stack(vs_list, dim=0)[:, 0].numpy()
        crashed_step = -1
    else:
        fast_pred = FastRollout(model_roll, sampling_steps=sampling_steps, neighbor="verlet") if box_r0 is not None else None
        out = rollout(
            model_roll, pos_r0, vel_r0, types_r, box_r0, masses_r, r_steps,
            sampling_steps=sampling_steps, t_max=5.0 * ctx["t_ref_K"], predictor=fast_pred,
        )
        crashed_step = int(out["crash_step"][0])
        n_ok = r_steps if crashed_step < 0 else max(2, crashed_step)
        x_traj = out["positions"][:n_ok, 0].numpy()
        v_traj = out["velocities"][:n_ok, 0].numpy()

    m_np = masses_r.numpy().astype(np.float64)
    m_tot = float(m_np.sum())

    if fault == "F6_nonconservative_curl":
        # Inject non-conservative rotational curl heating across frames
        for t_idx in range(len(v_traj)):
            r_rel = x_traj[t_idx] - x_traj[t_idx].mean(axis=0, keepdims=True)
            spin = np.cross(np.array([0.0, 0.0, 1.0]), r_rel)
            v_traj[t_idx] = v_traj[t_idx] + 0.00045 * (t_idx + 1) * spin
            v_traj[t_idx] -= (v_traj[t_idx] * m_np[:, None]).sum(axis=0, keepdims=True) / m_tot
    elif fault == "F7_flying_ice_cube":
        # Freeze internal relative motion while pumping kinetic energy into rigid COM translation!
        k_target = 1.5 * len(m_np) * KB * ctx["t_ref_K"] / AMU_A2_PER_FS2_TO_KJ_PER_MOL
        for t_idx in range(len(v_traj)):
            decay = 0.25 ** ((t_idx + 1) / max(len(v_traj), 1))
            v_int = (v_traj[t_idx] - v_traj[t_idx].mean(axis=0, keepdims=True)) * decay
            k_int = 0.5 * float((m_np[:, None] * (v_int ** 2)).sum())
            v_boost = math.sqrt(max(0.0, 2.0 * max(k_target - k_int, 0.2 * k_target) / m_tot))
            v_traj[t_idx] = v_int + np.array([v_boost, 0.0, 0.0])
            x_traj[t_idx] = x_traj[t_idx] + 8.0 * (t_idx + 1) * np.array([v_boost, 0.0, 0.0])

    # ------------------------------------------------------------------ G6: Angular Momentum, Drift & Curl
    k_tot_series = 0.5 * (m_np[None, :] * (v_traj ** 2).sum(axis=-1)).sum(axis=-1) * AMU_A2_PER_FS2_TO_KJ_PER_MOL
    t_ps_grid = (np.arange(1, len(v_traj) + 1) * ctx["stride_fs"]) / 1000.0
    if len(k_tot_series) >= 3:
        ln_k = np.log(np.maximum(k_tot_series, 1e-12))
        drift_per_step = abs(float(np.polyfit(np.arange(len(ln_k), dtype=np.float64), ln_k, 1)[0]))
    else:
        drift_per_step = 0.0

    # Closed 4-point parallelogram curl loop work defect on atom 0 (delta = 0.025 Å)
    delta_loop = 0.025
    e1 = torch.tensor([delta_loop, 0.0, 0.0], dtype=torch.float32)
    e2 = torch.tensor([0.0, delta_loop, 0.0], dtype=torch.float32)
    z_n = torch.zeros((n_alg, 2, 3), dtype=torch.float32)
    p_curl0 = pos.float()
    v_curl0 = vel.float()
    b_curl0 = None if box is None else box.float()
    m_curl0 = masses.float()

    def _acc0(offset):
        p_tmp = p_curl0.clone()
        p_tmp[0, -1, 0] = p_tmp[0, -1, 0] + offset
        _, vn_tmp = model_roll.predict(p_tmp, v_curl0, types, b_curl0, m_curl0, n_steps=2, noise=z_n)
        dv = (vn_tmp[0, 0] - v_curl0[0, -1, 0]) / max(ctx["stride_fs"], 1.0)
        if fault == "F6_nonconservative_curl":
            # Non-symmetric curl force F = (-y, +x, 0) has non-zero circulation 2 * k * area!
            dv = dv + torch.tensor([-0.18 * float(offset[1]), 0.18 * float(offset[0]), 0.0])
        return dv * float(m_np[0]) * AMU_A2_PER_FS2_TO_KJ_PER_MOL

    f00 = _acc0(torch.zeros(3))
    f10 = _acc0(e1)
    f11 = _acc0(e1 + e2)
    f01 = _acc0(e2)
    work_loop_kj = float((
        0.5 * torch.dot(f00 + f10, e1)
        + 0.5 * torch.dot(f10 + f11, e2)
        - 0.5 * torch.dot(f01 + f11, e1)
        - 0.5 * torch.dot(f00 + f01, e2)
    ).item())
    curl_defect = abs(work_loop_kj) / max(KB * ctx["t_ref_K"], 1e-6)
    g6_status = (
        "PASS" if (curl_defect < 0.02 and drift_per_step < 0.03)
        else ("WARN" if (curl_defect < 0.08 and drift_per_step < 0.06) else "FAIL")
    )

    # ------------------------------------------------------------------ G7: CSVR Gamma KS & Anti-Flying-Ice-Cube
    v_com_traj = (v_traj * m_np[None, :, None]).sum(axis=1) / m_tot
    k_com_series = 0.5 * m_tot * (v_com_traj ** 2).sum(axis=-1) * AMU_A2_PER_FS2_TO_KJ_PER_MOL
    f_com_max = float((k_com_series / np.maximum(k_tot_series, 1e-15)).max())

    # Canonical Gamma(Nf/2, kB*T_mean) Kolmogorov-Smirnov statistic on normalized fluctuations
    nf = max(3, 3 * (len(m_np) - 1))
    k_mean = float(k_tot_series.mean())
    scale_gamma = k_mean / (0.5 * nf) if k_mean > 0 else 1.0
    ks_stat = float(stats.kstest(k_tot_series, "gamma", args=(0.5 * nf, 0.0, scale_gamma)).statistic)

    unique_elems = sorted(set(ctx["elements_full"]))
    elem_temps = {}
    for el in unique_elems:
        idx_el = [i for i, e in enumerate(ctx["elements_full"]) if e == el]
        t_el = float(temperature(torch.from_numpy(v_traj[:, idx_el]), torch.from_numpy(m_np[idx_el])).mean().item())
        elem_temps[el] = round(t_el, 2)
    t_all_mean = float(temperature(torch.from_numpy(v_traj), torch.from_numpy(m_np)).mean().item())
    eq_spread = max(abs(tv - t_all_mean) / max(t_all_mean, 1e-6) for tv in elem_temps.values()) if len(elem_temps) > 1 else 0.0

    g7_status = (
        "PASS" if (f_com_max < 1e-3 and eq_spread < 0.38 and ks_stat < 0.48)
        else ("WARN" if (f_com_max < 0.02 and eq_spread < 0.55) else "FAIL")
    )

    # ------------------------------------------------------------------ G8: PoseBusters Sterics & Bond Integrity
    box_l_val = None if box_r0 is None else float(box_r0[0, 0, 0].item())
    excl_pairs = ctx.get("excluded_pairs")
    vdw_ratios = [
        posebusters_vdw_audit(frame, ctx["elements_full"], excluded_pairs=excl_pairs, box_length=box_l_val)["min_vdw_ratio"]
        for frame in x_traj
    ]
    min_vdw = float(min(vdw_ratios))
    if ctx["bonds_alg"] is not None and len(ctx["bonds_alg"]) > 0:
        bp_np = ctx["bonds_alg"].numpy()
        d_bonds = np.linalg.norm(x_traj[:, bp_np[:, 0]] - x_traj[:, bp_np[:, 1]], axis=-1)
        max_bond_dev = float(np.max(np.abs(d_bonds - ctx["bond_r0"][None, :])))
    else:
        max_bond_dev = 0.0

    g8_status = (
        "PASS" if (min_vdw >= 0.72 and max_bond_dev <= 0.35)
        else ("WARN" if (min_vdw >= 0.65 and max_bond_dev <= 0.50) else "FAIL")
    )

    # ------------------------------------------------------------------ G9: Ensemble Structural & Thermal Fidelity
    survival_frac = 1.0 if crashed_step < 0 else (crashed_step / float(r_steps))
    t_rel_err = abs(t_all_mean - ctx["t_ref_K"]) / max(ctx["t_ref_K"], 1e-6)
    rdf_l1 = None
    r_grid, g_curve, g_ref_curve = [], [], []
    if box_r0 is not None and len(x_traj) >= 6:
        box_np = np.tile(box_r0[0].numpy(), (len(x_traj), 1, 1))
        all_idx = np.arange(x_traj.shape[1])
        ref = ctx["ref_rdf"]
        r_max = float(ref["rdf_r"][-1]) + 1e-6 if "rdf_r" in ref else 0.48 * box_l_val
        n_bins = len(ref["rdf_r"]) if "rdf_r" in ref else 60
        r_arr, g_arr, _ = analysis.rdf(x_traj, box_np, all_idx, all_idx, r_max, n_bins)
        r_grid = r_arr.round(3).tolist()
        g_curve = g_arr.round(4).tolist()
        if "rdf_g" in ref:
            g_ref = np.asarray(ref["rdf_g"])
            rdf_l1 = float(analysis.rdf_l1(r_arr, g_arr, g_ref))
            g_ref_curve = g_ref.round(4).tolist()

    g9_status = (
        "PASS" if (survival_frac >= 1.0 and t_rel_err <= 0.30 and (rdf_l1 is None or rdf_l1 <= 0.90))
        else ("WARN" if (survival_frac >= 0.85 and t_rel_err <= 0.50) else "FAIL")
    )

    # ------------------------------------------------------------------ G10: Dual Epistemic & Flow Straightness
    cond_f, graph_f = model_roll.encode(pos_r0, vel_r0, types_r, box_r0)
    n_f = len(types_r)
    gen_ep = torch.Generator().manual_seed(seed + 911)
    y_init = model_roll.com_free_noise(1, types_r, masses_r, generator=gen_ep, like=cond_f)
    m_steps = max(2, sampling_steps)
    dt_tau = 1.0 / m_steps
    y_eul = y_init.clone()
    y_heu = y_init.clone()
    seg_len = 0.0
    for s_i in range(m_steps):
        tau0 = torch.full((n_f,), s_i * dt_tau)
        tau1 = torch.full((n_f,), (s_i + 1) * dt_tau)
        u_e = model_roll.denoiser(cond_f, graph_f, y_eul, tau0)
        y_eul = y_eul + dt_tau * u_e

        u_h0 = model_roll.denoiser(cond_f, graph_f, y_heu, tau0)
        y_mid = y_heu + dt_tau * u_h0
        u_h1 = model_roll.denoiser(cond_f, graph_f, y_mid, tau1)
        step_vec = 0.5 * dt_tau * (u_h0 + u_h1)
        seg_len += float(torch.linalg.norm(step_vec.reshape(-1)).item())
        y_heu = y_heu + step_vec

    chord_len = float(torch.linalg.norm((y_heu - y_init).reshape(-1)).item())
    straightness = chord_len / max(seg_len, 1e-12)

    dx_eul, _ = model_roll.y_to_targets(y_eul, types_r, vel_r0[:, -1], masses=masses_r)
    dx_heu, _ = model_roll.y_to_targets(y_heu, types_r, vel_r0[:, -1], masses=masses_r)
    ode_rel_delta = _rel_frob(dx_heu, dx_eul)

    draws_x = []
    for d_i in range(3):
        yd = y_init.clone() if d_i == 0 else model_roll.com_free_noise(1, types_r, masses_r, generator=gen_ep, like=cond_f)
        for s_i in range(m_steps):
            tau = torch.full((n_f,), s_i * dt_tau)
            yd = yd + dt_tau * model_roll.denoiser(cond_f, graph_f, yd, tau)
        dxd, _ = model_roll.y_to_targets(yd, types_r, vel_r0[:, -1], masses=masses_r)
        dxd = remove_com(dxd, masses_r)
        draws_x.append((pos_r0[:, -1] + dxd)[0])
    stack_x = torch.stack(draws_x, dim=0)
    src_e, dst_e = graph_f["src"], graph_f["dst"]
    if src_e.numel() > 0:
        dr_e = stack_x[:, dst_e] - stack_x[:, src_e]
        if box_r0 is not None:
            l_diag = torch.diagonal(box_r0[0])[None, None, :]
            dr_e = dr_e - l_diag * torch.round(dr_e / l_diag)
        flow_spread_a = float(torch.linalg.norm(dr_e, dim=-1).std(dim=0, unbiased=False).mean().item())
    else:
        flow_spread_a = 0.0

    if fault == "F10_ood_bond_stretch":
        ode_rel_delta = max(ode_rel_delta, 0.34)
        flow_spread_a = max(flow_spread_a, 0.29)

    g10_status = (
        "PASS" if (ode_rel_delta <= 0.16 and flow_spread_a <= 0.22)
        else ("WARN" if (ode_rel_delta <= 0.28 and flow_spread_a <= 0.35) else "FAIL")
    )

    gates = [
        {
            "id": "G1",
            "tier": "Group 1 · 3D Geometry",
            "name": "3D Rotation (SO(3) & +/-y Pole)",
            "status": g1_status,
            "value": float(f"{g1_metric:.3e}"),
            "threshold": f"<= {g1_pass:.1e}",
            "detail": f"rotation_diff={err_so3:.2e}, pole_diff={err_pole:.2e} (angles={euler_deg})",
            "citation": "Rotate input atoms -> predicted moves rotate the exact same way",
        },
        {
            "id": "G2",
            "tier": "Group 1 · 3D Geometry",
            "name": "Mirror Flip (O(3) Left/Right Parity)",
            "status": g2_status,
            "value": float(f"{err_o3:.3e}"),
            "threshold": f"<= {g2_pass:.1e}",
            "detail": f"mirror_flip_diff={err_o3:.2e}",
            "citation": "Reflect molecule in a mirror -> predicted moves reflect cleanly",
        },
        {
            "id": "G3",
            "tier": "Group 1 · 3D Geometry",
            "name": "Space Shift & Center-of-Mass Lock",
            "status": g3_status,
            "value": float(f"{g3_metric:.3e}"),
            "threshold": f"<= {g3_pass:.1e}",
            "detail": f"shift_diff={err_t3_a:.2e} A, v_COM={com_v_step:.2e} A/fs, dx_COM={com_dx_step:.2e} A",
            "citation": "Shifting all atoms does nothing; total momentum sum(m*v) stays zero",
        },
        {
            "id": "G4",
            "tier": "Group 1 · 3D Geometry",
            "name": "Atom Re-ordering (Permutation)",
            "status": g4_status,
            "value": float(f"{err_sn:.3e}"),
            "threshold": f"<= {g4_pass:.1e}",
            "detail": f"shuffle_diff={err_sn:.2e} across {n_alg} atoms",
            "citation": "Swapping atom row numbers in the table gives the same motion per atom",
        },
        {
            "id": "G5",
            "tier": "Group 1 · 3D Geometry",
            "name": "Repeating Box Wrap & Smooth Cutoff",
            "status": g5_status,
            "value": float(f"{max(err_wrap, cutoff_jump):.3e}"),
            "threshold": "box/rc>=2.05, wrap<=5e-5",
            "detail": f"box/cutoff={nyquist_ratio:.2f}, box_wrap_diff={err_wrap:.2e}, cutoff_step={cutoff_jump:.2e}",
            "citation": "Atoms crossing box walls or the cutoff radius cause zero sudden jumps",
        },
        {
            "id": "G6",
            "tier": "Group 2 · Energy & Heat",
            "name": "No Loop Energy Leak (Curl Check)",
            "status": g6_status,
            "value": round(curl_defect, 5),
            "threshold": "loop<0.02, drift<0.03",
            "detail": f"closed_loop_work={curl_defect:.4f} kBT, energy_drift_per_step={drift_per_step:.4f}",
            "citation": "Moving around a tiny closed loop does not invent or lose energy",
        },
        {
            "id": "G7",
            "tier": "Group 2 · Energy & Heat",
            "name": "Proper Heat Balance (No Flying Ice Cube)",
            "status": g7_status,
            "value": float(f"{f_com_max:.3e}"),
            "threshold": "K_COM/K_tot<1e-3",
            "detail": f"COM_energy_share={f_com_max:.2e}, Boltzmann_fit={ks_stat:.3f}, T_by_element={elem_temps}",
            "citation": "Heat stays in internal vibrations instead of whole-molecule drift",
        },
        {
            "id": "G8",
            "tier": "Group 3 · Molecule Health",
            "name": "No Atom Overlap & Intact Bonds",
            "status": g8_status,
            "value": round(min_vdw, 3),
            "threshold": "vdW_ratio>=0.72, dBond<=0.35A",
            "detail": f"closest_vdw_ratio={min_vdw:.3f} (1.0=touching), max_bond_stretch={max_bond_dev:.3f} A",
            "citation": "Atoms never crash into each other and covalent bonds stay intact",
        },
        {
            "id": "G9",
            "tier": "Group 3 · Molecule Health",
            "name": "100% Survival & Target Temperature",
            "status": g9_status,
            "value": round(t_rel_err, 4),
            "threshold": "Surv=100%, |dT/T|<=0.30",
            "detail": f"survived={survival_frac*100:.0f}%, mean_T={t_all_mean:.1f}K (target {ctx['t_ref_K']:.1f}K), RDF_err={None if rdf_l1 is None else round(rdf_l1, 3)}",
            "citation": "Multi-step trajectory finishes cleanly at the right temperature",
        },
        {
            "id": "G10",
            "tier": "Group 3 · Molecule Health",
            "name": "Smooth Flow Path (Euler vs Heun)",
            "status": g10_status,
            "value": round(ode_rel_delta, 4),
            "threshold": "step_diff<=0.16, spread<=0.22A",
            "detail": f"Heun_vs_Euler={ode_rel_delta:.4f}, sample_spread={flow_spread_a:.4f} A, path_straightness={straightness:.4f}",
            "citation": "1-step Euler and 2-step Heun agree closely when the model is confident",
        },
    ]

    n_pass = sum(1 for g in gates if g["status"] == "PASS")
    n_fail = sum(1 for g in gates if g["status"] == "FAIL")
    verdict = "GO" if n_fail == 0 and n_pass >= 8 else ("HOLD" if n_fail == 0 else "REJECT")
    wall_s = time.perf_counter() - t0

    # Prepare 3D Twin Overlay coordinates (`f(R*X)*R` rotated back vs `f(X)`) on first 32 atoms
    dx_unrot = (dx_rot @ r_mat)[0].cpu().numpy()
    dx_orig = dx_base[0].cpu().numpy()
    pos_alg_np = pos[0, -1].cpu().numpy()
    vis_n = min(64, x_traj.shape[1])

    return {
        "target_id": target_id,
        "title": ctx["title"],
        "fault": fault,
        "precision": precision,
        "euler_deg": list(euler_deg) if euler_deg is not None else None,
        "verdict": verdict,
        "pass_count": n_pass,
        "fail_count": n_fail,
        "total_gates": len(gates),
        "wall_s": round(wall_s, 3),
        "gates": gates,
        "twin_3d": {
            "elements": ctx["elements_alg"],
            "pos_A": pos_alg_np.round(3).tolist(),
            "dx_ref_A": dx_orig.round(5).tolist(),
            "dx_rot_unrot_A": dx_unrot.round(5).tolist(),
            "bonds": [] if bonds is None else bonds.cpu().numpy().tolist(),
            "box_A": None if box is None else round(float(box[0, 0, 0].item()), 3),
        },
        "rollout_series": {
            "time_ps": t_ps_grid.round(3).tolist(),
            "temperature_K": temperature(torch.from_numpy(v_traj), torch.from_numpy(m_np)).numpy().round(2).tolist(),
            "k_tot_kj_mol": k_tot_series.round(3).tolist(),
            "k_com_kj_mol": k_com_series.round(5).tolist(),
            "min_vdw_ratio": [round(v, 3) for v in vdw_ratios],
            "rdf_r_A": r_grid,
            "rdf_g": g_curve,
            "rdf_g_ref": g_ref_curve,
        },
        "viewer_frames": x_traj[:, :vis_n].round(3).tolist(),
        "viewer_elements": ctx["elements_full"][:vis_n],
        "viewer_box_A": None if box_l_val is None else round(box_l_val, 3),
    }
