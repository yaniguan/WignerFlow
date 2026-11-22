"""Inference-time switch S6: CSVR thermostat (Bussi, Donadio, Parrinello 2007), TrajCast style.

Applied to the model's predicted velocities after every rollout step. It is NOT part of the model;
any result that uses it is labeled as such. It answers: "is the model stable once its
kinetic-energy drift is corrected?"
"""

import math

import torch

from equitraj.units import AMU_A2_PER_FS2_TO_KJ_PER_MOL, KB


def csvr(vel, masses, t_target, tau_fs, dt_fs, dof, generator=None):
    """Rescale velocities (B, N, 3) so that the kinetic energy relaxes to t_target with time constant tau.

    Uses the exact CSVR update of the kinetic energy K:
        K' = K e^{-dt/tau} + K0/dof (1 - e^{-dt/tau}) (R1^2 + sum_{i=2}^{dof} Ri^2)
             + 2 e^{-dt/(2 tau)} sqrt(K0/dof (1 - e^{-dt/tau}) K) R1
    with K0 = dof k_B T / 2 and Ri standard normal.
    """
    if dof < 1:
        raise ValueError(f"dof must be >= 1, got {dof}")
    v = vel.double()
    m = masses.to(device=v.device, dtype=torch.float64)
    k = 0.5 * torch.einsum("n,bnd->b", m, v**2) * AMU_A2_PER_FS2_TO_KJ_PER_MOL
    k0 = 0.5 * dof * KB * t_target
    c = math.exp(-dt_fs / tau_fs)
    gen_dev = v.device if generator is None else generator.device
    r1 = torch.randn(k.shape, generator=generator, dtype=k.dtype, device=gen_dev).to(v.device)
    # sum of (dof - 1) squared normals = chi-squared with dof - 1 degrees of freedom = 2 * Gamma((dof-1)/2)
    if dof == 1:
        s = torch.zeros_like(k)
    elif generator is None:
        gamma = torch.distributions.Gamma(torch.tensor((dof - 1) / 2.0, dtype=k.dtype), torch.tensor(1.0, dtype=k.dtype))
        s = 2.0 * gamma.sample(k.shape).to(k.device)
    else:
        conc = torch.full(k.shape, (dof - 1) / 2.0, dtype=k.dtype, device=gen_dev)
        s = (2.0 * torch._standard_gamma(conc, generator=generator)).to(v.device)
    k_new = k * c + k0 / dof * (1 - c) * (r1**2 + s) + 2 * math.sqrt(c) * torch.sqrt(k0 / dof * (1 - c) * k) * r1
    k_new = torch.clamp(k_new, min=0.0)
    scale = torch.where(k > 0, torch.sqrt(k_new / k.clamp_min(1e-300)), torch.zeros_like(k))
    v_out = v * scale[:, None, None]
    if (k == 0).any():
        for b_idx in torch.nonzero(k == 0).flatten().tolist():
            if k_new[b_idx] > 0:
                z = torch.randn(v[b_idx].shape, generator=generator, dtype=v.dtype, device=gen_dev).to(v.device)
                z = z * m.rsqrt()[:, None]
                if v.shape[1] > 1 and dof <= 3 * v.shape[1] - 3:
                    z = z - (z * m[:, None]).sum(0, keepdim=True) / m.sum()
                kz = 0.5 * (m[:, None] * z**2).sum() * AMU_A2_PER_FS2_TO_KJ_PER_MOL
                if kz > 0:
                    v_out[b_idx] = z * torch.sqrt(k_new[b_idx] / kz)
    return v_out.to(vel.dtype)
