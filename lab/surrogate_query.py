"""Surrogate query: the lab agent asks WignerFlow for a cheap physics check.

A query is a small JSON dict (schema `wignerflow.surrogate_query/v1`). The reasoner
(YLM) sends either a `system` name or an `ouroboros-ocsr` `smiles` string; this module
validates it, maps supported SMILES without guessing, runs a model-only rollout from a
release checkpoint, and returns a JSON report (`wignerflow.surrogate_report/v1`) plus
short `[physics]` lines for `YLM`'s `lab.protocol_qa.build_prompt(..., physics_lines)`.
`run_batch_queries` packs multiple queries into one batched rollout and returns one
report (including `gate_verdict`) per query.

Only systems in REGISTRY are in domain. Today that is LJ argon (checkpoints/README.md).
Anything else returns status "unsupported". The report holds raw rollout numbers from
this run only. It is not a benchmark result.

    python -m lab.surrogate_query --query lab/fixtures/argon_query.json --dry-run
    python -m lab.surrogate_query --query lab/fixtures/argon_query.json --context
    python -m lab.surrogate_query --query lab/fixtures/ocsr_argon_query.json --context
"""

import argparse
import contextlib
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

QUERY_SCHEMA = "wignerflow.surrogate_query/v1"
REPORT_SCHEMA = "wignerflow.surrogate_report/v1"

# Facts copied from checkpoints/README.md; tests/test_lab_scale.py checks them against the file.
REGISTRY = {
    "argon": {
        "checkpoint": "checkpoints/argon_flow_v0.pt",
        "elements": ["Ar"],
        "n_atoms": 256,
        "stride_fs": 50.0,
        "source": "checkpoints/README.md",
    },
}

# Exact lookup from ouroboros-ocsr SMILES to a supported REGISTRY key. Never guess.
SMILES_TO_SYSTEM = {
    "[Ar]": "argon",
    "[18Ar]": "argon",
    "[36Ar]": "argon",
    "[40Ar]": "argon",
}

LIMITS = {"n_steps": (1, 10000), "sampling_steps": (1, 50)}
DEFAULTS = {"n_steps": 20, "sampling_steps": 10, "seed": 0, "device": "cpu"}
GATE_LIMITS = {"max_temp_rel_err": 0.15, "min_pair_distance_A": 2.5}
CAVEATS = [
    "model-only rollout: no thermostat, no force field",
    "trained on LJ argon only; not valid for other molecules",
    "short rollouts say little about long-time stability (see docs/STABILITY.md)",
]


def lookup_smiles(smiles):
    """Map an OCSR SMILES string to a supported REGISTRY system, or return None."""
    if not isinstance(smiles, str):
        return None
    return SMILES_TO_SYSTEM.get(smiles.strip())


def query_from_ocsr(record, **overrides):
    """Build a surrogate query dict from an ouroboros-ocsr row or molecule_context record."""
    if not isinstance(record, dict):
        raise ValueError("OCSR record must be a dict")
    smiles = (record.get("smiles") or record.get("pred") or record.get("prediction") or "").strip()
    out = {"schema": QUERY_SCHEMA, "smiles": smiles}
    out.update(overrides)
    return out


def validate_query(query):
    """Return a list of error strings. Empty list means the query is valid."""
    if not isinstance(query, dict):
        return ["query must be a JSON object"]
    errors = []
    allowed = {"schema", "system", "smiles", "n_steps", "sampling_steps", "seed", "device", "question"}
    extra = sorted(set(query) - allowed)
    if extra:
        errors.append(f"unknown keys: {extra}")
    schema = query.get("schema")
    if schema is not None and schema != QUERY_SCHEMA:
        errors.append(f"schema must be '{QUERY_SCHEMA}'")
    has_system = "system" in query
    has_smiles = "smiles" in query
    system = query.get("system")
    smiles = query.get("smiles")
    if not has_system and not has_smiles:
        errors.append("system or smiles must be provided")
    if has_system and (not isinstance(system, str) or not system.strip()):
        errors.append("system must be a non-empty string")
    if has_smiles and (not isinstance(smiles, str) or not smiles.strip()):
        errors.append("smiles must be a non-empty string")
    if (
        has_system and isinstance(system, str) and system.strip()
        and has_smiles and isinstance(smiles, str) and smiles.strip()
    ):
        mapped = lookup_smiles(smiles)
        sys_key = system.strip().lower()
        if sys_key in REGISTRY and mapped != sys_key:
            errors.append(f"system '{system}' and smiles '{smiles}' do not match")
    for key, (lo, hi) in LIMITS.items():
        val = query.get(key, DEFAULTS[key])
        if isinstance(val, bool) or not isinstance(val, int) or not lo <= val <= hi:
            errors.append(f"{key} must be an int in [{lo}, {hi}]")
    seed = query.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int):
        errors.append("seed must be an int")
    if query.get("device", "cpu") not in ("cpu", "cuda"):
        errors.append("device must be 'cpu' or 'cuda'")
    return errors


def plan_query(query):
    """Describe what run_query would do. Standard library only; never loads torch."""
    errors = validate_query(query)
    if errors:
        return {"schema": REPORT_SCHEMA, "status": "invalid", "errors": errors}
    q = {**DEFAULTS, **query}
    smiles = q.get("smiles").strip() if isinstance(q.get("smiles"), str) else None
    raw_system = q.get("system").strip() if isinstance(q.get("system"), str) else None

    if raw_system is not None:
        sys_key = raw_system.lower()
        entry = REGISTRY.get(sys_key)
        if entry is None:
            rep = {
                "schema": REPORT_SCHEMA,
                "status": "unsupported",
                "system": raw_system,
                "supported": sorted(REGISTRY),
                "reason": "no trained checkpoint for this system",
            }
            if smiles is not None:
                rep["smiles"] = smiles
            return rep
    else:
        mapped = lookup_smiles(smiles)
        if mapped is None:
            return {
                "schema": REPORT_SCHEMA,
                "status": "unsupported",
                "system": smiles,
                "smiles": smiles,
                "supported": sorted(REGISTRY),
                "supported_smiles": sorted(SMILES_TO_SYSTEM),
                "reason": "no trained checkpoint for this SMILES",
            }
        sys_key = mapped
        entry = REGISTRY[sys_key]

    ckpt = ROOT / entry["checkpoint"]
    planned = {
        "schema": REPORT_SCHEMA,
        "status": "planned",
        "system": sys_key,
        "checkpoint": entry["checkpoint"],
        "checkpoint_present": ckpt.exists(),
        "n_atoms": entry["n_atoms"],
        "n_steps": q["n_steps"],
        "sampling_steps": q["sampling_steps"],
        "seed": q["seed"],
        "device": q["device"],
        "stride_fs": entry["stride_fs"],
        "simulated_ps": q["n_steps"] * entry["stride_fs"] / 1000.0,
    }
    if smiles is not None:
        planned["smiles"] = smiles
    return planned


@contextlib.contextmanager
def _float32_default():
    """Release checkpoints are float32. Guard against a caller that left float64 as the default."""
    import torch

    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def load_release(path, device="cpu"):
    """Load a wignerflow-v0 release checkpoint (same format as scripts/demo.py)."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import torch
    from equitraj.layers import check_attention_version
    from equitraj.model import make_model

    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([slice])
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if ck.get("format") != "wignerflow-v0":
        raise ValueError(f"{path} is not a WignerFlow v0 release checkpoint")
    check_attention_version(ck, source=path)
    with _float32_default():
        model = make_model(n_types=len(ck["elements"]), k=ck["k"], stride_fs=ck["stride_fs"], **ck["model_cfg"])
        model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck


def _clean_float(val):
    if val is None:
        return None
    f = float(val)
    return f if math.isfinite(f) else None


def _build_gate_verdict(crashed_at_step, frames_used, n_steps, t_rel_err, min_dist):
    no_crash = crashed_at_step is None and frames_used == n_steps
    temp_ok = t_rel_err is not None and abs(t_rel_err) <= GATE_LIMITS["max_temp_rel_err"]
    sterics_ok = min_dist is not None and min_dist >= GATE_LIMITS["min_pair_distance_A"]
    return {
        "no_crash": bool(no_crash),
        "temperature_ok": bool(temp_ok),
        "sterics_ok": bool(sterics_ok),
        "passed": bool(no_crash and temp_ok and sterics_ok),
    }


def _make_predictor(model):
    from equitraj.api import FrameSplitPredictor

    try:
        return FrameSplitPredictor(model)
    except ValueError:
        return model


def _format_rollout_report(plan, ck, positions_i, velocities_i, crash_i, crash_reason_i, wall_s, batch_size=1):
    from equitraj.rollout import min_pair_distance, temperature

    req_steps = int(plan["n_steps"])
    crashed = None if (crash_i < 0 or crash_i > req_steps) else int(crash_i)
    reason = crash_reason_i if crashed is not None else None
    n_ok = min(req_steps, len(positions_i)) if crashed is None else max(crashed - 1, 0)
    x = positions_i[:n_ok]
    v = velocities_i[:n_ok]
    ref = ck["reference"]
    demo = ck["demo"]
    t_ref = float(ref["T_K"])
    t_mean = None
    t_rel = None
    d_min = None
    if n_ok > 0:
        t_raw = _clean_float(temperature(v, ck["masses"].cpu()).mean().item())
        d_boxes = None if demo["box"] is None else demo["box"].cpu()[None].expand(n_ok, 3, 3)
        d_raw = _clean_float(min_pair_distance(x.double(), d_boxes).min().item())
        if t_raw is not None:
            t_mean = t_raw
            t_rel = (t_mean - t_ref) / t_ref
        if d_raw is not None:
            d_min = d_raw
    gate = _build_gate_verdict(crashed, n_ok, req_steps, t_rel, d_min)
    return {
        **plan,
        "status": "ok",
        "train_step": ck["train_step"],
        "batch_size": int(batch_size),
        "crashed_at_step": crashed,
        "crash_reason": reason,
        "frames_used": n_ok,
        "wall_s": float(wall_s),
        "T_ref_K": t_ref,
        "T_model_mean_K": t_mean,
        "T_rel_err": t_rel,
        "min_pair_distance_A": d_min,
        "gate_verdict": gate,
        "caveats": list(CAVEATS),
    }


def run_query(query, model_and_ckpt=None):
    """Run a model-only rollout from the bundled start frames and return a JSON report."""
    plan = plan_query(query)
    if plan["status"] != "planned":
        return plan
    if model_and_ckpt is None and not plan["checkpoint_present"]:
        return {**plan, "status": "error", "errors": ["checkpoint file missing"]}
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import torch
    from equitraj.rollout import rollout

    if plan["device"] == "cuda" and not torch.cuda.is_available():
        return {**plan, "status": "error", "errors": ["CUDA requested but not available"]}

    if model_and_ckpt is None:
        model, ck = load_release(ROOT / plan["checkpoint"], plan["device"])
    else:
        model, ck = model_and_ckpt
        model = model.to(plan["device"]).eval()
    demo, ref = ck["demo"], ck["reference"]
    box = None if demo["box"] is None else demo["box"][None]
    gen = torch.Generator(device=plan["device"]).manual_seed(plan["seed"])
    pred = _make_predictor(model)
    with _float32_default():
        out = rollout(
            model, demo["pos"][None], demo["vel"][None], ck["types"], box, ck["masses"],
            plan["n_steps"], sampling_steps=plan["sampling_steps"], check_every=1,
            t_max=3 * ref["T_K"], predictor=pred, generator=gen,
        )
    crash = int(out["crash_step"][0])
    reason = out["diagnostics"]["crash_reason"][0] if crash >= 0 else None
    pos_0 = out["positions"][:, 0] if out["positions"] is not None else demo["pos"][:0]
    vel_0 = out["velocities"][:, 0] if out["velocities"] is not None else demo["vel"][:0]
    return _format_rollout_report(plan, ck, pos_0, vel_0, crash, reason, out["wall_s"], batch_size=1)


class _MultiSeedPredictor:
    """Wraps a predictor so each item in a batch draws from its own seeded Generator."""

    def __init__(self, base_predictor, model, generators):
        self.base = base_predictor
        self.model = model
        self.generators = generators

    def predict(self, pos, vel, types, box, masses, noise=None, n_steps=2):
        import torch

        if noise is None and self.model.head_type == "flow":
            parts = [
                self.model.com_free_noise(1, types, masses, generator=g, like=vel)
                for g in self.generators
            ]
            noise = torch.cat(parts, dim=0)
        return self.base.predict(pos, vel, types, box, masses, noise=noise, n_steps=n_steps)


def run_batch_queries(queries, model_and_ckpt=None):
    """Pack valid queries into batched pad-free rollouts and return one report per query."""
    if not isinstance(queries, (list, tuple)):
        raise ValueError("queries must be a list of query dicts")
    if not queries:
        return []

    results = [None] * len(queries)
    groups = {}
    for idx, q in enumerate(queries):
        plan = plan_query(q)
        if plan["status"] != "planned":
            results[idx] = plan
            continue
        if model_and_ckpt is None and not plan["checkpoint_present"]:
            results[idx] = {**plan, "status": "error", "errors": ["checkpoint file missing"]}
            continue
        key = (plan["checkpoint"], plan["device"], plan["sampling_steps"])
        groups.setdefault(key, []).append((idx, plan))

    if not groups:
        return results

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import torch
    from equitraj.rollout import rollout

    ckpt_cache = {}
    for (ckpt_rel, dev, samp_steps), items in groups.items():
        if dev == "cuda" and not torch.cuda.is_available():
            for idx, plan in items:
                results[idx] = {**plan, "status": "error", "errors": ["CUDA requested but not available"]}
            continue
        if model_and_ckpt is not None:
            model, ck = model_and_ckpt
            model = model.to(dev).eval()
        else:
            if (ckpt_rel, dev) not in ckpt_cache:
                ckpt_cache[(ckpt_rel, dev)] = load_release(ROOT / ckpt_rel, dev)
            model, ck = ckpt_cache[(ckpt_rel, dev)]

        b = len(items)
        max_steps = max(int(p["n_steps"]) for _, p in items)
        demo, ref = ck["demo"], ck["reference"]
        pos_b = demo["pos"][None].expand(b, *demo["pos"].shape).clone()
        vel_b = demo["vel"][None].expand(b, *demo["vel"].shape).clone()
        box_b = None if demo["box"] is None else demo["box"][None].expand(b, 3, 3).clone()
        gens = [torch.Generator(device=dev).manual_seed(int(p["seed"])) for _, p in items]
        pred = _MultiSeedPredictor(_make_predictor(model), model, gens)

        t0 = time.perf_counter()
        with _float32_default():
            out = rollout(
                model, pos_b, vel_b, ck["types"], box_b, ck["masses"],
                max_steps, sampling_steps=samp_steps, check_every=1,
                t_max=3 * ref["T_K"], predictor=pred,
            )
        wall_s = time.perf_counter() - t0
        for batch_i, (orig_idx, plan) in enumerate(items):
            crash_i = int(out["crash_step"][batch_i])
            reason_i = out["diagnostics"]["crash_reason"][batch_i] if crash_i >= 0 else None
            pos_i = out["positions"][:, batch_i] if out["positions"] is not None else demo["pos"][:0]
            vel_i = out["velocities"][:, batch_i] if out["velocities"] is not None else demo["vel"][:0]
            results[orig_idx] = _format_rollout_report(
                plan, ck, pos_i, vel_i, crash_i, reason_i, wall_s, batch_size=b
            )
    return results


def to_reasoner_context(report):
    """Short plain-text lines the reasoner can paste into its molecule context."""
    status = report.get("status")
    if status == "unsupported":
        return [f"[physics] no surrogate for '{report['system']}'; supported: {', '.join(report['supported'])}."]
    if status in ("invalid", "error"):
        return [f"[physics] query {status}: {'; '.join(report.get('errors', []))}"]
    if status == "planned":
        return [f"[physics] planned: {report['system']}, {report['n_steps']} steps "
                f"({report['simulated_ps']:g} ps), not run yet."]
    lines = [f"[physics] {report['system']}: {report['frames_used']} of {report['n_steps']} steps "
             f"({report['simulated_ps']:g} ps requested), checkpoint step {report['train_step']}."]
    if report["crashed_at_step"] is not None:
        lines.append(f"[physics] rollout crashed at step {report['crashed_at_step']} ({report['crash_reason']}).")
    if report["T_model_mean_K"] is not None:
        lines.append(f"[physics] mean T {report['T_model_mean_K']:.1f} K vs reference {report['T_ref_K']:.1f} K "
                     f"({100 * report['T_rel_err']:+.1f}%); closest pair {report['min_pair_distance_A']:.2f} A.")
    lines.append("[physics] caveats: " + "; ".join(report["caveats"]) + ".")
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query", required=True, help="path to a query JSON file (dict or list of dicts)")
    ap.add_argument("--dry-run", action="store_true", help="validate and plan only; do not load torch")
    ap.add_argument("--context", action="store_true", help="print reasoner context lines, not JSON")
    args = ap.parse_args(argv)
    payload = json.loads(Path(args.query).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        reports = [plan_query(q) for q in payload] if args.dry_run else run_batch_queries(payload)
        if args.context:
            print("\n\n".join("\n".join(to_reasoner_context(r)) for r in reports))
        else:
            print(json.dumps(reports, indent=2, allow_nan=False))
        ok = all(r["status"] in ("planned", "ok", "unsupported") for r in reports)
        return 0 if ok else 1
    report = plan_query(payload) if args.dry_run else run_query(payload)
    print("\n".join(to_reasoner_context(report)) if args.context else json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["status"] in ("planned", "ok", "unsupported") else 1


if __name__ == "__main__":
    sys.exit(main())
