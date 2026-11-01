"""Summarize one experiment across training seeds: how often is training successful, and how good is it when it is?

    python scripts/summarize_seeds.py --glob "results/argon/B_S3a_seed*/eval/metrics.json" --out results/argon/B_S3a_seeds

Reads the evaluate.py metrics of each seed and reports, with 95% Wilson intervals over seeds (the rollouts of one seed share
one trained model, so seeds, not rollouts, are the independent units):
  - fraction of seeds with no crashed rollout,
  - fraction of seeds that meet every v0 criterion,
  - mean ± std over the non-crashed seeds of temperature error, RDF peak error, VACF zero error and D error.
Writes <out>.json and <out>.md (a table that can go into a paper).
"""

import argparse
import glob
import json

import numpy as np

from equitraj.analysis import wilson_interval

COLUMNS = [("T error", lambda m: m.get("temperature", {}).get("rel_err"), 100, "%"),
           ("RDF peak pos (Å)", lambda m: m.get("rdf", {}).get("peak_pos_err_A"), 1, ""),
           ("RDF peak height err", lambda m: m.get("rdf", {}).get("peak_height_rel_err"), 100, "%"),
           ("VACF zero err", lambda m: m.get("vacf", {}).get("first_zero_rel_err"), 100, "%"),
           ("D err", lambda m: m.get("diffusion", {}).get("rel_err"), 100, "%")]


def summarize(metrics):
    n = len(metrics)
    stable = [m for m in metrics if m["stability"]["crashed"] == 0]
    passed = [m for m in metrics if m.get("v0_verdict", {}).get("all_pass")]
    out = {"n_seeds": n,
           "stable_seeds": len(stable), "stable_ci95": wilson_interval(len(stable), n),
           "all_pass_seeds": len(passed), "all_pass_ci95": wilson_interval(len(passed), n),
           "rollouts_crashed": sum(m["stability"]["crashed"] for m in metrics),
           "rollouts_total": sum(m["n_rollouts"] for m in metrics), "metrics": {}}
    for name, get, scale, _ in COLUMNS:
        vals = [get(m) for m in stable if get(m) is not None]
        if vals:
            out["metrics"][name] = {"mean": float(np.mean(vals)) * scale, "std": float(np.std(vals, ddof=1)) * scale if len(vals) > 1 else None,
                                    "n": len(vals)}
    return out


def to_markdown(s, runs):
    ci = lambda k, key: f"{s[k]}/{s['n_seeds']} (95% CI {s[key][0]:.0%}–{s[key][1]:.0%})"
    lines = ["| Statistic | Value |", "|---|---|",
             f"| Seeds | {s['n_seeds']} ({', '.join(runs)}) |",
             f"| Seeds with no crashed rollout | {ci('stable_seeds', 'stable_ci95')} |",
             f"| Seeds meeting every v0 criterion | {ci('all_pass_seeds', 'all_pass_ci95')} |",
             f"| Crashed rollouts (pooled, not independent) | {s['rollouts_crashed']}/{s['rollouts_total']} |"]
    for name, _, _, unit in COLUMNS:
        if name in s["metrics"]:
            v = s["metrics"][name]
            std = "" if v["std"] is None else f" ± {v['std']:.3g}"
            lines.append(f"| {name}, mean ± std over {v['n']} non-crashed seeds | {v['mean']:.3g}{std}{unit} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="metrics.json files, one per seed")
    ap.add_argument("--out", required=True, help="output prefix")
    args = ap.parse_args()
    paths = sorted(glob.glob(args.glob))
    assert paths, f"no files match {args.glob}"
    metrics = [json.load(open(p)) for p in paths]
    runs = [p.split("/")[-3] for p in paths]
    s = summarize(metrics)
    s["runs"] = runs
    with open(args.out + ".json", "w") as fh:
        json.dump(s, fh, indent=2)
    md = to_markdown(s, runs)
    with open(args.out + ".md", "w") as fh:
        fh.write(md)
    print(md)


if __name__ == "__main__":
    main()
