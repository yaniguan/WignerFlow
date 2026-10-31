"""Verdict: does the new checkpoint beat v0 on all four physical metrics of an eval_multiseed report?

    python scripts/compare_v0.py --metrics <eval>/metrics.json --out <eval>/v0_vs_v1.json

The first checkpoint in the report is v0, the second the candidate (scripts/eval_multiseed.py order).
Lower is better for temperature_rel_err, rdf_l1, vacf_zero_rel_err, D_rel_err. Also reports the crash
fraction, which must not be worse than v0's. Prints a Markdown table and writes JSON.
"""
import argparse
import json
from pathlib import Path

METRICS = ("temperature_rel_err", "rdf_l1", "vacf_zero_rel_err", "D_rel_err")


def compare(report):
    v0, v1 = report["checkpoints"][0], report["checkpoints"][1]
    rows, all_better = [], True
    for name in METRICS + ("crash_fraction",):
        a, b = v0["metrics"][name]["mean"], v1["metrics"][name]["mean"]
        if name == "crash_fraction":
            ok = a is not None and b is not None and b <= a
        else:
            ok = a is not None and b is not None and b < a
            all_better &= ok
        rows.append({"metric": name, "v0": a, "v1": b, "v0_ci95": v0["metrics"][name]["ci95"],
                     "v1_ci95": v1["metrics"][name]["ci95"], "v1_n_valid": v1["metrics"][name]["n_valid"],
                     "v1_better" if name != "crash_fraction" else "v1_not_worse": ok})
    crash_ok = rows[-1]["v1_not_worse"]
    return {"v0": v0["checkpoint"], "v1": v1["checkpoint"], "v1_train_step": v1.get("train_step"),
            "rows": rows, "v1_beats_v0_on_all_four": bool(all_better and crash_ok)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metrics", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    res = compare(json.loads(args.metrics.read_text()))
    args.out.write_text(json.dumps(res, indent=2) + "\n")
    fmt = lambda v: "null" if v is None else f"{v:.4g}"
    print("| metric | v0 | v1 | v1 better |\n|---|---:|---:|---|")
    for r in res["rows"]:
        print(f"| {r['metric']} | {fmt(r['v0'])} | {fmt(r['v1'])} | {r.get('v1_better', r.get('v1_not_worse'))} |")
    print("v1 beats v0 on all four:", res["v1_beats_v0_on_all_four"])


if __name__ == "__main__":
    main()
