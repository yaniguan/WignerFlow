"""CLI for 1,000,000-candidate unknown Li-electrolyte chemical space screening funnel.

Example:
    python scripts/electrolyte_funnel_1m.py --calisol-csv "CALiSol-23 Dataset.csv" --space-size 1000000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from equitraj.funnel import run_1m_hierarchical_funnel
from scripts.calisol_align import train_and_evaluate_calisol


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calisol-csv", required=True, help="Path to CALiSol-23 Dataset.csv")
    ap.add_argument("--space-size", type=int, default=1_000_000)
    ap.add_argument("--top-k", type=int, default=15)
    ap.add_argument("--json-out", help="Optional output path for JSON discovery report")
    args = ap.parse_args(argv)

    cal_res = train_and_evaluate_calisol(args.calisol_csv, panel_size=12)
    aligner, post_model = cal_res.pop("_artifacts")
    funnel_res = run_1m_hierarchical_funnel(
        aligner, post_model, n_candidates=args.space_size, tier2_keep=180, tier3_keep=20, top_k=args.top_k
    )
    text = json.dumps(funnel_res, indent=2)
    if args.json_out:
        out_p = Path(args.json_out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
