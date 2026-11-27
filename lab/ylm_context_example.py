"""Example wiring from WignerFlow `lab.surrogate_query --context` into YLM `build_prompt`.

Runs `lab.surrogate_query` and passes the `[physics]` lines to `YLM`'s
`lab.protocol_qa.build_prompt(item, molecule_block, physics_lines)`.
If `YLM` is on `PYTHONPATH` or passed via `--ylm-root`, its `build_prompt` is imported
directly; otherwise the matching 18-line prompt builder is used so this example and CI
tests work standalone.

    python -m lab.ylm_context_example
    python -m lab.ylm_context_example --ylm-root ../YLM
"""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab import surrogate_query as sq  # noqa: E402

FIX = ROOT / "lab" / "fixtures"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _fallback_build_prompt(item, molecule_block="", physics_lines=()):
    """Exact match to YLM `lab/protocol_qa.py:build_prompt`."""
    parts = []
    if molecule_block:
        parts.append(molecule_block.rstrip())
    if physics_lines:
        parts.append("\n".join(physics_lines))
    parts.append(f"Question:\n{item['question']}")
    if item["task"] == "mcq":
        parts.append("Choices:\n" + "\n".join(f"({LETTERS[i]}) {c}" for i, c in enumerate(item["choices"])))
        parts.append("Answer with one letter.")
    elif item["task"] == "order":
        parts.append("Answer with the step ids in the correct order, comma-separated.")
    else:
        parts.append("Answer with a short exact value.")
    return "\n\n".join(parts) + "\n"


def _load_build_prompt_from(qa_path):
    spec = importlib.util.spec_from_file_location("ylm_protocol_qa", qa_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_prompt


def resolve_build_prompt(ylm_root=None):
    """Load `build_prompt` from YLM `lab/protocol_qa.py` if present, else use the matching fallback."""
    if ylm_root is not None:
        qa_path = Path(ylm_root) / "lab" / "protocol_qa.py"
        if not qa_path.is_file():
            raise FileNotFoundError(f"YLM protocol_qa.py not found at {qa_path}")
        return _load_build_prompt_from(qa_path)
    for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        if entry.strip():
            cand = Path(entry.strip()) / "lab" / "protocol_qa.py"
            if cand.is_file():
                return _load_build_prompt_from(cand)
    return _fallback_build_prompt


def run_cli_context(query_path, dry_run=False):
    """Run `lab.surrogate_query` in-process on `query_path` and return `[physics]` lines."""
    query = json.loads(Path(query_path).read_text(encoding="utf-8"))
    report = sq.plan_query(query) if dry_run else sq.run_query(query)
    if report.get("status") not in ("planned", "ok", "unsupported"):
        raise RuntimeError(f"surrogate_query failed ({report.get('status')}): {report.get('errors')}")
    return sq.to_reasoner_context(report)


def build_example_prompt(query_path=None, smiles=None, source_id="fig1_ar", dry_run=False, ylm_root=None):
    """Build a complete YLM protocol QA prompt enriched with OCSR and WignerFlow `[physics]` context."""
    q_path = Path(query_path) if query_path is not None else (FIX / "ocsr_argon_query.json")
    raw_query = json.loads(q_path.read_text(encoding="utf-8"))
    label_smiles = smiles or raw_query.get("smiles") or raw_query.get("system") or "[Ar]"
    physics_lines = run_cli_context(q_path, dry_run=dry_run)
    molecule_block = f"Molecules (from OCSR; check flagged ones):\n- M1 ({source_id}): {label_smiles}\n"
    question_text = raw_query.get("question") or (
        f"Before handling M1 ({label_smiles}), what does the short WignerFlow check show?"
    )
    item = {
        "id": "cryo_ar_step_1",
        "task": "mcq",
        "question": question_text,
        "choices": [
            "Argon stays liquid near 95 K with no atom overlap in the short rollout",
            "Argon forms covalent bonds at 95 K",
            "No trained surrogate exists for this species",
        ],
        "answer": "A",
    }
    build_prompt = resolve_build_prompt(ylm_root=ylm_root)
    return build_prompt(item, molecule_block=molecule_block, physics_lines=physics_lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query", default=str(FIX / "ocsr_argon_query.json"), help="query JSON file")
    ap.add_argument("--dry-run", action="store_true", help="skip torch rollout and emit planned context")
    ap.add_argument("--ylm-root", default=None, help="optional path to a local YLM checkout")
    args = ap.parse_args(argv)
    print(build_example_prompt(query_path=args.query, dry_run=args.dry_run, ylm_root=args.ylm_root), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
