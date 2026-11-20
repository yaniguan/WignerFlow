"""Fast offline tests for lab/ (surrogate query, OCSR/YLM wiring + distillation plan)."""

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab import distill_edge, surrogate_query as sq, ylm_context_example as ylm_ex  # noqa: E402

FIX = ROOT / "lab" / "fixtures"
CKPT = ROOT / sq.REGISTRY["argon"]["checkpoint"]


def _load(name):
    return json.loads((FIX / name).read_text())


# ---- standard-library paths (no torch) ----

def test_validate_rejects_bad_queries():
    assert sq.validate_query("argon") == ["query must be a JSON object"]
    errs = sq.validate_query({"system": "", "n_steps": 0, "sampling_steps": 99, "seed": "x", "foo": 1})
    joined = " ".join(errs)
    for word in ("unknown keys", "system", "n_steps", "sampling_steps", "seed"):
        assert word in joined
    assert sq.validate_query({})  # neither system nor smiles
    assert sq.validate_query({"system": "argon", "smiles": "CCO"})  # system/smiles mismatch
    assert sq.validate_query({"system": "argon", "schema": "wrong/v0"})  # bad schema
    assert sq.validate_query({"system": "argon", "n_steps": True})  # bools are not ints


def test_smiles_lookup_refuses_everything_except_argon():
    assert sq.lookup_smiles("[Ar]") == "argon"
    assert sq.lookup_smiles(" [40Ar] ") == "argon"
    for unknown in ("CCO", "CC(=O)Oc1ccccc1C(=O)O", "Ar", "", None, 42):
        assert sq.lookup_smiles(unknown) is None


def test_ocsr_query_mapping_and_refusal():
    ar_q = sq.query_from_ocsr({"smiles": "[Ar]", "id": "img_1"}, n_steps=2, sampling_steps=1)
    plan_ar = sq.plan_query(ar_q)
    assert plan_ar["schema"] == sq.REPORT_SCHEMA
    assert plan_ar["status"] == "planned"
    assert plan_ar["system"] == "argon" and plan_ar["smiles"] == "[Ar]"

    unk_q = _load("ocsr_unsupported_query.json")
    plan_unk = sq.plan_query(unk_q)
    assert plan_unk["status"] == "unsupported"
    assert plan_unk["smiles"] == "CCO"
    assert sq.run_query(unk_q)["status"] == "unsupported"
    assert "no surrogate for 'CCO'" in sq.to_reasoner_context(plan_unk)[0]


def test_plan_argon_fixture():
    plan = sq.plan_query(_load("argon_query.json"))
    assert plan["status"] == "planned"
    assert plan["simulated_ps"] == pytest.approx(2 * 50.0 / 1000)
    assert plan["checkpoint_present"] == CKPT.exists()


def test_unsupported_system_is_refused_not_guessed():
    plan = sq.plan_query(_load("unsupported_query.json"))
    assert plan["status"] == "unsupported" and plan["supported"] == ["argon"]
    assert sq.run_query(_load("unsupported_query.json"))["status"] == "unsupported"
    assert "no surrogate" in sq.to_reasoner_context(plan)[0]


def test_registry_matches_checkpoint_readme():
    text = (ROOT / "checkpoints" / "README.md").read_text()
    entry = sq.REGISTRY["argon"]
    row = next(line for line in text.splitlines() if Path(entry["checkpoint"]).name in line)
    assert f"{entry['n_atoms']} atoms" in row
    assert f"stride {int(entry['stride_fs'])} fs" in row


def test_cli_dry_run(capsys):
    assert sq.main(["--query", str(FIX / "argon_query.json"), "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "planned"
    assert sq.main(["--query", str(FIX / "argon_query.json"), "--dry-run", "--context"]) == 0
    assert "not run yet" in capsys.readouterr().out


def test_distill_plan_dry_run(capsys):
    cfg = distill_edge.load_config()
    files = distill_edge.check_files(cfg)
    assert all(files.values()), files  # every referenced script exists in this repo
    assert distill_edge.main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "no results yet" in out and "scripts/distill.py" in out
    assert not re.search(r"\d+(\.\d+)?\s*(ns/day|%)", out)  # the plan carries no result numbers


def test_distill_config_rejects_duplicates(tmp_path):
    cfg = distill_edge.load_config()
    cfg["stages"].append(dict(cfg["stages"][0]))
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="unique"):
        distill_edge.load_config(bad)


# ---- real model path (torch + checkpoint; a 1-2 step rollout) ----

@pytest.fixture(scope="module")
def release():
    pytest.importorskip("torch")
    pytest.importorskip("e3nn")
    if not CKPT.exists():
        pytest.skip("release checkpoint not present")
    return sq.load_release(CKPT)


def test_checkpoint_matches_registry(release):
    _, ck = release
    entry = sq.REGISTRY["argon"]
    assert ck["elements"] == entry["elements"]
    assert len(ck["types"]) == entry["n_atoms"]
    assert float(ck["stride_fs"]) == entry["stride_fs"]


def test_run_query_one_step(release):
    # One model step with one flow step keeps this test fast on a CPU. It checks plumbing, not accuracy.
    # Other test files set float64 as the torch default; the query must still work.
    import torch

    query = {**_load("argon_query.json"), "n_steps": 1, "sampling_steps": 1}
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        report = sq.run_query(query, model_and_ckpt=release)
        assert torch.get_default_dtype() == torch.float64  # restored after the query
    finally:
        torch.set_default_dtype(previous)
    assert report["status"] == "ok"
    json.dumps(report)  # must be JSON-serializable for the reasoner
    if report["crashed_at_step"] is None:
        assert report["frames_used"] == 1
        assert report["T_model_mean_K"] > 0 and report["min_pair_distance_A"] > 0
    lines = sq.to_reasoner_context(report)
    assert lines[0].startswith("[physics] argon") and "caveats" in lines[-1]


def test_ylm_example_prompt_includes_real_physics_and_refuses_unknown(release, tmp_path):
    fast_ar = tmp_path / "fast_ar.json"
    fast_ar.write_text(json.dumps({"smiles": "[Ar]", "n_steps": 1, "sampling_steps": 1, "seed": 0}))
    prompt = ylm_ex.build_example_prompt(query_path=fast_ar, smiles="[Ar]")
    assert "Molecules (from OCSR; check flagged ones):" in prompt
    assert "[physics] argon: 1 of 1 steps" in prompt
    assert "[physics] mean T " in prompt
    assert "Question:\n" in prompt

    prompt_unk = ylm_ex.build_example_prompt(query_path=FIX / "ocsr_unsupported_query.json", smiles="CCO")
    assert "[physics] no surrogate for 'CCO'; supported: argon." in prompt_unk
