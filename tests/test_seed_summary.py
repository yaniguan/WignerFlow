import sys

import pytest

from equitraj.analysis import wilson_interval


def test_wilson_interval_known_values():
    lo, hi = wilson_interval(4, 5)
    assert lo == pytest.approx(0.376, abs=0.01) and hi == pytest.approx(0.964, abs=0.01)
    assert wilson_interval(0, 0) == (0.0, 1.0)
    lo, hi = wilson_interval(5, 5)
    assert hi == 1.0 and lo < 1.0  # a perfect record still has a lower bound


def test_summarize_counts_seeds_not_rollouts():
    sys.path.insert(0, "scripts")
    from summarize_seeds import summarize  # noqa: PLC0415

    good = {"stability": {"crashed": 0}, "n_rollouts": 15, "v0_verdict": {"all_pass": True},
            "temperature": {"rel_err": 0.02}, "rdf": {"peak_pos_err_A": 0.02, "peak_height_rel_err": 0.01},
            "vacf": {"first_zero_rel_err": 0.01}, "diffusion": {"rel_err": 0.05}}
    bad = {"stability": {"crashed": 15}, "n_rollouts": 15}
    s = summarize([good, good, bad])
    assert s["stable_seeds"] == 2 and s["n_seeds"] == 3 and s["all_pass_seeds"] == 2
    assert s["rollouts_crashed"] == 15 and s["rollouts_total"] == 45
    assert s["metrics"]["T error"]["n"] == 2  # the crashed seed has no temperature metric
