"""Checkpoint selection score: crashes dominate, temperature error is penalised."""

from equitraj.train import selection_score


def test_crash_dominates():
    a = {"val_crash_frac": 0.25, "val_rdf_l1": 0.1, "val_T_rel_err": 0.0}
    b = {"val_crash_frac": 0.0, "val_rdf_l1": 2.0, "val_T_rel_err": 1.0}
    assert selection_score(a) > selection_score(b)


def test_hot_checkpoint_loses_with_equal_rdf():
    good = {"val_crash_frac": 0.0, "val_rdf_l1": 0.2, "val_T_rel_err": 0.03}
    hot = {"val_crash_frac": 0.0, "val_rdf_l1": 0.18, "val_T_rel_err": 0.7}
    assert selection_score(good) < selection_score(hot)
    assert selection_score(good, t_weight=0.0) > selection_score(hot, t_weight=0.0)  # the old score prefers the hot one


def test_old_logs_without_temperature_still_score():
    assert selection_score({"val_crash_frac": 0.0, "val_rdf_l1": 0.5}) == 0.5
