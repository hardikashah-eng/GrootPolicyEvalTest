import math

import pytest

from groot_eval import stats


def test_wilson_matches_reference_values():
    lo, hi = stats.wilson_interval(0, 10)
    assert lo == 0.0
    assert hi == pytest.approx(0.2775, abs=1e-3)
    lo, hi = stats.wilson_interval(10, 10)
    assert lo == pytest.approx(0.7225, abs=1e-3)
    assert hi == 1.0
    lo, hi = stats.wilson_interval(50, 100)
    assert (lo, hi) == pytest.approx((0.4038, 0.5962), abs=1e-3)


def test_wilson_rejects_bad_counts_and_handles_zero_trials():
    with pytest.raises(ValueError):
        stats.wilson_interval(5, 3)
    assert stats.wilson_interval(0, 0) == (0.0, 1.0)


def test_interval_narrows_with_more_trials():
    w_small = (lambda lo, hi: hi - lo)(*stats.wilson_interval(8, 10))
    w_large = (lambda lo, hi: hi - lo)(*stats.wilson_interval(800, 1000))
    assert w_large < w_small / 5


def test_regression_detection_needs_significance():
    assert stats.is_regression(baseline=(90, 100), candidate=(70, 100))
    assert not stats.is_regression(baseline=(90, 100), candidate=(86, 100))
    assert not stats.is_regression(baseline=(70, 100), candidate=(90, 100))


def test_rankdata_averages_ties_and_spearman():
    assert list(stats.rankdata([10, 20, 20, 30])) == [1.0, 2.5, 2.5, 4.0]
    assert stats.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert stats.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert math.isnan(stats.spearman([1, 1, 1], [1, 2, 3]))


def test_required_trials_classic_value():
    assert stats.required_trials(0.05, 0.95, 0.5) == 385


def test_bootstrap_ci_brackets_mean():
    lo, hi = stats.bootstrap_ci([1.0, 2.0, 3.0, 4.0, 5.0] * 10)
    assert lo < 3.0 < hi
