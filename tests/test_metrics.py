"""Unit checks for the metric implementations against properties stated in the paper text.
Replicates: Adler et al., arXiv:2510.16060v2. Run: venvs/stats/bin/python tests/test_metrics.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np  # noqa: E402
from scipy.stats import norm  # noqa: E402

import metrics as M  # noqa: E402

rng = np.random.default_rng(0)
W, L = 4000, 64
y = rng.standard_normal((W, L))
z = norm.ppf(M.QUANTILES)


def gauss(scale):
    return np.broadcast_to(z * scale, (W, L, 9)).copy()


def test_pce_bounds():
    # [Paper §3.1] 'PCE is lower-bounded by 0 and upper-bounded by 0.5 for the case where predicted quantiles are
    # always all above or below the observed value'.  Upper bound: mean(1-q) = mean(q) = 0.5 for Q = 0.1..0.9.
    assert abs(M.pce(y, np.full((W, L, 9), 1e6)) - 0.5) < 1e-12
    assert abs(M.pce(y, np.full((W, L, 9), -1e6)) - 0.5) < 1e-12
    assert M.pce(y, gauss(1.0)) < 0.005  # calibrated Gaussian -> ~0 (pooled)


def test_cce_sign():
    # [Paper §3.1] positive CCE = overconfident (too narrow), negative = under-confident (too wide)
    assert abs(M.cce(y, gauss(1.0))) < 0.005
    assert M.cce(y, gauss(0.5)) > 0.1
    assert M.cce(y, gauss(2.0)) < -0.1


def test_siw_monotone():
    # [Paper §3.1] lower SIW = tighter intervals
    ref = y.ravel()
    assert M.siw(y, gauss(0.5), ref) < M.siw(y, gauss(1.0), ref) < M.siw(y, gauss(2.0), ref)
    assert abs(M.siw(y, gauss(1.0), ref) - 1.0) < 0.02  # calibrated widths match empirical spread


def test_per_window_bias():
    # documents assumption A1: literal per-window Eq. (1) is biased upward even for a perfect forecaster
    assert M.pce(y, gauss(1.0), "per_window") > 0.03


def test_mase_naive_is_one_ish():
    # [Paper Eq. (4)] median = last value repeated for a random walk -> MASE ~ O(1) and > 0
    rw = np.cumsum(rng.standard_normal((W, L + 1)), axis=1)
    q = np.repeat(rw[:, :1, None], L, axis=1).repeat(9, axis=2)
    assert 1.0 < M.mase(rw[:, 1:], q) < 15.0


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            print("ok", k)
