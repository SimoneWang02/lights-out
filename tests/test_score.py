import numpy as np
import pandas as pd
import pytest

from lightsout import score


def test_prior_and_shrinkage():
    rng = np.random.default_rng(0)
    rates = rng.gamma(shape=2.0, scale=1.5, size=20000)  # true mean 3, shape 2
    y = pd.Series(rng.poisson(rates))
    m, alpha = score.fit_prior(y)
    assert m == pytest.approx(3.0, abs=0.1)
    assert alpha == pytest.approx(2.0, abs=0.25)
    s = score.shrink(np.array([0.0, 3.0, 30.0]), m, alpha)
    assert 0 < s[0] < m          # zero crashes is pulled up toward the mean
    assert s[1] == pytest.approx(m, abs=0.1)
    assert m < s[2] < 30         # a large count is pulled down, but stays high
    # shrunk estimates are closer to the true rates than raw counts are
    est = score.shrink(y.to_numpy(dtype=float), m, alpha)
    assert np.mean((est - rates) ** 2) < np.mean((y - rates) ** 2)


def test_prior_without_overdispersion_shrinks_fully():
    m, alpha = score.fit_prior(pd.Series([2, 2, 2, 2]))
    assert score.shrink(np.array([10.0]), m, alpha)[0] == pytest.approx(2.0, abs=0.01)


def test_capture_curve():
    grid = np.array([0.0, 0.25, 0.5, 1.0])
    outcome = np.array([4.0, 0.0, 3.0, 1.0])
    perfect = score.capture_curve(outcome, outcome, grid)
    assert perfect.tolist() == [0.0, 0.5, 0.875, 1.0]
    worst = score.capture_curve(-outcome, outcome, grid)
    assert worst.tolist() == [0.0, 0.0, 0.125, 1.0]


def test_evaluate_arrival_order_is_near_diagonal():
    rng = np.random.default_rng(1)
    n = 5000
    cells = [f"c{i}" for i in range(200)]
    risk = dict(zip(cells, rng.gamma(1.0, 1.0, len(cells))))
    t = pd.DataFrame({
        "unique_key": np.arange(n).astype(str),
        "cell": rng.choice(cells, n),
        "created_ts": pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.uniform(0, 300, n), unit="D"),
    })
    t["crashes"] = rng.poisson(t["cell"].map(risk) * 0.05)
    scores = pd.DataFrame({"cell": cells, "score": [risk[c] for c in cells]})
    curve, summary = score.evaluate(scores, t)
    assert summary["arrival_top20"] == pytest.approx(0.2, abs=0.06)
    assert summary["score_top20"] > 0.4
    assert (curve["perfect"] >= curve["score"] - 1e-9).all()
