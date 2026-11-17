"""Planted-effect test: build a synthetic city where outages multiply the night crash rate by a
known factor, add the confounders the design is meant to remove, and check the estimator
recovers the factor."""
import datetime as dt

import h3
import numpy as np
import pandas as pd
import pytest

from lightsout import effect, sun

NYC = {"lat": 40.7128, "lon": -74.0060, "tz": "America/New_York"}
TRUE_RR = 1.5


def _synthetic(spark, true_rr, seed=1):
    rng = np.random.default_rng(seed)
    start, end = dt.date(2022, 1, 1), dt.date(2023, 12, 31)
    sun_df = sun.sun_table(spark, start, end, NYC)
    periods = sun.period_table(sun_df)
    p = periods.toPandas()
    p["month"] = pd.to_datetime(p["date"]).dt.month

    cells = sorted(h3.grid_disk(h3.latlng_to_cell(40.74, -73.97, 10), 12))  # 469 cells
    base = rng.gamma(shape=1.0, scale=1.0, size=len(cells))  # busy and quiet blocks
    grid = p.merge(pd.DataFrame({"cell": cells, "base": base}), how="cross")

    # Confounding by place: busy cells get more outages.
    # Confounding by season: outages are far more common in winter, when nights are long.
    winter = grid["month"].isin([11, 12, 1, 2])
    p_out = np.clip(0.04 + 0.08 * grid["base"] / (1 + grid["base"]) + 0.15 * winter, 0, 0.9)
    # an outage covers the day and the night of a date together
    key = grid["cell"] + grid["date"].astype(str)
    u = pd.Series(rng.random(key.nunique()), index=key.unique())
    grid["outage"] = (key.map(u).to_numpy() < p_out.to_numpy()).astype(int)

    night = grid["period"] == "night"
    # winter nights contain the evening rush hour: higher crash rate per night hour
    season = np.where(night & winter, 1.6, 1.0)
    # anything that raises crashes round the clock while a complaint is open (e.g. roadworks)
    busy_while_open = np.where(grid["outage"] == 1, 1.2, 1.0)
    lam = 0.012 * grid["base"] * grid["hours"] * season * busy_while_open
    lam = lam * np.where(night & (grid["outage"] == 1), true_rr, 1.0)
    grid["n"] = rng.poisson(lam)

    crashes = grid.loc[grid.index.repeat(grid["n"]), ["cell", "period", "date"]].rename(
        columns={"date": "period_date"})
    crashes["period_date"] = crashes["period_date"].astype(str)
    exposure = grid.loc[grid["outage"] == 1, ["cell", "date", "period"]].copy()
    exposure["date"] = exposure["date"].astype(str)

    c = spark.createDataFrame(crashes).selectExpr("cell", "period", "to_date(period_date) as period_date")
    e = spark.createDataFrame(exposure).selectExpr("cell", "to_date(date) as date", "period")
    return c, e, periods, grid


@pytest.mark.parametrize("true_rr", [TRUE_RR, 1.0])
def test_recovers_planted_effect(spark, true_rr):
    c, e, periods, grid = _synthetic(spark, true_rr)
    s = effect.strata(c, e, periods, "cell").toPandas()
    r = effect.estimate(s, n_boot=500)

    # every crash is accounted for
    assert int(s["a"].sum() + s["b"].sum()) == int(grid["n"].sum())
    # day ratio picks up only the round-the-clock factor (1.2); the night ratio has both
    assert r["rr_day"] == pytest.approx(1.2, abs=0.06)
    assert r["rr_night"] == pytest.approx(1.2 * true_rr, abs=0.09)
    # the headline ratio isolates the lighting effect
    assert r["ratio"] == pytest.approx(true_rr, abs=0.08)
    assert r["ratio_lo"] < true_rr < r["ratio_hi"]

    # a naive comparison that ignores place and season is badly off, which is why we stratify
    g = grid[grid["period"] == "night"]
    naive = (g.loc[g.outage == 1, "n"].sum() / g.loc[g.outage == 1, "hours"].sum()) / (
        g.loc[g.outage == 0, "n"].sum() / g.loc[g.outage == 0, "hours"].sum())
    assert naive > 1.2 * true_rr * 1.15


def test_mh_by_hand():
    # one stratum: 10 of 100 hours exposed, 3 crashes exposed, 9 unexposed -> RR = (3/10)/(9/90) = 3
    s = pd.DataFrame({"cell": [h3.latlng_to_cell(40.74, -73.97, 10)], "ym": ["2022-01"], "period": ["night"],
                      "a": [3], "b": [9], "t1": [10.0], "t": [100.0]})
    t = effect.mh_terms(s)
    assert t["num"].iloc[0] / t["den"].iloc[0] == pytest.approx(3.0)
