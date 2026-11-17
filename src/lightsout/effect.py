"""Stage 5: how much does crash risk change while a streetlight complaint is open?

Design
------
Each (cell, year-month) is a stratum and acts as its own control. Within a stratum we split
the night hours into "outage" (a complaint was open in the cell's footprint) and "no outage"
and compare crash rates per hour. Strata are pooled with the Mantel-Haenszel rate ratio:

    RR = sum_i a_i * T0_i / T_i  /  sum_i b_i * T1_i / T_i

    a_i, b_i   crashes during outage / non-outage hours in stratum i
    T1_i, T0_i outage / non-outage hours, T_i = T1_i + T0_i

Stratifying by month matters: winter nights are longer and swallow the evening rush hour,
and outages are reported more often in winter, so a pooled comparison would be confounded.

The same ratio is computed for daytime crashes on days the complaint was open. A broken lamp
cannot matter in daylight, so RR_day measures whatever else differs while a complaint is open
(more activity on the block, roadworks, reporting behaviour). The headline estimate is

    ratio = RR_night / RR_day

Uncertainty: cluster bootstrap. Cells are grouped by their resolution-8 parent (about
0.7 km2) and whole groups are resampled, so neighbouring cells that share complaints stay
together.
"""
from __future__ import annotations

from dataclasses import dataclass

import h3
import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from .config import Config
from .exposure import geo_defs
from .spark import read_table

# SQL filters on the crashes table (strings, so importing this module needs no Spark session)
OUTCOMES = {
    "pedestrian": "is_ped",
    "all_crashes": "true",
    "any_injury": "persons_injured + persons_killed > 0",
}


@dataclass(frozen=True)
class Run:
    label: str
    spec: str = "A"
    geo: str = "r10k1"
    kind: str = "dark"
    outcome: str = "pedestrian"
    drop_twilight: bool = False


GRID = [
    Run("Main: closed-date window, 7-cell footprint"),
    Run("Placebo complaints (lamp still works)", kind="placebo"),
    Run("Fixed 7-day window", spec="B"),
    Run("Placebo, fixed 7-day window", spec="B", kind="placebo"),
    Run("Closed-date window capped at 30 days", spec="A30"),
    Run("Same cell only (no neighbours)", geo="r10k0"),
    Run("Coarser cell (resolution 9)", geo="r9k0"),
    Run("Excluding 30 min around sunrise/sunset", drop_twilight=True),
    Run("Outcome: any injury crash", outcome="any_injury"),
    Run("Outcome: all crashes", outcome="all_crashes"),
]


def strata(crashes: DataFrame, exposure: DataFrame, periods: DataFrame, cell_col: str) -> DataFrame:
    """Per (cell, ym, period): crashes in/out of outage and outage/total hours.

    ``crashes`` is already filtered to the outcome of interest; ``exposure`` to one
    (spec, geo, kind). Only strata with at least one crash are returned: the others add
    nothing to a Mantel-Haenszel estimate.
    """
    ym = lambda c: F.date_format(c, "yyyy-MM")  # noqa: E731
    e = exposure.select("cell", "date", "period").withColumn("exposed", F.lit(1))

    counts = (
        crashes.groupBy(F.col(cell_col).alias("cell"), F.col("period_date").alias("date"), "period")
        .agg(F.count("*").alias("n"))
        .join(e, ["cell", "date", "period"], "left")
        .withColumn("exposed", F.coalesce("exposed", F.lit(0)))
        .groupBy("cell", ym("date").alias("ym"), "period")
        .agg(
            F.sum(F.col("n") * F.col("exposed")).alias("a"),
            F.sum(F.col("n") * (1 - F.col("exposed"))).alias("b"),
        )
    )
    p = F.broadcast(periods.select("date", "period", "hours"))
    t1 = (
        e.join(p, ["date", "period"])
        .groupBy("cell", ym("date").alias("ym"), "period")
        .agg(F.sum("hours").alias("t1"))
    )
    t = periods.groupBy(ym("date").alias("ym"), "period").agg(F.sum("hours").alias("t"))
    return (
        counts.join(t1, ["cell", "ym", "period"], "left")
        .join(F.broadcast(t), ["ym", "period"])
        .withColumn("t1", F.coalesce("t1", F.lit(0.0)))
    )


def mh_terms(s: pd.DataFrame) -> pd.DataFrame:
    """Add each stratum's contribution to the MH numerator and denominator."""
    s = s.copy()
    s["num"] = s["a"] * (s["t"] - s["t1"]) / s["t"]
    s["den"] = s["b"] * s["t1"] / s["t"]
    return s


def _rr(num: float, den: float) -> float:
    return float(num / den) if den > 0 else float("nan")


def estimate(s: pd.DataFrame, n_boot: int = 2000, seed: int = 7, cluster_res: int = 8) -> dict:
    """Mantel-Haenszel rate ratios for night and day, their ratio, and bootstrap 95% intervals."""
    s = mh_terms(s)
    s["cluster"] = s["cell"].map(lambda c: h3.cell_to_parent(c, cluster_res))
    g = (
        s.pivot_table(index="cluster", columns="period", values=["num", "den", "a", "b"], aggfunc="sum")
        .reindex(columns=pd.MultiIndex.from_product([["num", "den", "a", "b"], ["night", "day"]]))
        .fillna(0.0)
    )
    nn, dn = g[("num", "night")].to_numpy(), g[("den", "night")].to_numpy()
    nd, dd = g[("num", "day")].to_numpy(), g[("den", "day")].to_numpy()

    rr_n, rr_d = _rr(nn.sum(), dn.sum()), _rr(nd.sum(), dd.sum())
    out = {
        "rr_night": rr_n,
        "rr_day": rr_d,
        "ratio": rr_n / rr_d if rr_d and rr_d > 0 else float("nan"),
        "night_crashes_outage": int(g[("a", "night")].sum()),
        "night_crashes_no_outage": int(g[("b", "night")].sum()),
        "day_crashes_outage": int(g[("a", "day")].sum()),
        "day_crashes_no_outage": int(g[("b", "day")].sum()),
        "clusters": int(len(g)),
    }

    rng = np.random.default_rng(seed)
    k = len(g)
    w = rng.multinomial(k, np.full(k, 1.0 / k), size=n_boot).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        b_n = (w @ nn) / (w @ dn)
        b_d = (w @ nd) / (w @ dd)
        b_r = b_n / b_d
    for name, b in (("rr_night", b_n), ("rr_day", b_d), ("ratio", b_r)):
        b = b[np.isfinite(b)]
        lo, hi = (np.percentile(b, [2.5, 97.5]) if len(b) else (np.nan, np.nan))
        out[f"{name}_lo"], out[f"{name}_hi"] = float(lo), float(hi)
    return out


def run_one(cfg: Config, spark: SparkSession, run: Run, crashes=None, exposure=None, periods=None) -> dict:
    crashes = crashes if crashes is not None else read_table(cfg, spark, "gold", "crashes")
    exposure = exposure if exposure is not None else read_table(cfg, spark, "gold", "exposure")
    periods = periods if periods is not None else read_table(cfg, spark, "gold", "periods")

    c = crashes.filter(F.expr(OUTCOMES[run.outcome]))
    if run.drop_twilight:
        c = c.filter(~F.col("twilight"))
    e = exposure.filter((F.col("spec") == run.spec) & (F.col("geo") == run.geo) & (F.col("kind") == run.kind))
    cell_col = geo_defs(cfg)[run.geo][0]
    s = strata(c, e, periods, cell_col).toPandas()
    res = estimate(s)
    return {
        "label": run.label, "spec": run.spec, "geo": run.geo, "kind": run.kind,
        "outcome": run.outcome, "drop_twilight": run.drop_twilight, **res,
    }


def run_grid(cfg: Config, spark: SparkSession, grid=GRID) -> pd.DataFrame:
    crashes = read_table(cfg, spark, "gold", "crashes").cache()
    exposure = read_table(cfg, spark, "gold", "exposure")
    periods = read_table(cfg, spark, "gold", "periods").cache()
    rows = []
    for run in grid:
        r = run_one(cfg, spark, run, crashes, exposure, periods)
        print(
            f"{run.label:<45} night {r['rr_night']:.3f}  day {r['rr_day']:.3f}  "
            f"ratio {r['ratio']:.3f} [{r['ratio_lo']:.3f}, {r['ratio_hi']:.3f}]  "
            f"(night crashes during outage: {r['night_crashes_outage']:,})"
        )
        rows.append(r)
    return pd.DataFrame(rows)
