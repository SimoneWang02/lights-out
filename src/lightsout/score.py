"""Stage 6: a priority score for streetlight complaints, and a test of it against arrival order.

Score
-----
For a complaint in cell c, count nighttime pedestrian crashes in the cell's footprint over
the training period (up to ``train_end``). Most footprints have zero or one such crash, so
the raw count is noisy. We shrink it toward the borough average with a gamma-Poisson
(empirical Bayes) model:

    prior for a cell's rate ~ Gamma(shape=alpha_b, mean=m_b)   fitted per borough b
    score = m_b * (y + alpha_b) / (m_b + alpha_b)              posterior mean count

A footprint with many crashes keeps a score near its own count; one with little evidence is
pulled toward what is typical for its borough. The score is reported per year of nights.

Multiplying by the outage effect from stage 5 would not change the ranking (it is one
citywide number), so the score is the baseline risk alone.

Test
----
Complaints created after ``train_end`` are ranked by score. The outcome is the number of
nighttime pedestrian crashes in the footprint during the first ``label_nights`` nights after
the report, a window that does not depend on when the city actually closed the ticket.
We compare the share of those crashes captured by the top k% of complaints:

* by score;
* by arrival order: within each calendar week, the earliest k% of reports (what a
  first-come-first-served queue would reach first);
* by a perfect ranking (upper bound).
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import Window
from pyspark.sql import functions as F

from . import geo, sun
from .config import Config
from .exposure import geo_defs
from .spark import read_table


def footprint_counts(cells: DataFrame, crashes: DataFrame, ring: int, h3_mode: str = "udf") -> DataFrame:
    """(cell, y): crashes in each cell's footprint. ``crashes`` is pre-filtered."""
    r = geo.ring_table(cells, "cell", ring, h3_mode)
    per_cell = crashes.groupBy(F.col("cell").alias("footprint_cell")).agg(F.count("*").alias("n"))
    return r.join(per_cell, "footprint_cell", "left").groupBy("cell").agg(
        F.coalesce(F.sum("n"), F.lit(0)).alias("y")
    )


def fit_prior(y: pd.Series) -> tuple[float, float]:
    """Method-of-moments gamma prior for Poisson counts: returns (mean, shape)."""
    m, v = float(y.mean()), float(y.var(ddof=1)) if len(y) > 1 else 0.0
    if m <= 0:
        return 0.0, 1.0
    excess = v - m
    alpha = m * m / excess if excess > 1e-9 else 1e6  # no over-dispersion: shrink fully to the mean
    return m, float(alpha)


def shrink(y: np.ndarray, m: float, alpha: float) -> np.ndarray:
    if m <= 0:
        return np.zeros_like(y, dtype=float)
    return m * (y + alpha) / (m + alpha)


def cell_scores(cfg: Config, spark: SparkSession, h3_mode: str = "udf") -> pd.DataFrame:
    """Score every cell that has ever hosted a dark complaint."""
    a = cfg.analysis
    train_end = dt.date.fromisoformat(a["train_end"])
    start = dt.date.fromisoformat(a["start_date"])
    complaints = read_table(cfg, spark, "gold", "complaints").filter("kind = 'dark'")
    crashes = read_table(cfg, spark, "gold", "crashes").filter(
        F.col("is_ped") & (F.col("period") == "night") & (F.col("period_date") <= F.lit(train_end))
    )
    ring = geo_defs(cfg)["r10k1"][1]

    # borough of a cell = the most common borough among its complaints
    w = Window.partitionBy("cell").orderBy(F.desc("n"), "borough")
    boro = (
        complaints.groupBy("cell", "borough").agg(F.count("*").alias("n"))
        .withColumn("_r", F.row_number().over(w)).filter("_r = 1").select("cell", "borough")
    )
    pdf = footprint_counts(complaints, crashes, ring, h3_mode).join(boro, "cell").toPandas()

    years = ((train_end - start).days + 1) / 365.25
    known = {"MANHATTAN", "BROOKLYN", "QUEENS", "BRONX", "STATEN ISLAND"}
    pdf["prior_group"] = pdf["borough"].where(pdf["borough"].isin(known), "CITYWIDE")
    city = fit_prior(pdf["y"])
    out = []
    for g, part in pdf.groupby("prior_group"):
        m, alpha = city if g == "CITYWIDE" else fit_prior(part["y"])
        part = part.assign(prior_mean=m, prior_shape=alpha, score=shrink(part["y"].to_numpy(), m, alpha) / years)
        out.append(part)
    res = pd.concat(out, ignore_index=True)
    res["raw_rate"] = res["y"] / years
    return res[["cell", "borough", "y", "raw_rate", "prior_mean", "prior_shape", "score"]]


def capture_curve(order_value: np.ndarray, outcome: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Share of all outcome events found in the top q fraction when sorted by ``order_value`` descending."""
    idx = np.argsort(-order_value, kind="stable")
    cum = np.cumsum(outcome[idx])
    total = cum[-1] if len(cum) and cum[-1] > 0 else 1.0
    k = np.clip((grid * len(idx)).round().astype(int), 0, len(idx))
    return np.where(k > 0, cum[np.maximum(k - 1, 0)] / total, 0.0)


def holdout_complaints(cfg: Config, spark: SparkSession, h3_mode: str = "udf") -> DataFrame:
    """Dark complaints after train_end with the outcome: night pedestrian crashes in the footprint
    during the first ``label_nights`` nights."""
    a = cfg.analysis
    n = a["label_nights"]
    complaints = read_table(cfg, spark, "gold", "complaints").filter("kind = 'dark'")
    sun_df = read_table(cfg, spark, "gold", "sun")
    periods = read_table(cfg, spark, "gold", "periods")
    data_end = periods.agg(F.max("date")).first()[0]
    crashes = read_table(cfg, spark, "gold", "crashes").filter("is_ped and period = 'night'")

    c = sun.label_period(complaints, sun_df, "created_ts").withColumnRenamed("period_date", "night0")
    c = c.filter(
        (F.to_date("created_ts") > F.lit(a["train_end"])) & (F.date_add("night0", n - 1) <= F.lit(data_end))
    ).select("unique_key", "cell", "created_ts", "night0", "borough", "descriptor")

    ring = geo.ring_table(c, "cell", geo_defs(cfg)["r10k1"][1], h3_mode)
    cr = crashes.groupBy(F.col("cell").alias("footprint_cell"), "period_date").agg(F.count("*").alias("n"))
    hits = (
        c.join(ring, "cell").join(cr, "footprint_cell")
        .filter(F.col("period_date").between(F.col("night0"), F.date_add("night0", n - 1)))
        .groupBy("unique_key").agg(F.sum("n").alias("crashes"))
    )
    return c.join(hits, "unique_key", "left").fillna(0, ["crashes"])


def evaluate(scores: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Capture curves for score, arrival order and a perfect ranking, plus summary numbers."""
    t = test.merge(scores[["cell", "score"]], on="cell", how="left")
    t["score"] = t["score"].fillna(scores["score"].median())
    y = t["crashes"].to_numpy(dtype=float)

    # arrival order: position within the calendar week, earliest first
    week = t["created_ts"].dt.to_period("W")
    pos = t.groupby(week)["created_ts"].rank(method="first", pct=True)
    grid = np.linspace(0, 1, 101)
    curve = pd.DataFrame(
        {
            "top_share": grid,
            "score": capture_curve(t["score"].to_numpy(), y, grid),
            "arrival": capture_curve(-pos.to_numpy(), y, grid),
            "perfect": capture_curve(y, y, grid),
        }
    )
    at = lambda col, q: float(curve.loc[(curve.top_share - q).abs().idxmin(), col])  # noqa: E731
    summary = {
        "test_complaints": int(len(t)),
        "crashes_in_window": int(y.sum()),
        "complaints_with_crash": int((y > 0).sum()),
        "score_top10": at("score", 0.10), "arrival_top10": at("arrival", 0.10),
        "score_top20": at("score", 0.20), "arrival_top20": at("arrival", 0.20),
        "score_top50": at("score", 0.50), "arrival_top50": at("arrival", 0.50),
    }
    return curve, summary


def open_complaints_ranked(cfg: Config, spark: SparkSession, scores: pd.DataFrame, max_age_days: int = 365) -> pd.DataFrame:
    """Complaints still open at the end of the data, most urgent first."""
    c = read_table(cfg, spark, "gold", "complaints").filter("kind = 'dark' and interval_quality = 'open'")
    end = read_table(cfg, spark, "gold", "periods").agg(F.max("date")).first()[0]
    pdf = (
        c.withColumn("days_open", F.datediff(F.lit(end), F.to_date("created_ts")))
        .filter(F.col("days_open") <= max_age_days)
        .select("unique_key", "created_ts", "days_open", "descriptor", "borough", "incident_address",
                "intersection_street_1", "intersection_street_2", "lat", "lon", "cell")
        .toPandas()
    )
    pdf = pdf.merge(scores[["cell", "score", "y"]], on="cell", how="left")
    pdf["score"] = pdf["score"].fillna(scores["score"].median())
    pdf = pdf.sort_values(["score", "days_open"], ascending=[False, False]).reset_index(drop=True)
    pdf["rank_by_score"] = np.arange(1, len(pdf) + 1)
    pdf["rank_by_arrival"] = pdf["created_ts"].rank(method="first").astype(int)
    return pdf
