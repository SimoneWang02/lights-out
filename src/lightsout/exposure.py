"""Stages 3 and 4: analysis-ready crashes and complaints, and the outage exposure table.

Gold tables written here
------------------------
sun         one row per date: sunrise, sunset, next_sunrise, day_hours, night_hours
periods     one row per (date, period): start, end, hours
crashes     geocoded crashes since the start date with H3 cells and day/night labels
complaints  geocoded dark/placebo complaints with H3 cells
exposure    (spec, geo, kind, cell, date, period, n_complaints): the periods during which
            a cell had at least one open complaint in its footprint

Outage window specifications (``spec``)
---------------------------------------
A    created -> closed, capped at ``cap_days``; complaints whose closed date precedes the
     created date are excluded; still-open complaints run to the end of the data
A30  as A, capped at 30 days
B    a fixed ``fixed_window_nights`` days after the report, for every complaint; does not
     use the closed date at all

Spatial definitions (``geo``)
-----------------------------
r10k1  resolution-10 cell plus one ring of neighbours (7 cells), the main definition
r10k0  the resolution-10 cell only
r9k0   the resolution-9 parent cell only (about 7x the area of a resolution-10 cell)
"""
from __future__ import annotations

import datetime as dt

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from . import geo, sun
from .config import Config
from .spark import read_table, write_table

# geo name -> (cell column, ring size); "main" ring size is filled from config
GEOS = {"r10k1": ("cell", None), "r10k0": ("cell", 0), "r9k0": ("cell_coarse", 0)}
SPECS = ("A", "A30", "B")


def analysis_bounds(cfg: Config, collisions: DataFrame, complaints: DataFrame) -> tuple[dt.date, dt.date]:
    """Start date from config; end date = the last date both datasets cover."""
    start = dt.date.fromisoformat(cfg.analysis["start_date"])
    last_crash = collisions.agg(F.max("crash_date")).first()[0]
    last_complaint = complaints.agg(F.max(F.to_date("created_ts"))).first()[0]
    return start, min(last_crash, last_complaint)


def build_keys(cfg: Config, spark: SparkSession, h3_mode: str = "udf") -> None:
    a = cfg.analysis
    col = read_table(cfg, spark, "silver", "collisions")
    cmp_ = read_table(cfg, spark, "silver", "complaints")
    start, end = analysis_bounds(cfg, col, cmp_)
    print(f"analysis period: {start} to {end}")

    sun_df = sun.sun_table(spark, start - dt.timedelta(days=1), end + dt.timedelta(days=1), cfg.nyc)
    write_table(cfg, sun_df.coalesce(1), "gold", "sun")
    sun_df = read_table(cfg, spark, "gold", "sun")
    periods = sun.period_table(sun_df).filter(F.col("date").between(F.lit(start), F.lit(end)))
    write_table(cfg, periods.coalesce(1), "gold", "periods")

    res = {"cell": a["h3_resolution"], "cell_coarse": a["h3_resolution"] - 1}

    crashes = col.filter(
        F.col("coord_ok") & F.col("crash_ts").isNotNull() & F.col("crash_date").between(F.lit(start), F.lit(end))
    )
    crashes = sun.label_period(crashes, sun_df, "crash_ts", a["twilight_buffer_min"])
    crashes = geo.add_cells(crashes, res, h3_mode).filter(F.col("period_date") >= F.lit(start))
    crashes = crashes.select(
        "collision_id", "crash_ts", "crash_date", "period", "period_date", "twilight",
        "cell", "cell_coarse", "borough", "lat", "lon", "is_ped", "ped_injured", "ped_killed",
        "persons_injured", "persons_killed",
    )
    write_table(cfg, crashes, "gold", "crashes")

    complaints = cmp_.filter(
        F.col("coord_ok")
        & F.col("kind").isin("dark", "placebo")
        & F.to_date("created_ts").between(F.lit(start), F.lit(end))
    )
    complaints = geo.add_cells(complaints, res, h3_mode)
    write_table(cfg, complaints, "gold", "complaints")


def windows(complaints: DataFrame, spec: str, cfg: Config, end: dt.date) -> DataFrame:
    """Outage window [w_start, w_end) per complaint under one specification."""
    a = cfg.analysis
    data_end = F.to_timestamp(F.lit(f"{end + dt.timedelta(days=1)} 12:00:00"))
    created = F.col("created_ts")
    if spec == "B":
        df = complaints
        w_end = created + F.make_interval(days=F.lit(a["fixed_window_nights"]))
    else:
        cap = a["cap_days"] if spec == "A" else 30
        df = complaints.filter(F.col("interval_quality").isin("valid", "open"))
        w_end = F.least(F.coalesce(F.col("closed_ts"), data_end), created + F.make_interval(days=F.lit(cap)))
    return df.select(
        "unique_key", "kind", "cell", "cell_coarse", created.alias("w_start"), F.least(w_end, data_end).alias("w_end")
    ).filter(F.col("w_end") > F.col("w_start"))


def explode_periods(win: DataFrame, periods: DataFrame) -> DataFrame:
    """One row per (complaint, date, period) that the window overlaps."""
    days = win.withColumn(
        "date", F.explode(F.sequence(F.date_sub(F.to_date("w_start"), 1), F.to_date("w_end")))
    )
    p = F.broadcast(periods.select("date", "period", "p_start", "p_end"))
    return (
        days.join(p, "date")
        .filter((F.col("w_start") < F.col("p_end")) & (F.col("w_end") > F.col("p_start")))
        .drop("p_start", "p_end", "w_start", "w_end")
    )


def geo_defs(cfg: Config) -> dict[str, tuple[str, int]]:
    return {g: (c, cfg.analysis["ring"] if k is None else k) for g, (c, k) in GEOS.items()}


def build_exposure(cfg: Config, spark: SparkSession, h3_mode: str = "udf") -> None:
    complaints = read_table(cfg, spark, "gold", "complaints")
    periods = read_table(cfg, spark, "gold", "periods")
    end = periods.agg(F.max("date")).first()[0]

    rings = {
        g: geo.ring_table(complaints, col, k, h3_mode).cache() for g, (col, k) in geo_defs(cfg).items()
    }
    parts = []
    for spec in SPECS:
        cp = explode_periods(windows(complaints, spec, cfg, end), periods)
        for g, (col, _k) in geo_defs(cfg).items():
            r = rings[g]
            e = (
                cp.join(r, cp[col] == r["cell"])
                .groupBy(F.col("footprint_cell").alias("cell"), "date", "period", "kind")
                .agg(F.countDistinct("unique_key").cast("int").alias("n_complaints"))
                .withColumn("spec", F.lit(spec))
                .withColumn("geo", F.lit(g))
            )
            parts.append(e)
    out = parts[0]
    for e in parts[1:]:
        out = out.unionByName(e)
    write_table(cfg, out, "gold", "exposure", partition_by=["spec", "geo"])
