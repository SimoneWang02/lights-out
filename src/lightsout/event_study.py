"""Event study: crashes in a complaint's footprint by night relative to the report date.

For every complaint, night 0 is the first night that starts at or after the report (or the
night in progress when it was filed). We count pedestrian crashes in the footprint for
relative nights -W..+W, separately for night and day, and divide by the number of complaints
whose relative night falls inside the data period.

This view does not use the closed date. What to look for:
* a night-time rise before night 0: the lamp was already out before anyone reported it;
* a decline after night 0 as repairs happen;
* a flat daytime series (the negative control).

Complaints are restricted to "index" reports: no other complaint of the same kind in the
same cell during the previous 2*W days, so follow-up reports do not smear the picture.
"Lamppost Knocked Down" is excluded because a crash is often what knocked the post down.
"""
from __future__ import annotations

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import Window
from pyspark.sql import functions as F

from . import geo, sun
from .config import Config
from .exposure import geo_defs
from .spark import read_table


def index_events(complaints: DataFrame, sun_df: DataFrame, quiet_days: int) -> DataFrame:
    """One row per index complaint with ``night0``: the date of its first night."""
    c = complaints.filter(F.col("descriptor") != "Lamppost Knocked Down")
    c = sun.label_period(c, sun_df, "created_ts").withColumnRenamed("period_date", "night0")
    w = Window.partitionBy("kind", "cell").orderBy("created_ts")
    gap = F.datediff("created_ts", F.lag("created_ts").over(w))
    return (
        c.withColumn("_gap", gap)
        .filter(F.col("_gap").isNull() | (F.col("_gap") > quiet_days))
        .select("unique_key", "kind", "cell", "night0")
    )


def run(cfg: Config, spark: SparkSession, h3_mode: str = "udf") -> pd.DataFrame:
    w = cfg.analysis["event_window"]
    complaints = read_table(cfg, spark, "gold", "complaints")
    crashes = read_table(cfg, spark, "gold", "crashes").filter("is_ped")
    sun_df = read_table(cfg, spark, "gold", "sun")
    periods = read_table(cfg, spark, "gold", "periods")
    lo, hi = periods.agg(F.min("date"), F.max("date")).first()

    ev = index_events(complaints, sun_df, quiet_days=2 * w).cache()
    ring = geo.ring_table(ev, "cell", geo_defs(cfg)["r10k1"][1], h3_mode)
    foot = ev.join(ring, "cell").select("unique_key", "kind", "night0", "footprint_cell")

    cr = crashes.groupBy(F.col("cell").alias("footprint_cell"), "period", "period_date").agg(
        F.count("*").alias("n")
    )
    counts = (
        foot.join(cr, "footprint_cell")
        .withColumn("rel", F.datediff("period_date", "night0"))
        .filter(F.col("rel").between(-w, w))
        .groupBy("kind", "period", "rel")
        .agg(F.sum("n").alias("crashes"))
    )

    # complaints at risk for each relative night: night0 + rel must lie inside the data period
    rels = spark.range(-w, w + 1).select(F.col("id").cast("int").alias("rel"))
    at_risk = (
        ev.crossJoin(F.broadcast(rels))
        .filter(F.date_add("night0", F.col("rel")).between(F.lit(lo), F.lit(hi)))
        .groupBy("kind", "rel")
        .agg(F.count("*").alias("complaints"))
    )
    out = (
        at_risk.crossJoin(spark.createDataFrame([("night",), ("day",)], "period string"))
        .join(counts, ["kind", "period", "rel"], "left")
        .fillna(0, ["crashes"])
        .withColumn("per_1000", F.col("crashes") / F.col("complaints") * 1000)
        .orderBy("kind", "period", "rel")
    )
    return out.toPandas()
