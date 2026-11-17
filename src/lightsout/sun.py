"""Sunrise/sunset lookup table and day/night labelling.

The table has one row per calendar date (about 2,500 rows), is computed on the driver with
``astral`` and broadcast-joined, so executors never need the library.

Conventions
-----------
* Times are NYC wall clock, stored without a zone (see spark.py).
* "day d"   = [sunrise_d, sunset_d)
* "night d" = [sunset_d, sunrise_{d+1}); a 01:30 crash belongs to the previous date's night.
"""
from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import pandas as pd
from astral import Observer
from astral.sun import sun
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


def sun_table_pandas(start: dt.date, end: dt.date, lat: float, lon: float, tz: str) -> pd.DataFrame:
    """One row per date in [start, end] with wall-clock sunrise/sunset and period lengths in hours."""
    zone = ZoneInfo(tz)
    obs = Observer(latitude=lat, longitude=lon)
    rows = []
    # one extra day so every date knows the following sunrise
    for i in range((end - start).days + 2):
        d = start + dt.timedelta(days=i)
        s = sun(obs, date=d, tzinfo=zone)
        rows.append(
            {
                "date": d,
                "sunrise": s["sunrise"].replace(tzinfo=None, microsecond=0),
                "sunset": s["sunset"].replace(tzinfo=None, microsecond=0),
                # true elapsed time, unaffected by the clock change on DST nights
                "_sunrise_abs": s["sunrise"],
                "_sunset_abs": s["sunset"],
            }
        )
    df = pd.DataFrame(rows)
    df["next_sunrise"] = df["sunrise"].shift(-1)
    df["day_hours"] = (df["_sunset_abs"] - df["_sunrise_abs"]).dt.total_seconds() / 3600.0
    df["night_hours"] = (df["_sunrise_abs"].shift(-1) - df["_sunset_abs"]).dt.total_seconds() / 3600.0
    return df.iloc[:-1].drop(columns=["_sunrise_abs", "_sunset_abs"]).reset_index(drop=True)


def sun_table(spark: SparkSession, start: dt.date, end: dt.date, nyc: dict) -> DataFrame:
    pdf = sun_table_pandas(start, end, nyc["lat"], nyc["lon"], nyc["tz"])
    # ship the times as text and parse them in Spark, so no time-zone conversion can interfere
    for c in ("sunrise", "sunset", "next_sunrise"):
        pdf[c] = pdf[c].dt.strftime("%Y-%m-%d %H:%M:%S")
    pdf["date"] = pdf["date"].astype(str)
    return spark.createDataFrame(pdf).select(
        F.to_date("date").alias("date"),
        F.to_timestamp("sunrise").alias("sunrise"),
        F.to_timestamp("sunset").alias("sunset"),
        F.to_timestamp("next_sunrise").alias("next_sunrise"),
        "day_hours",
        "night_hours",
    )


def label_period(df: DataFrame, sun_df: DataFrame, ts_col: str, buffer_min: int = 30) -> DataFrame:
    """Add ``period`` ('day'/'night'), ``period_date`` and ``twilight`` for the timestamp ``ts_col``.

    ``period_date`` is the date of the day or night the timestamp falls in (the evening's
    date for nights). ``twilight`` marks times within ``buffer_min`` of sunrise or sunset.
    """
    s = F.broadcast(sun_df.select("date", "sunrise", "sunset"))
    ts = F.col(ts_col)
    out = df.withColumn("_d", F.to_date(ts)).join(s, F.col("_d") == F.col("date"), "left")
    before_sunrise = ts < F.col("sunrise")
    is_night = before_sunrise | (ts >= F.col("sunset"))
    secs = buffer_min * 60
    near = F.least(
        F.abs(F.unix_timestamp(ts) - F.unix_timestamp("sunrise")),
        F.abs(F.unix_timestamp(ts) - F.unix_timestamp("sunset")),
    )
    return (
        out.withColumn("period", F.when(is_night, "night").otherwise("day"))
        .withColumn("period_date", F.when(before_sunrise, F.date_sub("_d", 1)).otherwise(F.col("_d")))
        .withColumn("twilight", near < secs)
        .drop("_d", "date", "sunrise", "sunset")
    )


def period_table(sun_df: DataFrame) -> DataFrame:
    """Long form: one row per (date, period) with its start, end and length in hours."""
    day = sun_df.select(
        "date", F.lit("day").alias("period"), F.col("sunrise").alias("p_start"),
        F.col("sunset").alias("p_end"), F.col("day_hours").alias("hours"),
    )
    night = sun_df.select(
        "date", F.lit("night").alias("period"), F.col("sunset").alias("p_start"),
        F.col("next_sunrise").alias("p_end"), F.col("night_hours").alias("hours"),
    )
    return day.unionByName(night)
