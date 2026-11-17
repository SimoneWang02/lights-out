"""Stage 2: raw CSV -> typed, de-duplicated silver tables, plus a data-quality table.

Rows with unusable coordinates are kept and flagged (``coord_ok``) rather than dropped, so
the quality table can report them and a later street-name geocoding step can recover them.
"""
from __future__ import annotations

import re

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from .config import Config

COMPLAINT_TYPE = "Street Light Condition"

# The lamp gives no (or clearly reduced) light at night.
DARK = [
    "Street Light Out",
    "Multiple Street Lights Out",
    "Street Light Lamp Missing",
    "Street Light Lamp Dim",
    "Street Light Cycling",
    "Lamppost Knocked Down",
    "Lamppost Missing",
    "Fixture/Luminaire Missing",
    "Flood Light Lamp Out",
    "Flood Light Lamp Cycling",
]
# Something is wrong with the pole, but the lamp still lights the street at night.
# These complaints should show no nighttime effect and serve as a negative control.
PLACEBO = [
    "Lamppost Base Door/Cover Open",
    "Lamppost Base Door/Cover Damaged",
    "Lamppost Base Door/Cover Missing",
    "Fixture/Luminaire Door Open",
    "Street Light Dayburning",
    "Multiple St Lts Dayburning",
    "Flood Light Lamp Dayburning",
    "Lamppost Wire Exposed",
    "Foreign Attachment On Lamppost",
]

# The portal's CSV export renamed two columns; map them back to the API names.
_ALIASES = {
    "problem_formerly_complaint_type": "complaint_type",
    "problem_detail_formerly_descriptor": "descriptor",
    "additional_details": "descriptor_2",
    # the collisions API spells these two without the underscore
    "vehicle_type_code1": "vehicle_type_code_1",
    "vehicle_type_code2": "vehicle_type_code_2",
}


def normalize_columns(df: DataFrame) -> DataFrame:
    """'CRASH DATE' -> 'crash_date', 'Problem (formerly Complaint Type)' -> 'complaint_type'."""
    out = df
    for c in df.columns:
        n = re.sub(r"[^0-9a-z]+", "_", c.strip().lower()).strip("_")
        n = _ALIASES.get(n, n)
        if n != c:
            out = out.withColumnRenamed(c, n)
    return out


def _ts(col):
    """Parse the timestamp formats used by the API (ISO) and by the portal export (US style)."""
    c = F.trim(col)
    return F.coalesce(
        F.to_timestamp(c, "yyyy-MM-dd'T'HH:mm:ss.SSS"),
        F.to_timestamp(c, "yyyy-MM-dd'T'HH:mm:ss"),
        F.to_timestamp(c, "MM/dd/yyyy hh:mm:ss a"),
    )


def _coord_ok(cfg: Config, lat, lon):
    s, w, n, e = cfg.nyc["bbox"]
    return F.coalesce(lat.between(s, n) & lon.between(w, e), F.lit(False))


def _int(c):
    return F.coalesce(F.col(c).cast("int"), F.lit(0))


def _read_csv(spark: SparkSession, path: str, multiline: bool) -> DataFrame:
    return (
        spark.read.option("header", True)
        .option("quote", '"')
        .option("escape", '"')
        .option("multiLine", multiline)
        .csv(path)
    )


def clean_collisions(cfg: Config, raw: DataFrame) -> DataFrame:
    df = normalize_columns(raw)
    date_str = F.trim(F.col("crash_date"))
    crash_date = F.coalesce(
        F.to_date(date_str, "MM/dd/yyyy"), F.to_date(F.substring(date_str, 1, 10), "yyyy-MM-dd")
    )
    hhmm = F.lpad(F.trim(F.col("crash_time")), 5, "0")
    crash_ts = F.to_timestamp(F.concat_ws(" ", F.date_format(crash_date, "yyyy-MM-dd"), hhmm), "yyyy-MM-dd HH:mm")
    lat = F.col("latitude").cast("double")
    lon = F.col("longitude").cast("double")
    out = df.select(
        F.col("collision_id").cast("string").alias("collision_id"),
        crash_ts.alias("crash_ts"),
        crash_date.alias("crash_date"),
        F.upper(F.trim("borough")).alias("borough"),
        F.col("zip_code").alias("zip_code"),
        lat.alias("lat"),
        lon.alias("lon"),
        _coord_ok(cfg, lat, lon).alias("coord_ok"),
        F.upper(F.trim("on_street_name")).alias("on_street_name"),
        F.upper(F.trim("cross_street_name")).alias("cross_street_name"),
        F.upper(F.trim("off_street_name")).alias("off_street_name"),
        _int("number_of_persons_injured").alias("persons_injured"),
        _int("number_of_persons_killed").alias("persons_killed"),
        _int("number_of_pedestrians_injured").alias("ped_injured"),
        _int("number_of_pedestrians_killed").alias("ped_killed"),
        _int("number_of_cyclist_injured").alias("cyclist_injured"),
        _int("number_of_cyclist_killed").alias("cyclist_killed"),
        _int("number_of_motorist_injured").alias("motorist_injured"),
        _int("number_of_motorist_killed").alias("motorist_killed"),
        F.col("contributing_factor_vehicle_1").alias("factor_1"),
        F.col("vehicle_type_code_1").alias("vehicle_1"),
    )
    out = out.withColumn("is_ped", (F.col("ped_injured") + F.col("ped_killed")) > 0)
    return out.filter(F.col("collision_id").isNotNull()).dropDuplicates(["collision_id"])


def clean_complaints(cfg: Config, raw: DataFrame) -> DataFrame:
    df = normalize_columns(raw).filter(F.col("complaint_type") == COMPLAINT_TYPE)
    lat = F.col("latitude").cast("double")
    lon = F.col("longitude").cast("double")
    created = _ts(F.col("created_date"))
    closed = _ts(F.col("closed_date"))
    desc = F.trim(F.col("descriptor"))
    kind = F.when(desc.isin(DARK), "dark").when(desc.isin(PLACEBO), "placebo").otherwise("other")
    quality = (
        F.when(closed.isNull(), "open").when(closed < created, "negative").otherwise("valid")
    )
    out = df.select(
        F.col("unique_key").cast("string").alias("unique_key"),
        created.alias("created_ts"),
        closed.alias("closed_ts"),
        desc.alias("descriptor"),
        kind.alias("kind"),
        quality.alias("interval_quality"),
        ((F.unix_timestamp(closed) - F.unix_timestamp(created)) / 86400.0).alias("duration_days"),
        F.col("status").alias("status"),
        F.upper(F.trim("address_type")).alias("address_type"),
        F.upper(F.trim("borough")).alias("borough"),
        F.col("incident_zip").alias("incident_zip"),
        F.upper(F.trim("incident_address")).alias("incident_address"),
        F.upper(F.trim("street_name")).alias("street_name"),
        F.upper(F.trim("intersection_street_1")).alias("intersection_street_1"),
        F.upper(F.trim("intersection_street_2")).alias("intersection_street_2"),
        F.col("descriptor_2").alias("detail"),
        lat.alias("lat"),
        lon.alias("lon"),
        _coord_ok(cfg, lat, lon).alias("coord_ok"),
    )
    return out.filter(F.col("unique_key").isNotNull() & F.col("created_ts").isNotNull()).dropDuplicates(
        ["unique_key"]
    )


def quality_collisions(cfg: Config, raw: DataFrame, clean: DataFrame) -> list[tuple[str, str, int]]:
    start = cfg.analysis["start_date"]
    since = F.col("crash_date") >= F.lit(start)
    r = clean.agg(
        F.count("*").alias("rows_clean"),
        F.sum(F.col("crash_ts").isNull().cast("int")).alias("unparsed_timestamp"),
        F.sum(since.cast("int")).alias("since_start"),
        F.sum((since & F.col("is_ped")).cast("int")).alias("since_start_pedestrian"),
        F.sum((since & ~F.col("coord_ok")).cast("int")).alias("since_start_bad_coords"),
        F.sum((since & F.col("is_ped") & F.col("coord_ok")).cast("int")).alias("since_start_pedestrian_usable"),
    ).first()
    rows = [("collisions", "rows_raw", raw.count())]
    rows += [("collisions", k, int(v or 0)) for k, v in r.asDict().items()]
    return rows


def quality_complaints(clean: DataFrame) -> list[tuple[str, str, int]]:
    dark = F.col("kind") == "dark"
    r = clean.agg(
        F.count("*").alias("rows_clean"),
        F.sum(dark.cast("int")).alias("dark"),
        F.sum((F.col("kind") == "placebo").cast("int")).alias("placebo"),
        F.sum((F.col("kind") == "other").cast("int")).alias("other"),
        F.sum((~F.col("coord_ok")).cast("int")).alias("bad_coords"),
        F.sum((dark & ~F.col("coord_ok")).cast("int")).alias("dark_bad_coords"),
        F.sum((dark & (F.col("interval_quality") == "negative")).cast("int")).alias("dark_closed_before_created"),
        F.sum((dark & (F.col("interval_quality") == "open")).cast("int")).alias("dark_no_closed_date"),
        F.sum((dark & (F.col("interval_quality") == "valid") & (F.col("duration_days") < 1)).cast("int")).alias(
            "dark_closed_within_1_day"
        ),
        F.sum((dark & (F.col("duration_days") >= 90)).cast("int")).alias("dark_open_90_days_plus"),
    ).first()
    return [("complaints", k, int(v or 0)) for k, v in r.asDict().items()]


def run(cfg: Config, spark: SparkSession) -> DataFrame:
    from .spark import write_table

    # Both sources have quoted fields with line breaks (the collisions `location` column, 311
    # free text), so both need multiline parsing. Files are one per year, so reading still
    # parallelises across files.
    raw_col = _read_csv(spark, cfg.path("raw", "collisions"), multiline=True)
    write_table(cfg, clean_collisions(cfg, raw_col), "silver", "collisions")
    col = spark.read.parquet(cfg.path("silver", "collisions"))

    raw_cmp = _read_csv(spark, cfg.path("raw", "complaints"), multiline=True)
    write_table(cfg, clean_complaints(cfg, raw_cmp), "silver", "complaints")
    cmp_ = spark.read.parquet(cfg.path("silver", "complaints"))

    rows = quality_collisions(cfg, raw_col, col) + quality_complaints(cmp_)
    q = spark.createDataFrame(rows, "dataset string, metric string, value long").coalesce(1)
    write_table(cfg, q, "out", "quality")
    return q
