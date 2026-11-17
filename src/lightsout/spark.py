"""SparkSession construction and table read/write helpers.

Timestamps: both source datasets record New York wall-clock time with no zone. We set the
session time zone to UTC and never convert, so every timestamp in every table is "NYC wall
clock". Sunrise and sunset are converted to the same convention in sun.py. This avoids
Spark shifting or rejecting times around daylight-saving changes.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession

from .config import Config

_LOCAL_JDK = Path.home() / ".local" / "jdk" / "current"


def get_spark(cfg: Config, app: str = "lightsout") -> SparkSession:
    if "JAVA_HOME" not in os.environ and _LOCAL_JDK.exists():
        os.environ["JAVA_HOME"] = str(_LOCAL_JDK)
    # Python must agree with the session time zone, or collect() shifts timestamps
    os.environ["TZ"] = "UTC"
    time.tzset()

    s = cfg.spark
    b = SparkSession.builder.appName(app)
    if s.get("master"):
        b = b.master(s["master"])
    if s.get("driver_memory"):
        b = b.config("spark.driver.memory", s["driver_memory"])
    b = (
        b.config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", str(s.get("shuffle_partitions", 200)))
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
        .config("spark.ui.showConsoleProgress", "false")
    )
    if not cfg.is_remote:
        b = b.config("spark.sql.warehouse.dir", cfg.path("warehouse"))
        b = b.config("spark.driver.extraJavaOptions", "-Duser.timezone=UTC")
        b = b.config("spark.ui.enabled", "false")
    if s.get("hive"):
        b = b.enableHiveSupport()
    spark = b.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    if s.get("hive"):
        spark.sql(f"CREATE DATABASE IF NOT EXISTS {s['database']}")
    return spark


def write_table(cfg: Config, df: DataFrame, layer: str, name: str, partition_by=None) -> str:
    """Write a table as Parquet; on Hive-enabled configs also register it as <db>.<layer>_<name>."""
    path = cfg.path(layer, name)
    w = df.write.mode("overwrite")
    if partition_by:
        w = w.partitionBy(*partition_by)
    if cfg.spark.get("hive"):
        w.option("path", path).saveAsTable(f"{cfg.spark['database']}.{layer}_{name}")
    else:
        w.parquet(path)
    return path


def read_table(cfg: Config, spark: SparkSession, layer: str, name: str) -> DataFrame:
    return spark.read.parquet(cfg.path(layer, name))


def write_pandas(cfg: Config, spark: SparkSession, pdf, name: str) -> str:
    """Save a small result table (pandas) to the ``out`` layer."""
    return write_table(cfg, spark.createDataFrame(pdf).coalesce(1), "out", name)


def read_pandas(cfg: Config, spark: SparkSession, name: str):
    return read_table(cfg, spark, "out", name).toPandas()
