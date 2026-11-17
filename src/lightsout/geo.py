"""H3 spatial keys.

Two ways to compute cells, chosen by ``mode``:

* ``udf``    - pandas UDF; needs the ``h3`` package on every executor (fine locally, needs a
               packed environment on a cluster).
* ``driver`` - collect the distinct coordinates, compute cells on the driver, join back.
               Only the driver needs ``h3``. Slower, but works on any cluster.
"""
from __future__ import annotations

import h3
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType


def _cells(lat: pd.Series, lon: pd.Series, res: int) -> pd.Series:
    return pd.Series([h3.latlng_to_cell(a, b, res) for a, b in zip(lat, lon)], dtype="object")


def _cell_udf(res: int):
    @F.pandas_udf(StringType())
    def to_cell(lat: pd.Series, lon: pd.Series) -> pd.Series:
        return _cells(lat, lon, res)

    return to_cell


def add_cells(df: DataFrame, resolutions: dict[str, int], mode: str = "udf") -> DataFrame:
    """Add one H3 cell column per entry of ``resolutions`` ({column_name: resolution}).

    Rows must have valid ``lat``/``lon``.
    """
    if mode == "udf":
        out = df
        for name, res in resolutions.items():
            out = out.withColumn(name, _cell_udf(res)("lat", "lon"))
        return out

    spark = df.sparkSession
    pts = df.select("lat", "lon").distinct().toPandas()
    for name, res in resolutions.items():
        pts[name] = _cells(pts["lat"], pts["lon"], res)
    return df.join(spark.createDataFrame(pts), ["lat", "lon"], "left")


def ring_table(cells: DataFrame, cell_col: str, k: int, mode: str = "udf") -> DataFrame:
    """Distinct cells -> (cell, footprint_cell) pairs: the cell itself plus k rings of neighbours."""
    distinct = cells.select(F.col(cell_col).alias("cell")).distinct()
    if k == 0:
        return distinct.withColumn("footprint_cell", F.col("cell"))
    if mode == "udf":
        @F.pandas_udf(ArrayType(StringType()))
        def disk(c: pd.Series) -> pd.Series:
            return c.map(lambda x: list(h3.grid_disk(x, k)))

        return distinct.select("cell", F.explode(disk("cell")).alias("footprint_cell"))

    pdf = distinct.toPandas()
    pdf["footprint_cell"] = pdf["cell"].map(lambda x: list(h3.grid_disk(x, k)))
    return cells.sparkSession.createDataFrame(pdf.explode("footprint_cell"))


def parent(cell: str, res: int) -> str:
    return h3.cell_to_parent(cell, res)


def cell_center(cell: str) -> tuple[float, float]:
    return h3.cell_to_latlng(cell)


def cell_boundary(cell: str) -> list[tuple[float, float]]:
    """Hexagon outline as (lat, lon) pairs."""
    return list(h3.cell_to_boundary(cell))
