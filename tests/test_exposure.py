import datetime as dt

import h3
from pyspark.sql import functions as F

from lightsout import exposure, geo, sun

NYC = {"lat": 40.7128, "lon": -74.0060, "tz": "America/New_York"}
D = dt.date
END = D(2022, 12, 31)


def _periods(spark):
    return sun.period_table(sun.sun_table(spark, D(2022, 1, 1), END, NYC))


def _complaints(spark, rows):
    cell = h3.latlng_to_cell(40.74, -73.97, 10)
    data = [(k, kind, cell, h3.cell_to_parent(cell, 9), q, c, cl) for k, kind, q, c, cl in rows]
    df = spark.createDataFrame(
        data, "unique_key string, kind string, cell string, cell_coarse string, interval_quality string, "
              "created_ts string, closed_ts string")
    return df.withColumn("created_ts", F.to_timestamp("created_ts")).withColumn("closed_ts", F.to_timestamp("closed_ts"))


def _nights(df):
    return sorted(str(r["date"]) for r in df.filter("period = 'night'").collect())


def _days(df):
    return sorted(str(r["date"]) for r in df.filter("period = 'day'").collect())


def test_window_to_periods(spark, cfg):
    c = _complaints(spark, [
        # reported in the evening after dark, fixed three mornings later
        ("k1", "dark", "valid", "2022-03-01 20:00:00", "2022-03-04 09:00:00"),
    ])
    got = exposure.explode_periods(exposure.windows(c, "A", cfg, END), _periods(spark))
    # nights of 1, 2, 3 March (the last one ends at sunrise on the 4th, before the 09:00 repair)
    assert _nights(got) == ["2022-03-01", "2022-03-02", "2022-03-03"]
    # daytime of the 2nd and 3rd in full, plus the morning of the 4th; not the 1st (reported after sunset)
    assert _days(got) == ["2022-03-02", "2022-03-03", "2022-03-04"]


def test_same_day_closure_has_no_night(spark, cfg):
    c = _complaints(spark, [("k1", "dark", "valid", "2022-03-01 09:00:00", "2022-03-01 15:00:00")])
    got = exposure.explode_periods(exposure.windows(c, "A", cfg, END), _periods(spark))
    assert _nights(got) == [] and _days(got) == ["2022-03-01"]


def test_report_after_midnight_counts_the_night_in_progress(spark, cfg):
    c = _complaints(spark, [("k1", "dark", "valid", "2022-03-02 01:00:00", "2022-03-02 12:00:00")])
    got = exposure.explode_periods(exposure.windows(c, "A", cfg, END), _periods(spark))
    assert _nights(got) == ["2022-03-01"]


def test_specs(spark, cfg):
    c = _complaints(spark, [
        ("neg", "dark", "negative", "2022-03-01 10:00:00", "2022-02-28 10:00:00"),
        ("long", "dark", "valid", "2022-03-01 10:00:00", "2022-11-01 10:00:00"),
        ("open", "dark", "open", "2022-03-01 10:00:00", None),
    ])
    p = _periods(spark)

    def nights(spec):
        df = exposure.explode_periods(exposure.windows(c, spec, cfg, END), p).filter("period = 'night'")
        return {r["unique_key"]: r["count"] for r in df.groupBy("unique_key").count().collect()}

    a = nights("A")
    assert "neg" not in a  # closed before created: unusable under the closed-date spec
    assert a["long"] == cfg.analysis["cap_days"] and a["open"] == cfg.analysis["cap_days"]
    assert nights("A30") == {"long": 30, "open": 30}
    b = nights("B")
    assert b == {k: cfg.analysis["fixed_window_nights"] for k in ("neg", "long", "open")}


def test_ring_footprint(spark):
    cell = h3.latlng_to_cell(40.74, -73.97, 10)
    df = spark.createDataFrame([(cell,), (cell,)], "cell string")
    for mode in ("udf", "driver"):
        assert geo.ring_table(df, "cell", 1, mode).count() == 7
        assert geo.ring_table(df, "cell", 0, mode).count() == 1


def test_add_cells_modes_agree(spark):
    df = spark.createDataFrame([(40.74, -73.97), (40.65, -73.95)], "lat double, lon double")
    a = {r["lat"]: r["cell"] for r in geo.add_cells(df, {"cell": 10}, "udf").collect()}
    b = {r["lat"]: r["cell"] for r in geo.add_cells(df, {"cell": 10}, "driver").collect()}
    assert a == b and a[40.74] == h3.latlng_to_cell(40.74, -73.97, 10)
