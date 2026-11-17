import datetime as dt

from pyspark.sql import functions as F

from lightsout import sun

NYC = {"lat": 40.7128, "lon": -74.0060, "tz": "America/New_York"}
D = dt.date


def test_sun_table_is_plausible():
    t = sun.sun_table_pandas(D(2023, 1, 1), D(2023, 12, 31), 40.7128, -74.0060, "America/New_York")
    assert len(t) == 365
    row = t.set_index("date")
    # winter solstice: sunset around 16:30; summer solstice: around 20:30 (wall clock, DST applied)
    assert row.loc[D(2023, 12, 21), "sunset"].hour == 16
    assert row.loc[D(2023, 6, 21), "sunset"].hour == 20
    assert row.loc[D(2023, 6, 21), "sunrise"].hour == 5
    # a day and the night that follows add up to about 24 hours
    assert ((t["day_hours"] + t["night_hours"]).between(23.9, 24.1)).all()
    # the night of the autumn clock change really is an hour longer than the wall clock suggests
    nov4 = row.loc[D(2023, 11, 4)]
    wall = (nov4["next_sunrise"] - nov4["sunset"]).total_seconds() / 3600
    assert abs(nov4["night_hours"] - (wall + 1)) < 0.01


def _label(spark, stamps):
    s = sun.sun_table(spark, D(2023, 1, 1), D(2023, 12, 31), NYC)
    df = spark.createDataFrame([(x,) for x in stamps], "ts string").withColumn("ts", F.to_timestamp("ts"))
    rows = sun.label_period(df, s, "ts").collect()
    return {str(r["ts"]): (r["period"], str(r["period_date"]), r["twilight"]) for r in rows}


def test_label_period(spark):
    got = _label(
        spark,
        [
            "2023-06-21 12:00:00",  # midday
            "2023-06-21 23:30:00",  # late evening
            "2023-06-22 01:30:00",  # after midnight -> belongs to the night of the 21st
            "2023-12-21 17:30:00",  # dark in December ...
            "2023-06-21 17:30:00",  # ... light in June at the same clock time
            "2023-06-21 20:35:00",  # just after sunset -> night, and inside the twilight buffer
            "2023-03-12 03:30:00",  # morning of the spring clock change
        ],
    )
    assert got["2023-06-21 12:00:00"] == ("day", "2023-06-21", False)
    assert got["2023-06-21 23:30:00"] == ("night", "2023-06-21", False)
    assert got["2023-06-22 01:30:00"] == ("night", "2023-06-21", False)
    assert got["2023-12-21 17:30:00"][0] == "night"
    assert got["2023-06-21 17:30:00"][0] == "day"
    assert got["2023-06-21 20:35:00"] == ("night", "2023-06-21", True)
    assert got["2023-03-12 03:30:00"] == ("night", "2023-03-11", False)
