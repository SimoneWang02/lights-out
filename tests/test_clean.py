from lightsout import clean


def test_collisions_export_format(spark, cfg):
    raw = spark.createDataFrame(
        [
            ("09/11/2021", "2:39", "BROOKLYN", "11208", "40.667202", "-73.8665", "1", "0", "1", "0", "a1"),
            ("09/11/2021", "2:39", "BROOKLYN", "11208", "40.667202", "-73.8665", "1", "0", "1", "0", "a1"),  # duplicate
            ("12/14/2021", "17:05", None, None, None, None, "0", "0", "0", "0", "a2"),  # no coordinates
            ("12/14/2021", "8:00", None, None, "0", "0", "2", "0", "0", "0", "a3"),  # (0, 0) coordinates
        ],
        "`CRASH DATE` string, `CRASH TIME` string, BOROUGH string, `ZIP CODE` string, LATITUDE string, "
        "LONGITUDE string, `NUMBER OF PERSONS INJURED` string, `NUMBER OF PERSONS KILLED` string, "
        "`NUMBER OF PEDESTRIANS INJURED` string, `NUMBER OF PEDESTRIANS KILLED` string, COLLISION_ID string",
    )
    for c in ["ON STREET NAME", "CROSS STREET NAME", "OFF STREET NAME", "NUMBER OF CYCLIST INJURED",
              "NUMBER OF CYCLIST KILLED", "NUMBER OF MOTORIST INJURED", "NUMBER OF MOTORIST KILLED",
              "CONTRIBUTING FACTOR VEHICLE 1", "VEHICLE TYPE CODE 1"]:
        raw = raw.withColumn(c, raw["BOROUGH"])
    rows = {r["collision_id"]: r for r in clean.clean_collisions(cfg, raw).collect()}
    assert set(rows) == {"a1", "a2", "a3"}
    assert str(rows["a1"]["crash_ts"]) == "2021-09-11 02:39:00"
    assert rows["a1"]["is_ped"] and rows["a1"]["coord_ok"]
    assert not rows["a2"]["coord_ok"] and not rows["a3"]["coord_ok"]
    assert str(rows["a2"]["crash_ts"]) == "2021-12-14 17:05:00"
    assert not rows["a3"]["is_ped"]


COMPLAINT_COLS = [
    "unique_key", "created_date", "closed_date", "complaint_type", "descriptor", "descriptor_2", "status",
    "address_type", "borough", "incident_zip", "incident_address", "street_name", "intersection_street_1",
    "intersection_street_2", "latitude", "longitude",
]


def _complaint(key, created, closed, descriptor, ctype="Street Light Condition", lat="40.74", lon="-73.97"):
    return (key, created, closed, ctype, descriptor, None, "Closed", "INTERSECTION", "MANHATTAN", "10016",
            None, None, "EAST 30 STREET", "2 AVENUE", lat, lon)


def test_complaints_classification_and_intervals(spark, cfg):
    raw = spark.createDataFrame(
        [
            _complaint("1", "2022-03-01T20:00:00.000", "2022-03-05T09:00:00.000", "Street Light Out"),
            _complaint("2", "2022-12-27T10:50:00.000", "2022-12-26T10:49:00.000", "Street Light Out"),
            _complaint("3", "2022-03-01T20:00:00.000", None, "Multiple Street Lights Out"),
            _complaint("4", "2022-03-01T20:00:00.000", "2022-03-02T20:00:00.000", "Street Light Dayburning"),
            _complaint("5", "2022-03-01T20:00:00.000", "2022-03-02T20:00:00.000", "Traffic Signal Light"),
            _complaint("6", "2022-03-01T20:00:00.000", "2022-03-02T20:00:00.000", "Noise", ctype="Noise"),
            _complaint("7", "2022-03-01T20:00:00.000", "2022-03-02T20:00:00.000", "Street Light Out", lat=None),
        ],
        ", ".join(f"{c} string" for c in COMPLAINT_COLS),
    )
    rows = {r["unique_key"]: r for r in clean.clean_complaints(cfg, raw).collect()}
    assert set(rows) == {"1", "2", "3", "4", "5", "7"}  # the non-streetlight row is dropped
    assert (rows["1"]["kind"], rows["1"]["interval_quality"]) == ("dark", "valid")
    assert abs(rows["1"]["duration_days"] - (3 + 13 / 24)) < 1e-6
    assert rows["2"]["interval_quality"] == "negative"
    assert rows["3"]["interval_quality"] == "open"
    assert rows["4"]["kind"] == "placebo"
    assert rows["5"]["kind"] == "other"
    assert not rows["7"]["coord_ok"]


def test_complaints_portal_export_format(spark, cfg):
    raw = spark.createDataFrame(
        [("9", "03/01/2022 08:00:00 PM", "03/05/2022 09:00:00 AM", "Street Light Condition", "Street Light Out",
          None, "Closed", "INTERSECTION", "MANHATTAN", "10016", None, None, "A", "B", "40.74", "-73.97")],
        "`Unique Key` string, `Created Date` string, `Closed Date` string, "
        "`Problem (formerly Complaint Type)` string, `Problem Detail (formerly Descriptor)` string, "
        "`Additional Details` string, Status string, `Address Type` string, Borough string, `Incident Zip` string, "
        "`Incident Address` string, `Street Name` string, `Intersection Street 1` string, "
        "`Intersection Street 2` string, Latitude string, Longitude string",
    )
    r = clean.clean_complaints(cfg, raw).first()
    assert str(r["created_ts"]) == "2022-03-01 20:00:00"
    assert str(r["closed_ts"]) == "2022-03-05 09:00:00"
    assert r["kind"] == "dark"
