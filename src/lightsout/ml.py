"""Stage 7: Spark MLlib classifier.

Question: when a dark-streetlight complaint arrives, will there be a nighttime pedestrian
crash in its footprint during the next ``label_nights`` nights?

Rows     dark complaints created at least one year after the data start (so every row has a
         full trailing year of history) and early enough for the label window to be observed
Label    1 if at least one nighttime pedestrian crash in the footprint in the label window
Features everything is measured strictly before the report date:
           trailing-365-day crash counts in the footprint (night/day x pedestrian/all)
           trailing-365-day count of other dark complaints in the same cell
           night length on the report date, month (as sine/cosine), day of week, hour
           borough, complaint descriptor, address type
Split    by time: train up to ``train_end``, test after it
Models   logistic regression and gradient-boosted trees, both weighted for class imbalance
Baseline the stage-6 priority score used directly as a ranking

The positive class is rare (around 1%), so area under the precision-recall curve and the
share of positives in the top decile say more than accuracy or ROC alone.
"""
from __future__ import annotations

import datetime as dt
import math

import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from . import geo, sun
from .config import Config
from .exposure import geo_defs
from .spark import read_table

NUMERIC = [
    "night_ped_365", "day_ped_365", "night_all_365", "day_all_365", "complaints_365",
    "night_hours", "month_sin", "month_cos", "hour", "is_weekend",
]
CATEGORICAL = ["borough", "descriptor", "address_type"]


def build_dataset(cfg: Config, spark: SparkSession, h3_mode: str = "udf") -> DataFrame:
    a = cfg.analysis
    n = a["label_nights"]
    start = dt.date.fromisoformat(a["start_date"])
    first = start + dt.timedelta(days=366)
    sun_df = read_table(cfg, spark, "gold", "sun")
    data_end = read_table(cfg, spark, "gold", "periods").agg(F.max("date")).first()[0]
    crashes = read_table(cfg, spark, "gold", "crashes")
    all_dark = read_table(cfg, spark, "gold", "complaints").filter("kind = 'dark'")

    c = sun.label_period(all_dark, sun_df, "created_ts").withColumnRenamed("period_date", "night0")
    c = c.withColumn("created_date", F.to_date("created_ts")).filter(
        (F.col("created_date") >= F.lit(first)) & (F.date_add("night0", n - 1) <= F.lit(data_end))
    )
    c = c.select(
        "unique_key", "cell", "created_ts", "created_date", "night0",
        "borough", "descriptor", "address_type",
    )

    ring = geo.ring_table(c, "cell", geo_defs(cfg)["r10k1"][1], h3_mode)
    foot = c.select("unique_key", "cell", "created_date", "night0").join(ring, "cell")

    daily = crashes.groupBy(F.col("cell").alias("footprint_cell"), "period", "period_date").agg(
        F.count("*").alias("n_all"), F.sum(F.col("is_ped").cast("int")).alias("n_ped")
    )
    j = foot.join(daily, "footprint_cell")
    hist = (F.col("period_date") < F.col("created_date")) & (
        F.col("period_date") >= F.date_sub("created_date", 365)
    )
    future = (F.col("period") == "night") & F.col("period_date").between(
        F.col("night0"), F.date_add("night0", n - 1)
    )
    night, day = F.col("period") == "night", F.col("period") == "day"
    s = lambda cond, col: F.sum(F.when(cond, F.col(col)).otherwise(0))  # noqa: E731
    agg = j.filter(hist | future).groupBy("unique_key").agg(
        s(hist & night, "n_ped").alias("night_ped_365"),
        s(hist & day, "n_ped").alias("day_ped_365"),
        s(hist & night, "n_all").alias("night_all_365"),
        s(hist & day, "n_all").alias("day_all_365"),
        s(future, "n_ped").alias("future_night_ped"),
        s(future, "n_all").alias("future_night_all"),
    )

    # earlier dark complaints in the same cell during the previous year
    prev = all_dark.select(F.col("cell").alias("p_cell"), F.to_date("created_ts").alias("p_date"))
    cmp_hist = (
        c.select("unique_key", "cell", "created_date")
        .join(prev, (F.col("cell") == F.col("p_cell"))
              & (F.col("p_date") < F.col("created_date"))
              & (F.col("p_date") >= F.date_sub("created_date", 365)))
        .groupBy("unique_key").agg(F.count("*").alias("complaints_365"))
    )

    nights = F.broadcast(sun_df.select(F.col("date").alias("created_date"), "night_hours"))
    month = F.month("created_date")
    counts = ["night_ped_365", "day_ped_365", "night_all_365", "day_all_365",
              "future_night_ped", "future_night_all", "complaints_365"]
    out = (
        c.join(agg, "unique_key", "left").join(cmp_hist, "unique_key", "left").join(nights, "created_date", "left")
        .fillna(0, counts)
        .withColumn("month_sin", F.sin(month * (2 * math.pi / 12)))
        .withColumn("month_cos", F.cos(month * (2 * math.pi / 12)))
        .withColumn("hour", F.hour("created_ts").cast("double"))
        .withColumn("is_weekend", F.dayofweek("created_date").isin(1, 7).cast("double"))
        .withColumn("label", (F.col("future_night_ped") > 0).cast("double"))
        .withColumn("label_any_crash", (F.col("future_night_all") > 0).cast("double"))
        .withColumn("is_test", F.col("created_date") > F.lit(a["train_end"]))
        .fillna("UNKNOWN", CATEGORICAL)
    )
    for col in counts:
        out = out.withColumn(col, F.col(col).cast("double"))
    return out


def _stages(scale: bool):
    idx = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in CATEGORICAL]
    ohe = OneHotEncoder(inputCols=[f"{c}_idx" for c in CATEGORICAL], outputCols=[f"{c}_vec" for c in CATEGORICAL])
    assembler = VectorAssembler(inputCols=NUMERIC + [f"{c}_vec" for c in CATEGORICAL], outputCol="raw_features")
    stages = [*idx, ohe, assembler]
    if scale:
        stages.append(StandardScaler(inputCol="raw_features", outputCol="features", withMean=False))
    return stages, ("features" if scale else "raw_features")


def _metrics(pred: DataFrame, label: str, score_col: str) -> dict:
    """ROC/PR areas plus precision and recall in the top decile by score."""
    ev = BinaryClassificationEvaluator(labelCol=label, rawPredictionCol=score_col)
    roc = ev.evaluate(pred, {ev.metricName: "areaUnderROC"})
    pr = ev.evaluate(pred, {ev.metricName: "areaUnderPR"})
    n, pos = pred.agg(F.count("*"), F.sum(label)).first()
    # exactly the top 10% of rows; ties are broken by complaint id so tied scores cannot
    # inflate the decile (the test set is small enough for a single-partition sort)
    rank = F.row_number().over(Window.orderBy(F.desc(score_col), "unique_key"))
    top = (
        pred.select(label, rank.alias("_rank")).filter(F.col("_rank") <= int(n * 0.1))
        .agg(F.count("*"), F.sum(label)).first()
    )
    base = pos / n if n else float("nan")
    prec = (top[1] or 0) / top[0] if top[0] else float("nan")
    return {
        "auc_roc": roc, "auc_pr": pr, "base_rate": base,
        "top10_precision": prec, "top10_recall": (top[1] or 0) / pos if pos else float("nan"),
        "top10_lift": prec / base if base else float("nan"),
    }


def train_and_evaluate(data: DataFrame, baseline_scores: DataFrame | None = None, label: str = "label"):
    """Fit both models on the training period, evaluate on the test period.

    ``baseline_scores``: (cell, score) from stage 6, evaluated as a ranking on the same test rows.
    Returns (metrics table, feature importances, test predictions).
    """
    train = data.filter(~F.col("is_test")).cache()
    test = data.filter("is_test").cache()
    n, pos = train.agg(F.count("*"), F.sum(label)).first()
    w_pos = (n - pos) / pos if pos else 1.0
    train = train.withColumn("weight", F.when(F.col(label) == 1, w_pos).otherwise(1.0))

    rows, preds = [], None
    models = {
        "logistic_regression": (
            True, lambda f: LogisticRegression(featuresCol=f, labelCol=label, weightCol="weight",
                                               maxIter=100, regParam=0.01)),
        "gradient_boosted_trees": (
            False, lambda f: GBTClassifier(featuresCol=f, labelCol=label, weightCol="weight", maxIter=60,
                                           maxDepth=4, stepSize=0.1, subsamplingRate=0.8, seed=7)),
    }
    importances = None
    for name, (scale, make) in models.items():
        stages, feat = _stages(scale)
        model = Pipeline(stages=[*stages, make(feat)]).fit(train)
        p = model.transform(test).withColumn("p", vector_to_array("probability")[1])
        rows.append({"model": name, **_metrics(p, label, "p")})
        keep = p.select("unique_key", F.col("p").alias(f"p_{name}"))
        preds = keep if preds is None else preds.join(keep, "unique_key")
        if name == "gradient_boosted_trees":
            attrs = p.schema[feat].metadata.get("ml_attr", {}).get("attrs", {})
            names = {a["idx"]: a["name"] for group in attrs.values() for a in group}
            imp = model.stages[-1].featureImportances.toArray()
            importances = (
                pd.DataFrame({"feature": [names.get(i, str(i)) for i in range(len(imp))], "importance": imp})
                .sort_values("importance", ascending=False).reset_index(drop=True)
            )

    if baseline_scores is not None:
        b = test.join(baseline_scores, "cell", "left").fillna(0.0, ["score"])
        rows.append({"model": "priority_score_baseline", **_metrics(b, label, "score")})
        preds = preds.join(b.select("unique_key", F.col("score").alias("p_priority_score_baseline")), "unique_key")
    # a one-feature ranking for reference: last year's night pedestrian crashes in the footprint
    rows.append({"model": "trailing_count_only", **_metrics(test, label, "night_ped_365")})

    preds = preds.join(test.select("unique_key", label), "unique_key")
    meta = {"train_rows": int(n), "train_positives": int(pos), "test_rows": test.count()}
    return pd.DataFrame(rows).assign(**meta), importances, preds
