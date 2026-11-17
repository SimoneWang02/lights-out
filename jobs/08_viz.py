"""Stage 8: report figures, their table twins, and the interactive map -> <repo>/reports/."""
import pandas as pd
from pyspark.sql import functions as F

from lightsout import viz
from lightsout.config import REPO_ROOT, job_args
from lightsout.spark import get_spark, read_pandas, read_table


def _maybe(cfg, spark, name):
    try:
        return read_pandas(cfg, spark, name)
    except Exception:  # stage not run yet
        print(f"skipping: out/{name} not found")
        return None


if __name__ == "__main__":
    cfg, _ = job_args(__doc__)
    spark = get_spark(cfg, "lightsout-viz")
    fig = REPO_ROOT / "reports" / "figures"
    tab = REPO_ROOT / "reports" / "tables"

    # 1. hour x month heatmap of pedestrian crashes
    crashes = read_table(cfg, spark, "gold", "crashes").filter("is_ped")
    sun_df = read_table(cfg, spark, "gold", "sun")
    days = read_table(cfg, spark, "gold", "periods").filter("period = 'day'") \
        .groupBy(F.month("date").alias("month")).agg(F.count("*").alias("n_days")).toPandas()
    hm = crashes.groupBy(F.month("crash_date").alias("month"), F.hour("crash_ts").alias("hour")) \
        .agg(F.count("*").alias("crashes")).toPandas().merge(days, on="month")
    hm["per_100_days"] = hm["crashes"] / hm["n_days"] * 100
    clock = lambda c: F.hour(c) + F.minute(c) / 60  # noqa: E731
    sun_m = sun_df.groupBy(F.month("date").alias("month")).agg(
        F.avg(clock("sunrise")).alias("sunrise_h"), F.avg(clock("sunset")).alias("sunset_h")).toPandas()
    viz.hour_month_heatmap(hm, sun_m, fig / "01_hour_month_heatmap.png")
    viz.dump_table(hm.sort_values(["month", "hour"]), tab / "01_hour_month_heatmap.csv")

    # 2. effect estimates
    effects = _maybe(cfg, spark, "effects")
    if effects is not None:
        viz.forest(effects, fig / "02_effect_forest.png")
        viz.dump_table(effects, tab / "02_effects.csv")

    # 3. event study
    es = _maybe(cfg, spark, "event_study")
    if es is not None:
        viz.event_study(es, fig / "03_event_study_dark.png", "dark")
        viz.event_study(es, fig / "03_event_study_placebo.png", "placebo")
        viz.dump_table(es, tab / "03_event_study.csv")

    # 4. score vs arrival order
    curve, summary = _maybe(cfg, spark, "score_curve"), _maybe(cfg, spark, "score_summary")
    if curve is not None and summary is not None:
        s = summary.iloc[0].to_dict()
        viz.capture(curve.sort_values("top_share"), s, fig / "04_score_vs_arrival.png")
        viz.dump_table(curve.sort_values("top_share"), tab / "04_score_curve.csv")
        viz.dump_json(s, tab / "04_score_summary.json")

    # 5. classifier
    for suffix, target in (("", "night pedestrian crash within 7 nights"),
                           ("_any_crash", "any night crash within 7 nights")):
        mm = _maybe(cfg, spark, f"ml_metrics{suffix}")
        if mm is not None:
            viz.model_bars(mm, fig / f"05_models{suffix}.png", target)
            viz.dump_table(mm, tab / f"05_ml_metrics{suffix}.csv")
            imp = _maybe(cfg, spark, f"ml_importances{suffix}")
            if imp is not None:
                viz.dump_table(imp, tab / f"05_ml_importances{suffix}.csv")

    # 6. map
    scores, ranked = _maybe(cfg, spark, "cell_scores"), _maybe(cfg, spark, "open_ranked")
    if scores is not None and ranked is not None:
        years = read_table(cfg, spark, "gold", "periods").filter("period = 'night'").count() / 365.25
        nights = (
            read_table(cfg, spark, "gold", "exposure")
            .filter("spec = 'A' and geo = 'r10k0' and kind = 'dark' and period = 'night'")
            .groupBy("cell").agg((F.count("*") / F.lit(years)).alias("outage_nights")).toPandas()
        )
        cells = scores.merge(nights, on="cell", how="left").fillna({"outage_nights": 0.0})
        cells["expected"] = cells["score"] * cells["outage_nights"] / 365.0
        viz.build_map(cells, ranked, REPO_ROOT / "reports" / "map.html")
        viz.dump_table(ranked.drop(columns=["cell"]).head(500), tab / "06_open_complaints_top500.csv")
        viz.dump_table(cells.sort_values("expected", ascending=False).head(500), tab / "06_cells_top500.csv")

    print(f"figures: {fig}\nmap:     {REPO_ROOT / 'reports' / 'map.html'}")
    spark.stop()
