"""Stage 6: priority score per cell, comparison with arrival order, ranked list of open complaints."""
import pandas as pd

from lightsout import score
from lightsout.config import job_args
from lightsout.spark import get_spark, write_pandas

if __name__ == "__main__":
    cfg, args = job_args(
        __doc__,
        lambda p: p.add_argument("--h3-mode", choices=["udf", "driver"], default="udf"),
    )
    spark = get_spark(cfg, "lightsout-score")
    scores = score.cell_scores(cfg, spark, args.h3_mode)
    write_pandas(cfg, spark, scores, "cell_scores")
    print(scores.groupby("borough")[["prior_mean", "prior_shape"]].first())

    holdout = score.holdout_complaints(cfg, spark, args.h3_mode).toPandas()
    curve, summary = score.evaluate(scores, holdout)
    write_pandas(cfg, spark, curve, "score_curve")
    write_pandas(cfg, spark, pd.DataFrame([summary]), "score_summary")
    for k, v in summary.items():
        print(f"{k:<24} {v:.3f}" if isinstance(v, float) else f"{k:<24} {v:,}")

    ranked = score.open_complaints_ranked(cfg, spark, scores)
    write_pandas(cfg, spark, ranked, "open_ranked")
    print(ranked.head(10)[["rank_by_score", "rank_by_arrival", "score", "days_open", "borough",
                           "intersection_street_1", "intersection_street_2"]].to_string(index=False))
    spark.stop()
