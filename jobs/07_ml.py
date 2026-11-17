"""Stage 7: MLlib classifier for 'night pedestrian crash near this complaint within a week'."""
from lightsout import ml
from lightsout.config import job_args
from lightsout.spark import get_spark, read_table, write_pandas, write_table

if __name__ == "__main__":
    cfg, args = job_args(
        __doc__,
        lambda p: p.add_argument("--h3-mode", choices=["udf", "driver"], default="udf"),
    )
    spark = get_spark(cfg, "lightsout-ml")
    write_table(cfg, ml.build_dataset(cfg, spark, args.h3_mode), "gold", "ml_dataset")
    data = read_table(cfg, spark, "gold", "ml_dataset")
    scores = read_table(cfg, spark, "out", "cell_scores").select("cell", "score")

    for label in ("label", "label_any_crash"):
        metrics, importances, preds = ml.train_and_evaluate(data, scores, label)
        suffix = "" if label == "label" else "_any_crash"
        write_pandas(cfg, spark, metrics, f"ml_metrics{suffix}")
        write_pandas(cfg, spark, importances, f"ml_importances{suffix}")
        write_table(cfg, preds, "out", f"ml_predictions{suffix}")
        print(f"\n== target: {label} ==")
        print(metrics.round(4).to_string(index=False))
        print(importances.head(10).round(4).to_string(index=False))
    spark.stop()
