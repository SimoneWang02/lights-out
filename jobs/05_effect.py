"""Stage 5: outage effect estimates (sensitivity grid, placebo) and the event study."""
from lightsout import effect, event_study
from lightsout.config import job_args
from lightsout.spark import get_spark, write_pandas

if __name__ == "__main__":
    cfg, args = job_args(
        __doc__,
        lambda p: p.add_argument("--h3-mode", choices=["udf", "driver"], default="udf"),
    )
    spark = get_spark(cfg, "lightsout-effect")
    write_pandas(cfg, spark, effect.run_grid(cfg, spark), "effects")
    es = event_study.run(cfg, spark, args.h3_mode)
    write_pandas(cfg, spark, es, "event_study")
    print(es[(es.kind == "dark") & (es.rel.between(-3, 3))].to_string(index=False))
    spark.stop()
