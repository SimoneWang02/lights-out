"""Stage 4: outage windows -> (cell, date, period) exposure table for every spec and geo."""
from pyspark.sql import functions as F

from lightsout import exposure
from lightsout.config import job_args
from lightsout.spark import get_spark, read_table

if __name__ == "__main__":
    cfg, args = job_args(
        __doc__,
        lambda p: p.add_argument("--h3-mode", choices=["udf", "driver"], default="udf"),
    )
    spark = get_spark(cfg, "lightsout-exposure")
    exposure.build_exposure(cfg, spark, args.h3_mode)
    (read_table(cfg, spark, "gold", "exposure")
        .groupBy("spec", "geo", "kind", "period").agg(F.count("*").alias("cell_periods"))
        .orderBy("spec", "geo", "kind", "period").show(60))
    spark.stop()
