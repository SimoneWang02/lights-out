"""Stage 3: sun table, day/night labels and H3 cells -> gold crashes and complaints."""
from lightsout import exposure
from lightsout.config import job_args
from lightsout.spark import get_spark

if __name__ == "__main__":
    cfg, args = job_args(
        __doc__,
        lambda p: p.add_argument("--h3-mode", choices=["udf", "driver"], default="udf",
                                 help="'driver' needs h3 only on the driver (cluster fallback)"),
    )
    spark = get_spark(cfg, "lightsout-keys")
    exposure.build_keys(cfg, spark, args.h3_mode)
    spark.stop()
