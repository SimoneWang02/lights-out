"""Stage 2: raw CSV -> silver tables and the data-quality table."""
from lightsout import clean
from lightsout.config import job_args
from lightsout.spark import get_spark

if __name__ == "__main__":
    cfg, _ = job_args(__doc__)
    spark = get_spark(cfg, "lightsout-clean")
    clean.run(cfg, spark).show(50, truncate=False)
    spark.stop()
