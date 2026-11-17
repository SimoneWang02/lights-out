import pytest

from lightsout.config import load_config
from lightsout.spark import get_spark


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def spark(cfg):
    spark = get_spark(cfg, "lightsout-tests")
    spark.conf.set("spark.sql.shuffle.partitions", "4")
    yield spark
    spark.stop()
