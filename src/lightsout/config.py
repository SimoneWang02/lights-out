"""Configuration loading and table paths.

Every job takes ``--conf conf/<name>.yaml``. The config decides where data lives
(local folder or HDFS) and the analysis parameters, so the same code runs in both places.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]

    @property
    def data_root(self) -> str:
        root = str(self.raw["data_root"])
        if "://" in root or root.startswith("/"):
            return root.rstrip("/")
        return str(REPO_ROOT / root)

    @property
    def is_remote(self) -> bool:
        return "://" in str(self.raw["data_root"])

    def path(self, layer: str, name: str = "") -> str:
        """Location of a table: <data_root>/<layer>/<name>. Layers: raw, silver, gold, out."""
        return "/".join(p for p in (self.data_root, layer, name) if p)

    @property
    def spark(self) -> dict[str, Any]:
        return self.raw["spark"]

    @property
    def analysis(self) -> dict[str, Any]:
        return self.raw["analysis"]

    @property
    def nyc(self) -> dict[str, Any]:
        return self.raw["nyc"]


def load_config(path: str | os.PathLike | None = None) -> Config:
    path = Path(path or os.environ.get("LIGHTSOUT_CONF") or REPO_ROOT / "conf" / "local.yaml")
    with open(path) as f:
        return Config(yaml.safe_load(f))


def job_args(description: str, extra=None) -> tuple[Config, argparse.Namespace]:
    """Standard argument parsing for scripts under jobs/."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--conf", default=None, help="config file (default: conf/local.yaml)")
    if extra:
        extra(parser)
    args = parser.parse_args()
    return load_config(args.conf), args
