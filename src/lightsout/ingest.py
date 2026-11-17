"""Download the two source datasets from NYC Open Data.

Both come from the SoQL API as one CSV per calendar year, fetched a few years at a time.
(The portal's single-file export of the collisions table is about 450 MB and tends to drop
the connection part-way; yearly files are quicker and a failed year can simply be retried.)

* Collisions: every year since 2012.
* 311: filtered to Street Light Condition, every year since 2020.

The API uses the original column names (complaint_type, descriptor, crash_date ...). The
portal's own CSV export uses display names ("CRASH DATE", "Problem (formerly Complaint
Type)"). clean.py accepts either, so a file exported by hand from the portal also works.

The download date is saved to raw/extraction.json.

Files land on the local disk. For Dataproc, copy data/raw to HDFS with `hdfs dfs -put`.
"""
from __future__ import annotations

import datetime as dt
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

COLLISIONS_URL = "https://data.cityofnewyork.us/resource/h9gi-nx95.csv"
SR311_URL = "https://data.cityofnewyork.us/resource/erm2-nwe9.csv"
COMPLAINT_TYPE = "Street Light Condition"
UA = {"User-Agent": "lightsout-course-project"}


def _stream(url: str, dest: Path, params: dict | None = None, retries: int = 4) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, params=params, headers=UA, stream=True, timeout=(30, 300)) as r:
                r.raise_for_status()
                n = 0
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                        n += len(chunk)
            tmp.replace(dest)
            return n
        except (requests.RequestException, OSError) as e:
            if attempt == retries:
                raise
            print(f"  retry {attempt}/{retries - 1} after error: {e}", flush=True)
            time.sleep(5 * attempt)
    return 0


def _yearly(url: str, dest_dir: Path, stem: str, date_col: str, order: str, first_year: int,
            extra_where: str = "", force: bool = False, workers: int = 4) -> list[Path]:
    this_year = dt.date.today().year

    def one(year: int) -> Path:
        dest = dest_dir / f"{stem}_{year}.csv"
        # past years barely change; the current year is always refreshed
        if dest.exists() and not force and year < this_year:
            print(f"{stem} {year}: already present, skipping", flush=True)
            return dest
        where = f"{date_col} >= '{year}-01-01T00:00:00' AND {date_col} < '{year + 1}-01-01T00:00:00'"
        if extra_where:
            where = f"{extra_where} AND {where}"
        n = _stream(url, dest, params={"$where": where, "$limit": 5_000_000, "$order": order})
        print(f"{stem} {year}: {n / 1e6:.1f} MB", flush=True)
        return dest

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, range(first_year, this_year + 1)))


def download_collisions(raw_dir: Path, first_year: int = 2012, force: bool = False) -> list[Path]:
    return _yearly(COLLISIONS_URL, raw_dir / "collisions", "collisions", "crash_date", "collision_id",
                   first_year, force=force)


def download_311(raw_dir: Path, first_year: int = 2020, force: bool = False) -> list[Path]:
    return _yearly(SR311_URL, raw_dir / "complaints", "streetlight", "created_date", "unique_key",
                   first_year, extra_where=f"complaint_type='{COMPLAINT_TYPE}'", force=force)


def run(raw_dir: str | Path, force: bool = False) -> None:
    raw_dir = Path(raw_dir)
    download_collisions(raw_dir, force=force)
    download_311(raw_dir, force=force)
    meta = {"extraction_date": dt.date.today().isoformat()}
    (raw_dir / "extraction.json").write_text(json.dumps(meta, indent=2))
    print(f"extraction date recorded: {meta['extraction_date']}")
