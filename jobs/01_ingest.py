"""Stage 1: download raw CSVs to <repo>/data/raw (always local disk)."""
from lightsout import ingest
from lightsout.config import REPO_ROOT, job_args

if __name__ == "__main__":
    cfg, args = job_args(
        __doc__, lambda p: p.add_argument("--force", action="store_true", help="re-download everything")
    )
    raw = REPO_ROOT / "data" / "raw" if cfg.is_remote else cfg.path("raw")
    ingest.run(raw, force=args.force)
