"""CLI entrypoint for pipeline/backfill.py against a local db (e.g. an
already-downloaded shard snapshot) -- see that module's docstring for the
root cause this recovers from. run_pipeline.py runs the same logic
automatically on every pod boot; this script is for running it manually
against a db that isn't attached to a live run_pipeline.py process.

Run with:
  python3 scripts/backfill_uploads.py --db work/pipeline.db
  python3 scripts/backfill_uploads.py --db work/pipeline.db --dry-run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline import backfill, config, db, logging_utils, storage  # noqa: E402

logger = logging_utils.get_logger()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default=str(db.DEFAULT_DB_PATH), help="sqlite db path (this pod's local pipeline.db)")
    parser.add_argument("--dry-run", action="store_true", help="report what would be re-uploaded without uploading anything")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    secrets = config.EnvSecrets.from_env()
    try:
        secrets.require_r2()
    except config.ConfigError as exc:
        logger.error("cannot backfill, missing R2 credentials: %s", exc)
        return 1

    conn = db.connect(args.db)

    if args.dry_run:
        rows = conn.execute(
            "SELECT local_flac_path FROM clips WHERE uploaded = 1 AND local_flac_path IS NOT NULL"
        ).fetchall()
        on_disk = sum(1 for r in rows if Path(r["local_flac_path"]).exists())
        print(f"DRY RUN: would check/re-upload {on_disk} clip(s); {len(rows) - on_disk} already unrecoverable (file gone)")
        return 0

    client = storage.build_client(secrets)
    result = backfill.backfill_uploaded_clips(conn, client, secrets.r2_bucket_name)
    conn.close()

    print(f"\nbackfill complete: {result.candidates} candidate(s) on disk, {result.missing} already unrecoverable")
    print(f"  {result.already_present} already present in R2 (no-op)")
    print(f"  {result.reuploaded} re-uploaded and verified")
    print(f"  {result.failed} failed -- see log above")
    return 1 if result.failed else 0


if __name__ == "__main__":
    sys.exit(main())
