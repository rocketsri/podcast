"""Tars pipeline/, config/, requirements.txt, and run_pipeline.py, then
uploads the tarball to R2 so infra/bootstrap.sh can fetch it onto a fresh
RunPod pod (no custom Docker image / registry needed -- see the plan's
RunPod execution design). Secrets never go in this tarball; they reach the
pod exclusively via RunPod's own env-var injection at pod creation.

Run with: python3 scripts/package_code.py
"""
from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline import config, logging_utils, storage  # noqa: E402

logger = logging_utils.get_logger()

PACKAGED_PATHS = ["pipeline", "config", "requirements.txt", "run_pipeline.py"]
DEFAULT_TARBALL_NAME = "code.tar.gz"
DEFAULT_R2_KEY = "code/code.tar.gz"


def _exclude_bytecode(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo | None:
    if "__pycache__" in tarinfo.name or tarinfo.name.endswith((".pyc", ".pyo")):
        return None
    return tarinfo


def build_tarball(output_path: str | Path, repo_root: Path = REPO_ROOT) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output_path, "w:gz") as tar:
        for rel_path in PACKAGED_PATHS:
            full_path = repo_root / rel_path
            if not full_path.exists():
                raise FileNotFoundError(f"expected path missing, cannot package: {full_path}")
            tar.add(full_path, arcname=rel_path, filter=_exclude_bytecode)
    return output_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(REPO_ROOT / "work" / DEFAULT_TARBALL_NAME), help="local tarball path")
    parser.add_argument("--r2-key", default=DEFAULT_R2_KEY, help="destination key in the R2 bucket")
    parser.add_argument("--no-upload", action="store_true", help="build the tarball locally only; skip R2 upload")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    tarball_path = build_tarball(args.output)
    size_mb = tarball_path.stat().st_size / (1024 * 1024)
    logger.info("built %s (%.2f MB)", tarball_path, size_mb)

    if args.no_upload:
        logger.info("--no-upload set: tarball left at %s, not uploaded", tarball_path)
        return 0

    secrets = config.EnvSecrets.from_env()
    try:
        secrets.require_r2()
    except config.ConfigError as exc:
        logger.error("cannot upload, missing R2 credentials: %s", exc)
        return 1

    client = storage.build_client(secrets)
    storage.upload_file(client, secrets.r2_bucket_name, tarball_path, args.r2_key)
    logger.info("uploaded %s -> s3://%s/%s", tarball_path, secrets.r2_bucket_name, args.r2_key)
    return 0


if __name__ == "__main__":
    sys.exit(main())
