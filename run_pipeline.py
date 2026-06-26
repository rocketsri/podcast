"""CLI entrypoint: the per-pod process that drives the stage machine over
its shard of queued episodes (see pipeline_runner.run_queue). What actually
runs on a RunPod pod, per infra/bootstrap.sh.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pipeline import asr, backfill, config, db, diarize, heartbeat, logging_utils, pipeline_runner, storage, vad

logger = logging_utils.get_logger()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to pipeline.yaml (default: config/pipeline.yaml)")
    parser.add_argument("--db", default=str(db.DEFAULT_DB_PATH), help="sqlite db path")
    parser.add_argument("--work-dir", default="work", help="local scratch dir for raw/wav/clip files")
    parser.add_argument("--log-path", default="work/pipeline.log")
    parser.add_argument("--shard", type=int, default=None, help="this pod's assigned_shard id (omit for single-pod/Stage-1)")
    parser.add_argument("--pod-id", default="local-pod", help="this pod's id, used in heartbeat status keys")
    parser.add_argument("--max-episodes", type=int, default=None, help="cap episodes processed this run (smoke tests)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--no-upload", action="store_true", help="skip R2 upload; clips stay local (dev/smoke-test mode without R2 credentials)")
    parser.add_argument("--status-port", type=int, default=None, help="override config.monitoring.status_http_port; 0 disables the HTTP status server")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = config.load_config(args.config)
    secrets = config.EnvSecrets.from_env()

    logging_utils.configure_logging(args.log_path)
    conn = db.connect(args.db)
    db.init_db(conn)

    logger.info("loading models (device=%s) ...", args.device)
    vad_model = vad.load_model()
    diarize_pipeline = diarize.load_pipeline(
        secrets.hf_token, device=args.device, embedding_exclude_overlap=cfg.clustering.embedding_exclude_overlap,
    )
    compute_type = "float16" if args.device == "cuda" else "int8"
    asr_model = asr.load_model(cfg.models.asr, device=args.device, compute_type=compute_type)
    models = pipeline_runner.Models(vad_model=vad_model, diarize_pipeline=diarize_pipeline, asr_model=asr_model)

    storage_client, bucket = None, None
    if not args.no_upload:
        try:
            secrets.require_r2()
            storage_client = storage.build_client(secrets)
            bucket = secrets.r2_bucket_name
        except config.ConfigError as exc:
            logger.warning("R2 credentials incomplete (%s) -- running in local-only mode, clips stay on disk", exc)

    if storage_client is not None and bucket is not None:
        # Runs on every boot (including a plain restart_pod() of an
        # already-running pod -- see pipeline/backfill.py's docstring for why
        # that's the only way a code fix reaches a live pod here). Cheap
        # no-op when there's nothing stale: only clips already flagged
        # uploaded=1 with a local file still on disk are even considered.
        result = backfill.backfill_uploaded_clips(conn, storage_client, bucket, key_prefix=secrets.r2_key_prefix)
        if result.candidates:
            logger.info(
                "startup backfill: %d candidate(s) checked, %d already in R2, %d re-uploaded, %d failed, %d unrecoverable (file gone)",
                result.candidates, result.already_present, result.reuploaded, result.failed, result.missing,
            )

    if db.get_run_meta(conn, "gpu_hourly_rate_usd") is None:
        db.set_run_meta(conn, "gpu_hourly_rate_usd", str(cfg.cost.assumed_gpu_hourly_usd))

    ctx = pipeline_runner.RunContext(
        conn=conn, cfg=cfg, models=models, work_dir=Path(args.work_dir),
        storage_client=storage_client, bucket=bucket, pod_id=args.pod_id, shard_id=args.shard,
        db_path=args.db, log_path=args.log_path, num_pods=secrets.num_pods, r2_key_prefix=secrets.r2_key_prefix,
    )

    status_port = args.status_port if args.status_port is not None else cfg.monitoring.status_http_port
    status_server = None
    if status_port:
        status_server = heartbeat.StatusServer(
            lambda: ctx.latest_status or {"pod_id": args.pod_id, "status": "starting"}, port=status_port
        )
        status_server.start()
        logger.info("status server listening on :%d", status_port)

    try:
        summary = pipeline_runner.run_queue(ctx, shard=args.shard, max_episodes=args.max_episodes)
    finally:
        if status_server is not None:
            status_server.stop()

    logger.info("run complete: %s", summary)
    logger.info("total cost so far: $%.4f", db.total_cost(conn))
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
