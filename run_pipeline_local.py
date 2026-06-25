"""CLI entrypoint for the free/local trial path: CPU-only models (Silero
VAD, Resemblyzer-based pipeline/local_diarize.py in place of gated pyannote,
faster-whisper int8) -- no signup, no API keys, no rented GPU. Clips and the
manifest stay on local disk (pipeline_runner.py's existing storage_client=None
fallback); there is no R2 upload to configure.

See run_pipeline.py for the credentialed RunPod+R2+pyannote original, left
untouched -- this is a parallel entrypoint for when those credentials aren't
available, not a replacement.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pipeline import asr, config, db, heartbeat, local_diarize, logging_utils, pipeline_runner, vad

logger = logging_utils.get_logger()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to pipeline.yaml (default: config/pipeline.yaml)")
    parser.add_argument("--db", default=str(db.DEFAULT_DB_PATH), help="sqlite db path")
    parser.add_argument("--work-dir", default="work", help="local scratch dir for raw/wav/clip files")
    parser.add_argument("--log-path", default="work/pipeline.log")
    parser.add_argument("--shard", type=int, default=None, help="this pod's assigned_shard id (omit for single-pod runs)")
    parser.add_argument("--pod-id", default="local-free-pod", help="this pod's id, used in heartbeat status keys")
    parser.add_argument("--max-episodes", type=int, default=None, help="cap episodes processed this run (smoke tests)")
    parser.add_argument("--status-port", type=int, default=None, help="override config.monitoring.status_http_port; 0 disables the HTTP status server")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = config.load_config(args.config)

    logging_utils.configure_logging(args.log_path)
    conn = db.connect(args.db)
    db.init_db(conn)

    logger.info("loading CPU-only models (silero VAD + resemblyzer diarization + faster-whisper int8) ...")
    vad_model = vad.load_model()
    diarize_pipeline = local_diarize.load_pipeline(device="cpu", match_threshold=cfg.clustering.match_threshold)
    asr_model = asr.load_model(cfg.models.asr, device="cpu", compute_type="int8")
    models = pipeline_runner.Models(vad_model=vad_model, diarize_pipeline=diarize_pipeline, asr_model=asr_model)

    # No rented hardware in this path -- actual GPU/compute spend is $0,
    # unlike the credentialed RunPod path's cfg.cost.assumed_gpu_hourly_usd default.
    db.set_run_meta(conn, "gpu_hourly_rate_usd", "0")

    ctx = pipeline_runner.RunContext(
        conn=conn, cfg=cfg, models=models, work_dir=Path(args.work_dir),
        storage_client=None, bucket=None, pod_id=args.pod_id, shard_id=args.shard,
        diarize_fn=local_diarize.diarize,
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
