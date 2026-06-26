"""Structured (JSON-lines) logging to a local file, plus periodic R2 sync.

Secret redaction here is defense in depth, not the primary control: secrets
reach a RunPod pod exclusively via RunPod's own env-var injection at pod
creation (never the code tarball or git), but an exception message or a
third-party library's repr could still echo a credential into a log line,
so every formatted record is scrubbed before it touches disk or R2.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

LOGGER_NAME = "pipeline"
_REDACTED = "[REDACTED]"

# Matches "<anything>(api_key|secret|token|password)<anything> <:|=> <value>"
# case-insensitively, so compound env-style names (R2_SECRET_ACCESS_KEY,
# PODCASTINDEX_API_SECRET, HF_TOKEN, RUNPOD_API_KEY, ...) are caught by the
# substring, not just an exact key match.
_SECRET_KEY_PATTERN = re.compile(
    r"(?i)(\w*(?:api[_-]?key|secret|token|password)\w*)(\s*[:=]\s*)([\"']?)([^\s\"',}]+)([\"']?)"
)


def redact(text: str) -> str:
    return _SECRET_KEY_PATTERN.sub(
        lambda m: f"{m.group(1)}{m.group(2)}{m.group(3)}{_REDACTED}{m.group(5)}", text
    )


class _RedactingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record))


class JsonFormatter(_RedactingFormatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return redact(json.dumps(payload))


def configure_logging(log_path: str | Path, level: int = logging.INFO) -> logging.Logger:
    """(Re)configures the shared `pipeline` logger: JSON lines to `log_path`
    (for R2 sync + later inspection) and a human-readable stream to stderr.
    Safe to call more than once (e.g. on resume) -- clears prior handlers
    rather than stacking duplicates."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(JsonFormatter())
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(_RedactingFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(stream_handler)

    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def sync_log_to_r2(client, bucket: str, pod_id: str, log_path: str | Path, key_prefix: str = "") -> None:
    from pipeline import storage  # local import: keeps logging_utils importable with no boto3 present

    storage.upload_file(client, bucket, log_path, storage.log_key(pod_id, key_prefix))
