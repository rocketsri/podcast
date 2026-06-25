"""boto3 S3-compatible client for Cloudflare R2: put/get/list for clips,
manifest, and status objects. R2 speaks the S3 API over a per-account
endpoint (https://<account_id>.r2.cloudflarestorage.com) with
region_name="auto" -- Cloudflare's own documented connection shape for the
S3-compatible API. Confirmed against the installed boto3==1.35.99 client via
inspect (upload_file/download_file take positional Filename/Bucket/Key;
put_object/get_object/list_objects_v2 take the standard Bucket/Key/Body/
Prefix kwargs that have been stable across the S3 API for years), not
guessed.

This module has no network test coverage in this sandbox -- the egress
proxy explicitly denies Cloudflare hosts (see LIMITATIONS.md) -- so the pure
key-naming helpers are unit-tested, but put/get/list themselves are thin,
well-trusted boto3 pass-throughs rather than independently reimplemented
logic.
"""

from __future__ import annotations

import json
from pathlib import Path

import boto3
from botocore.client import Config as BotoConfig
from botocore.exceptions import ClientError

from pipeline import config


def build_client(secrets: config.EnvSecrets):
    return boto3.client(
        "s3",
        endpoint_url=f"https://{secrets.r2_account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=secrets.r2_access_key_id,
        aws_secret_access_key=secrets.r2_secret_access_key,
        region_name="auto",
        config=BotoConfig(retries={"max_attempts": 4, "mode": "standard"}),
    )


def clip_key(podcast_id: str, episode_id: str, clip_id: str) -> str:
    return f"clips/{podcast_id}/{episode_id}/{clip_id}.flac"


def clip_url(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key}"


def manifest_key(podcast_id: str | None = None) -> str:
    return f"manifest/{podcast_id}.jsonl" if podcast_id else "manifest/manifest.jsonl"


def status_key(pod_id: str) -> str:
    return f"status/{pod_id}.json"


def db_snapshot_key(pod_id: str) -> str:
    return f"db_snapshots/{pod_id}/pipeline.db"


def upload_clip(client, bucket: str, local_flac_path: str | Path, podcast_id: str, episode_id: str, clip_id: str) -> str:
    """Uploads an already-encoded clip flac and returns the s3:// URL to
    persist as clips.audio_path -- the exact form the spec's manifest schema
    expects."""
    key = clip_key(podcast_id, episode_id, clip_id)
    client.upload_file(str(local_flac_path), bucket, key)
    return clip_url(bucket, key)


def upload_file(client, bucket: str, local_path: str | Path, key: str) -> None:
    client.upload_file(str(local_path), bucket, key)


def download_file(client, bucket: str, key: str, local_path: str | Path) -> None:
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(bucket, key, str(local_path))


def put_json(client, bucket: str, key: str, data: dict) -> None:
    client.put_object(Bucket=bucket, Key=key, Body=json.dumps(data).encode("utf-8"), ContentType="application/json")


def get_json(client, bucket: str, key: str) -> dict:
    response = client.get_object(Bucket=bucket, Key=key)
    return json.loads(response["Body"].read())


def object_exists(client, bucket: str, key: str) -> bool:
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        if exc.response["ResponseMetadata"]["HTTPStatusCode"] == 404:
            return False
        raise


def list_keys(client, bucket: str, prefix: str = "") -> list[str]:
    keys = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys
