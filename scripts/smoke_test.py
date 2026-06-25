"""Connectivity + auth smoke test for all 4 external services this pipeline
depends on: PodcastIndex, RunPod, HuggingFace (gated pyannote model
agreement), and Cloudflare R2. No GPU required.

Two modes:
  --check-network-only: a cheap raw-reachability probe (DNS+TCP+TLS, no
    auth) against each host. This is "Step 0" of the staged rollout -- run
    this first after any network policy change, before touching real
    credentials, since a proxy/firewall policy denial surfaces here as a
    connection failure distinct from an auth failure.
  (default): full auth-validating checks -- PodcastIndexClient.check_connectivity(),
    RunPodClient.check_connectivity(), HfApi.whoami() + a model_info() call
    against the gated pyannote model to confirm the license was accepted,
    and a real R2 put/get/list/delete of a throwaway object.

All of these are free (no paid API tier, and R2 throwaway object churn is
negligible against the free tier), so this script needs no spend
confirmation -- unlike scripts/bootstrap_pod.py, which actually creates a
billed pod.

Run with: python3 scripts/smoke_test.py [--check-network-only]
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline import config, logging_utils, storage  # noqa: E402
from pipeline.podcastindex_client import PodcastIndexClient, PodcastIndexError  # noqa: E402
from pipeline.runpod_client import RunPodClient, RunPodError  # noqa: E402

logger = logging_utils.get_logger()

NETWORK_CHECK_TIMEOUT = 10.0
GATED_MODEL_ID = "pyannote/speaker-diarization-3.1"


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def check_host_reachable(name: str, url: str) -> CheckResult:
    """Any HTTP response (even 4xx/5xx) means DNS+TCP+TLS succeeded -- a
    network/proxy policy denial fails before that point and raises a
    requests exception, which is what this distinguishes from an auth or
    application-level error on the other side."""
    try:
        resp = requests.get(url, timeout=NETWORK_CHECK_TIMEOUT)
        return CheckResult(name, True, f"reachable (HTTP {resp.status_code})")
    except requests.exceptions.RequestException as exc:
        return CheckResult(name, False, f"unreachable: {exc}")


def run_network_only_checks(secrets: config.EnvSecrets) -> list[CheckResult]:
    hosts = {
        "podcastindex": "https://api.podcastindex.org/api/1.0/search/byterm",
        "runpod": "https://rest.runpod.io/v1/openapi.json",
        "huggingface": "https://huggingface.co",
    }
    if secrets.r2_account_id:
        hosts["r2"] = f"https://{secrets.r2_account_id}.r2.cloudflarestorage.com"
    else:
        hosts["r2"] = "https://r2.cloudflarestorage.com"
    return [check_host_reachable(name, url) for name, url in hosts.items()]


def check_podcastindex(secrets: config.EnvSecrets) -> CheckResult:
    if not secrets.podcastindex_api_key or not secrets.podcastindex_api_secret:
        return CheckResult("podcastindex", False, "PODCASTINDEX_API_KEY/SECRET not set")
    try:
        PodcastIndexClient(secrets.podcastindex_api_key, secrets.podcastindex_api_secret).check_connectivity()
        return CheckResult("podcastindex", True, "auth ok")
    except PodcastIndexError as exc:
        return CheckResult("podcastindex", False, str(exc))


def check_runpod(secrets: config.EnvSecrets) -> CheckResult:
    if not secrets.runpod_api_key:
        return CheckResult("runpod", False, "RUNPOD_API_KEY not set")
    try:
        RunPodClient(secrets.runpod_api_key).check_connectivity()
        return CheckResult("runpod", True, "auth ok")
    except RunPodError as exc:
        return CheckResult("runpod", False, str(exc))


def check_huggingface(secrets: config.EnvSecrets) -> CheckResult:
    if not secrets.hf_token:
        return CheckResult("huggingface", False, "HF_TOKEN not set")
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.utils import HfHubHTTPError

        api = HfApi()
        who = api.whoami(token=secrets.hf_token)
        try:
            api.model_info(GATED_MODEL_ID, token=secrets.hf_token)
        except HfHubHTTPError as exc:
            return CheckResult(
                "huggingface", False,
                f"token valid (user={who.get('name')}) but gated model access failed for "
                f"{GATED_MODEL_ID} -- accept the license at https://huggingface.co/{GATED_MODEL_ID}: {exc}",
            )
        return CheckResult("huggingface", True, f"token valid (user={who.get('name')}), gated model access confirmed")
    except Exception as exc:  # noqa: BLE001 - surface any hf_hub error as a failed check, not a crash
        return CheckResult("huggingface", False, str(exc))


def check_r2(secrets: config.EnvSecrets) -> CheckResult:
    if not secrets.r2_account_id or not secrets.r2_access_key_id or not secrets.r2_secret_access_key or not secrets.r2_bucket_name:
        return CheckResult("r2", False, "R2_ACCOUNT_ID/ACCESS_KEY_ID/SECRET_ACCESS_KEY/BUCKET_NAME not fully set")
    try:
        client = storage.build_client(secrets)
        key = "smoke_test/throwaway.json"
        storage.put_json(client, secrets.r2_bucket_name, key, {"smoke_test": True})
        got = storage.get_json(client, secrets.r2_bucket_name, key)
        if got != {"smoke_test": True}:
            return CheckResult("r2", False, f"put/get round-trip mismatch: got {got!r}")
        keys = storage.list_keys(client, secrets.r2_bucket_name, prefix="smoke_test/")
        if key not in keys:
            return CheckResult("r2", False, f"uploaded key {key!r} not found in list_keys result")
        client.delete_object(Bucket=secrets.r2_bucket_name, Key=key)
        return CheckResult("r2", True, "put/get/list/delete round-trip ok")
    except Exception as exc:  # noqa: BLE001 - boto3/botocore raise many distinct exception types
        return CheckResult("r2", False, str(exc))


def print_results(results: list[CheckResult]) -> bool:
    all_ok = True
    for r in results:
        status = "OK" if r.ok else "FAIL"
        print(f"[{status}] {r.name}: {r.detail}")
        all_ok = all_ok and r.ok
    return all_ok


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-network-only", action="store_true", help="raw reachability only, no credentials used")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    secrets = config.EnvSecrets.from_env()

    if args.check_network_only:
        print("=== network reachability (no auth) ===")
        results = run_network_only_checks(secrets)
        ok = print_results(results)
        if not ok:
            logger.error("one or more hosts unreachable -- this is the network policy gate; do not proceed until fixed")
        return 0 if ok else 1

    print("=== full connectivity + auth checks ===")
    results = [
        check_podcastindex(secrets),
        check_runpod(secrets),
        check_huggingface(secrets),
        check_r2(secrets),
    ]
    ok = print_results(results)
    if not ok:
        logger.error("one or more services failed; fix credentials/access before any real run")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
