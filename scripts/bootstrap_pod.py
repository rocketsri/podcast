"""Creates N RunPod GPU pods, each running infra/bootstrap.sh as the pod's
dockerStartCmd. Each pod clones the repo, discovers its own independent
batch of episodes via scripts/select_podcasts_free.py (no PodcastIndex
credentials configured for this run), and processes everything it finds in
single-pod mode -- see infra/bootstrap.sh's docstring for why pods don't
share one pre-partitioned queue the way the original R2-tarball design's
--shard model assumed.

RunPod's create-pod API has no way to fetch an arbitrary file at container
start, so infra/bootstrap.sh's own source is read off disk here and passed
inline as `["bash", "-c", <script text>]` -- the script carries no secrets
(it only references env vars), those arrive separately via the per-pod
`env=` RunPod injects at creation, never through this inlined command or
the code tarball.

SPENDS REAL MONEY the moment --confirm is passed. Without --confirm this
only prints the plan (pod count, GPU type, assumed hourly rate,
hours-to-BUDGET_CAP_USD) and exits -- per PLAN.md's "notify before the
first dollar is spent" requirement.

Code reaches the pod via `git clone` (infra/bootstrap.sh), not the R2
tarball PLAN.md originally described -- git clone was already working and
there was no reason to switch back once R2 turned out to be reachable too
(see infra/bootstrap.sh). R2 is still used for everything the pod itself
does (clip/manifest/heartbeat upload). Pass --github-token if the repo
isn't public.

Run with: python3 scripts/bootstrap_pod.py --num-pods N --confirm
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline import config, logging_utils  # noqa: E402
from pipeline.runpod_client import GPU_TYPE_IDS, RunPodClient, RunPodError  # noqa: E402

logger = logging_utils.get_logger()

BOOTSTRAP_SCRIPT_PATH = REPO_ROOT / "infra" / "bootstrap.sh"
DEFAULT_IMAGE = "runpod/pytorch:1.0.7-cu1281-torch271-ubuntu2204"
DEFAULT_GPU_TYPE_ID = "NVIDIA GeForce RTX 3090"  # PLAN.md's locked-in compute choice
DEFAULT_GIT_REPO_URL = "https://github.com/rocketsri/podcast.git"
DEFAULT_GIT_BRANCH = "claude/podcast-speech-builder-e0kfs5"


def build_env(
    secrets: config.EnvSecrets, pod_id_label: str, shard_id: int,
    git_repo_url: str, git_branch: str, github_token: str, num_pods: int,
) -> dict[str, str]:
    env = {
        "POD_ID": pod_id_label,
        "SHARD": str(shard_id),
        "GIT_REPO_URL": git_repo_url,
        "GIT_BRANCH": git_branch,
        "R2_ACCOUNT_ID": secrets.r2_account_id,
        "R2_ACCESS_KEY_ID": secrets.r2_access_key_id,
        "R2_SECRET_ACCESS_KEY": secrets.r2_secret_access_key,
        "R2_BUCKET_NAME": secrets.r2_bucket_name,
        "HF_TOKEN": secrets.hf_token,
        "NUM_PODS": str(num_pods),
    }
    if github_token:
        env["GITHUB_TOKEN"] = github_token
    return env


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-pods", type=int, required=True, help="number of independent GPU pods to create")
    parser.add_argument(
        "--shard-offset", type=int, default=0,
        help="starting shard_id for this batch (e.g. 1 if shard 0 is already running elsewhere) -- avoids reusing a pod name/heartbeat key from a prior launch",
    )
    parser.add_argument("--gpu-type-id", default=DEFAULT_GPU_TYPE_ID, choices=GPU_TYPE_IDS)
    parser.add_argument("--cloud-type", default="COMMUNITY", choices=["COMMUNITY", "SECURE"])
    parser.add_argument("--container-disk-gb", type=int, default=30)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--git-repo-url", default=DEFAULT_GIT_REPO_URL)
    parser.add_argument("--git-branch", default=DEFAULT_GIT_BRANCH)
    parser.add_argument("--github-token", default="", help="only needed if the repo is private")
    parser.add_argument("--pod-name-prefix", default="podcast-shard")
    parser.add_argument(
        "--assumed-hourly-usd", type=float, default=0.30,
        help="per-pod $/hr used only for the pre-flight estimate printed before --confirm (RunPod Community Cloud RTX 3090 has historically run ~$0.19-0.30/hr; check current spot pricing before confirming)",
    )
    parser.add_argument("--confirm", action="store_true", help="actually call the RunPod API and create pods (spends real money); omit for a dry-run plan only")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.num_pods < 1:
        logger.error("--num-pods must be >= 1")
        return 1

    secrets = config.EnvSecrets.from_env()
    missing = [
        name
        for name, value in (
            ("RUNPOD_API_KEY", secrets.runpod_api_key),
            ("HF_TOKEN", secrets.hf_token),
            ("R2_ACCOUNT_ID", secrets.r2_account_id),
            ("R2_ACCESS_KEY_ID", secrets.r2_access_key_id),
            ("R2_SECRET_ACCESS_KEY", secrets.r2_secret_access_key),
            ("R2_BUCKET_NAME", secrets.r2_bucket_name),
        )
        if not value
    ]
    if missing:
        logger.error("missing required environment variables: %s", ", ".join(missing))
        return 1

    if not BOOTSTRAP_SCRIPT_PATH.exists():
        logger.error("missing %s", BOOTSTRAP_SCRIPT_PATH)
        return 1
    bootstrap_script = BOOTSTRAP_SCRIPT_PATH.read_text()

    total_hourly = args.assumed_hourly_usd * args.num_pods
    print("=== bootstrap_pod.py plan ===")
    print(f"image: {args.image}")
    print(f"gpu_type_id: {args.gpu_type_id}  x{args.num_pods} pods  ({args.cloud_type})")
    print("each pod discovers its own episode batch at boot (scripts/select_podcasts_free.py) -- no pre-partitioned queue to report here")
    print(f"assumed rate: ${args.assumed_hourly_usd:.2f}/hr/pod -> ${total_hourly:.2f}/hr combined")
    if total_hourly > 0:
        print(f"budget cap (BUDGET_CAP_USD): ${secrets.budget_cap_usd:.2f} -> ~{secrets.budget_cap_usd / total_hourly:.1f}h of combined runtime before the cap")
    print(f"time cap (TIME_CAP_HOURS): {secrets.time_cap_hours:.1f}h -> worst-case spend at that wall-clock: ${total_hourly * secrets.time_cap_hours:.2f}")

    if not args.confirm:
        print("\nDRY RUN -- no pods created. Re-run with --confirm to actually provision (this spends real money).")
        return 0

    client = RunPodClient(secrets.runpod_api_key)
    created = []
    for i in range(args.num_pods):
        shard_id = args.shard_offset + i
        pod_id_label = f"{args.pod_name_prefix}-{shard_id}"
        env = build_env(secrets, pod_id_label, shard_id, args.git_repo_url, args.git_branch, args.github_token, args.num_pods)
        try:
            result = client.create_pod(
                name=pod_id_label,
                image_name=args.image,
                gpu_type_id=args.gpu_type_id,
                cloud_type=args.cloud_type,
                container_disk_in_gb=args.container_disk_gb,
                env=env,
                docker_start_cmd=["bash", "-c", bootstrap_script],
                ports=["8080/http"],
            )
        except RunPodError as exc:
            logger.error("failed to create pod for shard %d: %s", shard_id, exc)
            continue
        created.append((pod_id_label, result))
        logger.info("created pod for shard %d (%s): runpod id=%s", shard_id, pod_id_label, result.get("id", result))

    print(f"\ncreated {len(created)}/{args.num_pods} pods")
    for pod_id_label, result in created:
        print(f"  {pod_id_label} -> runpod id {result.get('id', '?')}")
    return 0 if len(created) == args.num_pods else 1


if __name__ == "__main__":
    sys.exit(main())
