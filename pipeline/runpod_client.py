"""RunPod REST API v1 client: pod lifecycle (create/list/get/stop/start/
terminate). Used by scripts/smoke_test.py and scripts/bootstrap_pod.py --
the scripts that need to talk to RunPod, sharing one auth/request
implementation rather than each rolling their own.

VERIFICATION STATUS: confirmed live against
https://rest.runpod.io/v1/openapi.json (reachable from this environment,
no auth required to fetch the spec itself). There is no /gpuTypes (or
/gpu-types) resource in the real v1 API -- gpuTypeIds takes one of the
display-name strings enumerated in the PodCreateInput schema (e.g.
"NVIDIA GeForce RTX 4090", "NVIDIA A40"); GPU_TYPE_IDS below is that enum,
kept here so callers don't need to fetch the spec themselves. Pod
lifecycle is POST /pods, GET/PATCH/DELETE /pods/{podId}, and
POST /pods/{podId}/{stop,start,restart,reset,update} -- there is no
"resume" verb (the real path is "start"). dockerStartCmd/dockerEntrypoint
are arrays of argv tokens, not a single shell string.
"""

from __future__ import annotations

import requests

API_BASE = "https://rest.runpod.io/v1"
USER_AGENT = "podcast-speech-dataset-pipeline/1.0"

# A sample of the gpuTypeIds enum from PodCreateInput (live openapi.json,
# https://rest.runpod.io/v1/openapi.json) -- not exhaustive, just enough
# known-good ids to pick from without re-fetching the spec at runtime.
GPU_TYPE_IDS = (
    "NVIDIA GeForce RTX 3090",  # PLAN.md's locked-in compute choice
    "NVIDIA GeForce RTX 4090",
    "NVIDIA A40",
    "NVIDIA RTX A5000",
    "NVIDIA GeForce RTX 5090",
    "NVIDIA H100 80GB HBM3",
)


class RunPodError(RuntimeError):
    pass


class RunPodClient:
    def __init__(self, api_key: str, session: requests.Session | None = None, timeout: float = 30.0):
        if not api_key:
            raise RunPodError("RunPod API key required")
        self._api_key = api_key
        self._session = session or requests.Session()
        self._timeout = timeout

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        }

    def _request(self, method: str, path: str, json_body: dict | None = None) -> dict | list:
        url = f"{API_BASE}{path}"
        resp = self._session.request(
            method, url, headers=self._headers(), json=json_body, timeout=self._timeout
        )
        if resp.status_code not in (200, 201, 204):
            raise RunPodError(f"RunPod {method} {path} returned HTTP {resp.status_code}: {resp.text[:500]}")
        if not resp.content:
            return {}
        return resp.json()

    def list_pods(self) -> list[dict]:
        data = self._request("GET", "/pods")
        return data if isinstance(data, list) else data.get("pods", [])

    def get_pod(self, pod_id: str) -> dict:
        return self._request("GET", f"/pods/{pod_id}")

    def create_pod(
        self,
        *,
        name: str,
        image_name: str,
        gpu_type_id: str,
        cloud_type: str = "COMMUNITY",
        gpu_count: int = 1,
        container_disk_in_gb: int = 20,
        volume_in_gb: int = 0,
        env: dict[str, str] | None = None,
        docker_start_cmd: list[str] | None = None,
        ports: list[str] | None = None,
    ) -> dict:
        body = {
            "name": name,
            "imageName": image_name,
            "gpuTypeIds": [gpu_type_id],
            "gpuCount": gpu_count,
            "cloudType": cloud_type,
            "containerDiskInGb": container_disk_in_gb,
            "volumeInGb": volume_in_gb,
            "env": env or {},
        }
        if docker_start_cmd:
            body["dockerStartCmd"] = docker_start_cmd
        if ports:
            body["ports"] = ports
        return self._request("POST", "/pods", json_body=body)

    def stop_pod(self, pod_id: str) -> dict:
        return self._request("POST", f"/pods/{pod_id}/stop")

    def start_pod(self, pod_id: str) -> dict:
        return self._request("POST", f"/pods/{pod_id}/start")

    def restart_pod(self, pod_id: str) -> dict:
        """Keeps the pod pinned to the same physical machine/disk, unlike
        stop+start or terminate+create which can reschedule onto a different
        host (confirmed in this project: PROBLEMS.md #13). The lowest-risk
        way to redeploy fixed code onto a live pod without losing whatever
        is only on its local container disk."""
        return self._request("POST", f"/pods/{pod_id}/restart")

    def terminate_pod(self, pod_id: str) -> None:
        self._request("DELETE", f"/pods/{pod_id}")

    def check_connectivity(self) -> None:
        """Cheap auth-validating call for scripts/smoke_test.py; raises RunPodError on failure."""
        self.list_pods()
