"""RunPod REST API v1 client: pod lifecycle (create/list/get/stop/resume/
terminate) and GPU type listing. Used by scripts/smoke_test.py,
scripts/bootstrap_pod.py, and scripts/poll_status.py -- the three scripts
that need to talk to RunPod, sharing one auth/request implementation rather
than each rolling their own.

VERIFICATION STATUS: this sandbox's network egress proxy hard-blocks
rest.runpod.io and docs.runpod.io (confirmed via direct 403 policy-denial
responses, including to RunPod's own OpenAPI spec endpoint), so the exact
endpoint paths and request/response field names below could not be
confirmed against RunPod's live API reference at write time. They reflect
RunPod's published REST API v1 conventions (Bearer auth, /pods and
/gpuTypes resources) as documented as of this codebase's writing, not a
verified live call. Field names are isolated to _build_create_pod_body()
and the per-method paths below specifically so they're easy to correct in
one place once network access is available -- which must happen before
the first real pod-creation call (see the smoke-test gate in
scripts/bootstrap_pod.py and run_pipeline's plan).
"""

from __future__ import annotations

import requests

API_BASE = "https://rest.runpod.io/v1"
USER_AGENT = "podcast-speech-dataset-pipeline/1.0"


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
        if resp.status_code not in (200, 201):
            raise RunPodError(f"RunPod {method} {path} returned HTTP {resp.status_code}: {resp.text[:500]}")
        if not resp.content:
            return {}
        return resp.json()

    def list_gpu_types(self) -> list[dict]:
        """Returns available GPU types with id/displayName/pricing fields."""
        data = self._request("GET", "/gpuTypes")
        return data if isinstance(data, list) else data.get("gpuTypes", [])

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
        docker_start_cmd: str | None = None,
        ports: str | None = None,
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

    def resume_pod(self, pod_id: str) -> dict:
        return self._request("POST", f"/pods/{pod_id}/resume")

    def terminate_pod(self, pod_id: str) -> None:
        self._request("DELETE", f"/pods/{pod_id}")

    def check_connectivity(self) -> None:
        """Cheap auth-validating call for scripts/smoke_test.py; raises RunPodError on failure."""
        self.list_gpu_types()
