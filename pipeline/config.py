"""Loads config/pipeline.yaml into a validated, dotted-attribute config object."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "pipeline.yaml"


class ConfigError(ValueError):
    pass


class _Section:
    """Read-only dotted-attribute view over a dict, recursively."""

    def __init__(self, data: dict, path: str = ""):
        self._path = path
        for key, value in data.items():
            if isinstance(value, dict):
                value = _Section(value, f"{path}.{key}" if path else key)
            object.__setattr__(self, key, value)
        self._keys = list(data.keys())

    def __setattr__(self, name, value):
        if name in ("_path", "_keys"):
            object.__setattr__(self, name, value)
            return
        raise AttributeError("config sections are read-only")

    def __getattr__(self, name):
        full = f"{self._path}.{name}" if self._path else name
        raise ConfigError(f"missing config key: {full}")

    def __repr__(self):
        return f"_Section({self._path!r}, keys={self._keys!r})"

    def as_dict(self) -> dict:
        out = {}
        for key in self._keys:
            value = getattr(self, key)
            out[key] = value.as_dict() if isinstance(value, _Section) else value
        return out


REQUIRED_TOP_LEVEL_SECTIONS = (
    "models",
    "audio",
    "vad",
    "segmentation",
    "quality",
    "clustering",
    "cost",
    "target_corpus",
    "parallelism",
    "monitoring",
)


@dataclass(frozen=True)
class EnvSecrets:
    podcastindex_api_key: str
    podcastindex_api_secret: str
    runpod_api_key: str
    r2_account_id: str
    r2_access_key_id: str
    r2_secret_access_key: str
    r2_bucket_name: str
    hf_token: str
    budget_cap_usd: float
    time_cap_hours: float

    @classmethod
    def from_env(cls, env: dict | None = None) -> "EnvSecrets":
        env = env if env is not None else os.environ
        return cls(
            podcastindex_api_key=env.get("PODCASTINDEX_API_KEY", ""),
            podcastindex_api_secret=env.get("PODCASTINDEX_API_SECRET", ""),
            runpod_api_key=env.get("RUNPOD_API_KEY", ""),
            r2_account_id=env.get("R2_ACCOUNT_ID", ""),
            r2_access_key_id=env.get("R2_ACCESS_KEY_ID", ""),
            r2_secret_access_key=env.get("R2_SECRET_ACCESS_KEY", ""),
            r2_bucket_name=env.get("R2_BUCKET_NAME", ""),
            hf_token=env.get("HF_TOKEN", ""),
            budget_cap_usd=float(env.get("BUDGET_CAP_USD", "100")),
            time_cap_hours=float(env.get("TIME_CAP_HOURS", "24")),
        )

    def require_for_network_ops(self) -> None:
        missing = [
            name
            for name, value in (
                ("PODCASTINDEX_API_KEY", self.podcastindex_api_key),
                ("PODCASTINDEX_API_SECRET", self.podcastindex_api_secret),
                ("RUNPOD_API_KEY", self.runpod_api_key),
                ("R2_ACCOUNT_ID", self.r2_account_id),
                ("R2_ACCESS_KEY_ID", self.r2_access_key_id),
                ("R2_SECRET_ACCESS_KEY", self.r2_secret_access_key),
                ("R2_BUCKET_NAME", self.r2_bucket_name),
                ("HF_TOKEN", self.hf_token),
            )
            if not value
        ]
        if missing:
            raise ConfigError(f"missing required environment variables: {', '.join(missing)}")


def load_config(path: str | Path | None = None) -> _Section:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ConfigError(f"config file did not parse to a mapping: {path}")
    missing = [key for key in REQUIRED_TOP_LEVEL_SECTIONS if key not in data]
    if missing:
        raise ConfigError(f"config file missing required sections: {', '.join(missing)}")
    return _Section(data)
