"""
Configuration management for TomeWhisper.

Reads server_config.yaml at repo root, overrides with env vars.
Usage:
    from tome_core.config import config
    print(config.ocr_url)
"""

import os
from pathlib import Path

import yaml


def _find_repo_root() -> Path:
    """Find the repository root (parent of tome_core)."""
    return Path(__file__).parent.parent


def _load_config() -> dict:
    """Load config from YAML + env vars. Returns a flat dict."""
    cfg: dict = {}
    repo_root = _find_repo_root()

    # 1. Load YAML
    yaml_path = repo_root.parent / "server_config.yaml"
    if not yaml_path.exists():
        yaml_path = repo_root / "server_config.yaml"
    if yaml_path.exists():
        try:
            with open(yaml_path) as f:
                raw = yaml.safe_load(f) or {}
            hostname = raw.get("hostname", "")
            if hostname and "@" in hostname:
                cfg["remote_host"] = hostname.split("@")[-1]
            elif hostname:
                cfg["remote_host"] = hostname
            if raw.get("password"):
                cfg["remote_password"] = raw["password"]
        except Exception:
            pass

    # 2. Env var overrides
    for key, env_var in [
        ("remote_host", "TOMEWHISPER_HOST"),
        ("remote_port", "TOMEWHISPER_PORT"),
        ("remote_password", "TOMEWHISPER_PASSWORD"),
        ("default_model", "TOMEWHISPER_MODEL"),
        ("batch_size", "TOMEWHISPER_BATCH_SIZE"),
        ("figures_hires", "TOMEWHISPER_FIGURES_HIRES"),
    ]:
        val = os.environ.get(env_var)
        if val:
            cfg[key] = val

    # 3. Defaults
    cfg.setdefault("remote_host", "192.168.31.156")
    cfg.setdefault("remote_port", 8000)
    cfg.setdefault("default_model", "infly/Infinity-Parser2-Pro")
    cfg.setdefault("batch_size", 16)
    cfg.setdefault("figures_hires", 4096)

    if isinstance(cfg["remote_port"], str):
        cfg["remote_port"] = int(cfg["remote_port"])
    if isinstance(cfg["batch_size"], str):
        cfg["batch_size"] = int(cfg["batch_size"])
    if isinstance(cfg["figures_hires"], str):
        cfg["figures_hires"] = int(cfg["figures_hires"])

    return cfg


class Config:
    """Singleton configuration object."""

    def __init__(self):
        self._cfg = _load_config()

    @property
    def remote_host(self) -> str:
        return self._cfg["remote_host"]

    @property
    def remote_port(self) -> int:
        return self._cfg["remote_port"]

    @property
    def ocr_url(self) -> str:
        return f"http://{self.remote_host}:{self.remote_port}/ocr"

    @property
    def batch_url(self) -> str:
        return f"http://{self.remote_host}:{self.remote_port}/ocr/batch"

    @property
    def health_url(self) -> str:
        return f"http://{self.remote_host}:{self.remote_port}/health"

    @property
    def default_model(self) -> str:
        return self._cfg["default_model"]

    @property
    def remote_password(self) -> str:
        return self._cfg.get("remote_password", "")

    @property
    def batch_size(self) -> int:
        return self._cfg["batch_size"]

    @property
    def figures_hires(self) -> int:
        return self._cfg["figures_hires"]


# Global singleton
config = Config()
