"""Environment-backed worker settings."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    default_wall_seconds: int = 3000


def load_settings() -> Settings:
    return Settings(default_wall_seconds=int(os.environ.get("DEFAULT_WALL_SECONDS", "3000")))
