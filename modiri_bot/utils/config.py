"""Load config/config.yaml and .env into plain Python objects."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv(REPO_ROOT / ".env")


def load_config(path: Path | str = CONFIG_PATH) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@dataclass(frozen=True)
class SymbolConfig:
    name: str
    timeframe: str
    pip_size: float
    pip_value_per_lot: float
    contract_size: float
    min_lot: float
    lot_step: float
    spread_pips: float
    commission_per_lot: float

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SymbolConfig":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__})


def get_symbol_config(cfg: dict[str, Any], symbol_name: str) -> SymbolConfig:
    for entry in cfg["symbols"]:
        if entry["name"] == symbol_name:
            return SymbolConfig.from_dict(entry)
    raise KeyError(f"Symbol '{symbol_name}' not found in config.yaml")


def mt5_credentials() -> dict[str, Any]:
    return {
        "login": int(os.environ["MT5_LOGIN"]),
        "password": os.environ["MT5_PASSWORD"],
        "server": os.environ["MT5_SERVER"],
        "path": os.environ.get("MT5_TERMINAL_PATH") or None,
    }
