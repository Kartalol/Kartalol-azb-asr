"""Small helpers shared by repository CLI entry points."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def load_yaml(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("YAML configs require PyYAML") from exc
    with Path(path).open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError("Configuration root must be a mapping")
    return value


def apply_config_defaults(parser: argparse.ArgumentParser, config: dict[str, Any]) -> None:
    known = {action.dest for action in parser._actions}
    unknown = sorted(set(config) - known)
    if unknown:
        parser.error(f"Unknown config keys: {', '.join(unknown)}")
    parser.set_defaults(**config)


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def write_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(json_safe(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
