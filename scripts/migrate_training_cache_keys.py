#!/usr/bin/env python3
"""Migrate PPO training-cache metadata to stable source-data fingerprints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from copy import deepcopy
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training_cache_fingerprint import (  # noqa: E402
    TRAINING_DATA_FINGERPRINT_SCHEMA,
    fingerprint_training_sources,
)


TRAIN_CACHE_SCHEMA = "ppo_eiie_training_cache_v3"
TRAIN_START = "2024-01-01"
VAL_END = "2025-01-01"


def content_hash(payload: dict) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def semantic_ppo_config(config: dict) -> dict:
    semantic = deepcopy(config)
    semantic.pop("device", None)
    semantic.pop("verbose", None)
    policy_kwargs = semantic.get("policy_kwargs", {})
    extractor = policy_kwargs.get("features_extractor_class")
    if isinstance(extractor, str):
        policy_kwargs["features_extractor_class"] = extractor.rsplit(".", 1)[-1]
    return semantic


def semantic_cache_config(status: dict) -> dict | None:
    semantic = status.get("semantic_cache_config")
    if isinstance(semantic, dict):
        return deepcopy(semantic)

    legacy = status.get("training_cache_config")
    if not isinstance(legacy, dict):
        return None
    semantic = deepcopy(legacy)
    semantic.pop("schema", None)
    semantic.pop("library_versions", None)
    semantic.pop("commission_name", None)
    environment = semantic.get("environment", {})
    environment.pop("cwd", None)
    environment.pop("plot_on_terminal", None)
    semantic["ppo"] = semantic_ppo_config(semantic.get("ppo", {}))
    semantic["algorithm"] = "PPO"
    semantic["policy"] = "MultiInputPolicy"
    return semantic


def discover_group_fingerprints(project_root: Path) -> dict[str, str]:
    real_files = sorted((project_root / "data" / "real").glob("*.csv"))
    if not real_files:
        raise FileNotFoundError("No real CSV files found under data/real")
    real_sources = [("real", path) for path in real_files]
    fingerprints = {
        "real_trained": fingerprint_training_sources(
            project_root=project_root,
            sources=real_sources,
            start=TRAIN_START,
            end=VAL_END,
        )
    }

    synthetic_root = project_root / "data" / "synthetic"
    for model_dir in sorted(
        path for path in synthetic_root.iterdir() if path.is_dir()
    ):
        paths = sorted(model_dir.glob("path*.csv"))
        if not paths:
            continue
        fingerprints[f"synthetic::{model_dir.name}"] = fingerprint_training_sources(
            project_root=project_root,
            sources=[*real_sources, *(("synthetic", path) for path in paths)],
            start=TRAIN_START,
            end=VAL_END,
        )
    return fingerprints


def atomic_write_json(path: Path, payload: dict) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary_path = Path(handle.name)
    try:
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def migrate_status(
    status_path: Path,
    fingerprints: dict[str, str],
) -> tuple[dict, bool, str]:
    status = json.loads(status_path.read_text())
    group = status.get("group")
    if group not in fingerprints:
        raise KeyError(f"{status_path}: no data fingerprint for group {group!r}")
    semantic = semantic_cache_config(status)
    if semantic is None:
        raise ValueError(f"{status_path}: no semantic or legacy cache config")

    semantic["training_data_fingerprint"] = fingerprints[group]
    new_key = content_hash(semantic)
    old_key = status.get("cache_key")
    old_schema = status.get("cache_schema")
    old_fingerprint_schema = status.get("training_data_fingerprint_schema")

    legacy_keys = list(status.get("legacy_cache_keys", []))
    existing_legacy_key = status.get("legacy_cache_key")
    if existing_legacy_key:
        legacy_keys.append(existing_legacy_key)
    if old_key and old_key != new_key:
        legacy_keys.append(old_key)
    legacy_keys = list(dict.fromkeys(legacy_keys))

    status["cache_key"] = new_key
    status["cache_schema"] = TRAIN_CACHE_SCHEMA
    status["semantic_cache_config"] = semantic
    status["training_data_fingerprint"] = fingerprints[group]
    status["training_data_fingerprint_schema"] = TRAINING_DATA_FINGERPRINT_SCHEMA
    if legacy_keys:
        status["legacy_cache_keys"] = legacy_keys
        status["legacy_cache_key"] = legacy_keys[-1]
    if old_schema and old_schema != TRAIN_CACHE_SCHEMA:
        status.setdefault("migrated_from_cache_schema", old_schema)

    changed = (
        old_key != new_key
        or old_schema != TRAIN_CACHE_SCHEMA
        or old_fingerprint_schema != TRAINING_DATA_FINGERPRINT_SCHEMA
    )
    return status, changed, new_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "ppo_synthetic_vs_real" / "train_cache",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write migrated metadata. Without this flag, perform a dry run.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    cache_root = args.cache_root.resolve()
    fingerprints = discover_group_fingerprints(project_root)
    status_paths = sorted(cache_root.rglob("status.json"))
    if not status_paths:
        raise FileNotFoundError(f"No status.json files found under {cache_root}")

    changed_count = 0
    key_counts: Counter[str] = Counter()
    group_counts: Counter[str] = Counter()
    for status_path in status_paths:
        migrated, changed, new_key = migrate_status(status_path, fingerprints)
        key_counts[new_key] += 1
        group_counts[str(migrated.get("group"))] += 1
        if changed:
            changed_count += 1
            if args.write:
                atomic_write_json(status_path, migrated)

    action = "updated" if args.write else "would update"
    duplicate_entries = sum(count - 1 for count in key_counts.values() if count > 1)
    print(f"fingerprint schema: {TRAINING_DATA_FINGERPRINT_SCHEMA}")
    print(
        "group fingerprints:",
        json.dumps(
            {group: value[:12] for group, value in fingerprints.items()},
            sort_keys=True,
        ),
    )
    print(f"{action}: {changed_count}/{len(status_paths)} status files")
    print("status groups:", json.dumps(group_counts, sort_keys=True))
    print(f"redundant entries sharing a semantic key: {duplicate_entries}")
    print("cache directories were intentionally left in place")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
