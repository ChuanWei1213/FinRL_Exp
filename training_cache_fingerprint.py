"""Stable, platform-independent fingerprints for FinRL training data."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable


TRAINING_DATA_FINGERPRINT_SCHEMA = "market_source_rows_v1"


def _canonical_decimal(value: str, *, path: Path, column: str) -> str:
    try:
        number = Decimal(value.strip())
    except InvalidOperation as exc:
        raise ValueError(
            f"{path}: invalid decimal value in {column!r}: {value!r}"
        ) from exc
    if not number.is_finite():
        raise ValueError(f"{path}: non-finite value in {column!r}: {value!r}")
    if number == 0:
        return "0"
    return str(number.normalize())


def _source_identity(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        # External data roots remain portable across machines.
        return path.name


def _canonical_source_rows(
    path: Path,
    source_kind: str,
    start: date,
    end: date,
) -> list[tuple[str, ...]]:
    if source_kind == "real":
        required = ("date", "open", "high", "low", "close")
    elif source_kind == "synthetic":
        required = ("date", "tic", "close", "high", "low")
    else:
        raise ValueError(f"Unsupported training-data source kind: {source_kind!r}")

    rows: list[tuple[str, ...]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        field_lookup = {
            field.strip().lower(): field
            for field in (reader.fieldnames or [])
            if field is not None
        }
        missing = [field for field in required if field not in field_lookup]
        if missing:
            raise ValueError(f"{path}: missing required columns {missing}")

        for record in reader:
            raw_date = record[field_lookup["date"]].strip()
            try:
                row_date = date.fromisoformat(raw_date[:10])
            except ValueError as exc:
                raise ValueError(f"{path}: invalid date value {raw_date!r}") from exc
            if not start <= row_date < end:
                continue

            date_value = row_date.isoformat()
            if source_kind == "real":
                rows.append(
                    (
                        date_value,
                        path.stem.upper(),
                        *(
                            _canonical_decimal(
                                record[field_lookup[column]],
                                path=path,
                                column=column,
                            )
                            for column in ("open", "high", "low", "close")
                        ),
                    )
                )
            else:
                rows.append(
                    (
                        date_value,
                        record[field_lookup["tic"]].strip().upper(),
                        *(
                            _canonical_decimal(
                                record[field_lookup[column]],
                                path=path,
                                column=column,
                            )
                            for column in ("close", "high", "low")
                        ),
                    )
                )

    if not rows:
        raise ValueError(
            f"{path}: no rows in training fingerprint window [{start}, {end})"
        )
    return sorted(rows)


def fingerprint_training_sources(
    *,
    project_root: Path,
    sources: Iterable[tuple[str, Path]],
    start: str,
    end: str,
) -> str:
    """Hash canonical source values used by training or validation.

    Decimal source values are canonicalized before hashing, avoiding pandas/NumPy
    binary-float serialization differences across operating systems. File order and
    CSV row order do not affect the result, while per-path identity remains part of
    the data topology used by MultiPathEnv.
    """

    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end)
    normalized_sources = sorted(
        (
            source_kind,
            Path(path),
            _source_identity(Path(path), project_root),
        )
        for source_kind, path in sources
    )
    if not normalized_sources:
        raise ValueError("At least one training-data source is required")

    digest = hashlib.sha256()
    digest.update((TRAINING_DATA_FINGERPRINT_SCHEMA + "\n").encode())
    for source_kind, path, identity in normalized_sources:
        digest.update(
            json.dumps(
                {"kind": source_kind, "source": identity},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        digest.update(b"\n")
        for row in _canonical_source_rows(path, source_kind, start_date, end_date):
            digest.update(json.dumps(row, separators=(",", ":")).encode())
            digest.update(b"\n")
    return digest.hexdigest()
