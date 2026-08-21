"""Stable, platform-independent fingerprints for FinRL training data."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import date
from datetime import datetime
from datetime import timezone
from decimal import Decimal, InvalidOperation
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from filelock import FileLock


LEGACY_TRAINING_DATA_FINGERPRINT_SCHEMA = "market_source_rows_v1"
TRAINING_DATA_FINGERPRINT_SCHEMA = "market_source_topology_v2"
SYNTHETIC_PERIOD_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}_to_\d{4}-\d{2}-\d{2}$")
FINGERPRINT_RESULT_CACHE_SCHEMA = "training_fingerprint_result_v1"
RAW_FILE_SET_FINGERPRINT_SCHEMA = "raw_file_set_v1"


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _request_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_json_bytes(dict(payload))).hexdigest()


def _stat_signature(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return (
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )


class SourceDigestRegistry:
    """Hash each immutable source once and reject mid-suite mutations."""

    def __init__(self) -> None:
        self._records: dict[Path, tuple[tuple[int, int, int, int, int], str]] = {}
        self.files_hashed = 0
        self.bytes_hashed = 0
        self.memory_hits = 0

    def digest(self, path: str | Path) -> str:
        resolved = Path(path).resolve()
        signature = _stat_signature(resolved)
        existing = self._records.get(resolved)
        if existing is not None:
            if existing[0] != signature:
                raise RuntimeError(
                    f"Source file changed during experiment suite: {resolved}"
                )
            self.memory_hits += 1
            return existing[1]

        digest = hashlib.sha256()
        size = 0
        with resolved.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                size += len(chunk)
                digest.update(chunk)
        if _stat_signature(resolved) != signature:
            raise RuntimeError(
                f"Source file changed while it was being hashed: {resolved}"
            )
        value = digest.hexdigest()
        self._records[resolved] = (signature, value)
        self.files_hashed += 1
        self.bytes_hashed += size
        return value

    def assert_unchanged(self, path: str | Path) -> None:
        resolved = Path(path).resolve()
        existing = self._records.get(resolved)
        if existing is not None and existing[0] != _stat_signature(resolved):
            raise RuntimeError(
                f"Source file changed during experiment suite: {resolved}"
            )

    def statistics(self) -> dict[str, int]:
        return {
            "raw_files_hashed": self.files_hashed,
            "raw_bytes_hashed": self.bytes_hashed,
            "raw_digest_memory_hits": self.memory_hits,
        }


class _FingerprintTrieNode:
    def __init__(
        self,
        *,
        path: Path | None = None,
        source_kind: str | None = None,
        header: bytes = b"",
    ) -> None:
        self.path = path
        self.source_kind = source_kind
        self.header = header
        self.children: dict[tuple[Any, ...], "_FingerprintTrieNode"] = {}
        self.result_names: list[str] = []


class TrainingFingerprintStore:
    """Suite-local memoization backed by an optional content-addressed store."""

    def __init__(
        self,
        cache_root: str | Path | None,
        *,
        digest_registry: SourceDigestRegistry | None = None,
    ) -> None:
        self.cache_root = Path(cache_root).resolve() if cache_root is not None else None
        self.digest_registry = digest_registry or SourceDigestRegistry()
        self._memory: dict[str, str] = {}
        self.memory_hits = 0
        self.persistent_hits = 0
        self.semantic_computations = 0
        self.legacy_computations = 0
        self.raw_file_set_computations = 0
        self.canonical_source_reads = 0

    def _entry_path(self, request_key: str) -> Path | None:
        if self.cache_root is None:
            return None
        return self.cache_root / request_key[:2] / f"{request_key}.json"

    def _read_entry(
        self, request_key: str, request: Mapping[str, Any]
    ) -> str | None:
        memory = self._memory.get(request_key)
        if memory is not None:
            self.memory_hits += 1
            return memory
        path = self._entry_path(request_key)
        if path is None or not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
        value = payload.get("fingerprint")
        if (
            payload.get("cache_schema") != FINGERPRINT_RESULT_CACHE_SCHEMA
            or payload.get("request_key") != request_key
            or payload.get("request") != request
            or not isinstance(value, str)
            or not re.fullmatch(r"[0-9a-f]{64}", value)
        ):
            return None
        self._memory[request_key] = value
        self.persistent_hits += 1
        return value

    def _write_entry(
        self,
        request_key: str,
        request: Mapping[str, Any],
        fingerprint: str,
    ) -> None:
        self._memory[request_key] = fingerprint
        path = self._entry_path(request_key)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = Path(str(path) + ".lock")
        with FileLock(lock_path):
            if self._read_persistent_entry(path, request_key, request) is not None:
                return
            payload = {
                "cache_schema": FINGERPRINT_RESULT_CACHE_SCHEMA,
                "request_key": request_key,
                "request": dict(request),
                "fingerprint": fingerprint,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
            os.replace(temporary, path)

    @staticmethod
    def _read_persistent_entry(
        path: Path, request_key: str, request: Mapping[str, Any]
    ) -> str | None:
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
        value = payload.get("fingerprint")
        if (
            payload.get("cache_schema") == FINGERPRINT_RESULT_CACHE_SCHEMA
            and payload.get("request_key") == request_key
            and payload.get("request") == request
            and isinstance(value, str)
            and re.fullmatch(r"[0-9a-f]{64}", value)
        ):
            return value
        return None

    def _semantic_request(
        self,
        sources: Sequence[tuple[str, Path]],
        start: str,
        end: str,
    ) -> tuple[dict[str, Any], list[tuple[tuple[Any, ...], Path, str, bytes]]]:
        slots: dict[str, int] = {}
        descriptors: list[dict[str, Any]] = []
        steps: list[tuple[tuple[Any, ...], Path, str, bytes]] = []
        for raw_kind, raw_path in sources:
            source_kind = str(raw_kind)
            path = Path(raw_path)
            slot = slots.get(source_kind, 0)
            slots[source_kind] = slot + 1
            content_sha256 = self.digest_registry.digest(path)
            descriptor: dict[str, Any] = {
                "kind": source_kind,
                "slot": slot,
                "content_sha256": content_sha256,
            }
            if source_kind == "real":
                descriptor["real_ticker"] = path.stem.upper()
            header = _json_bytes({"kind": source_kind, "slot": slot})
            token = (
                source_kind,
                slot,
                content_sha256,
                descriptor.get("real_ticker"),
            )
            descriptors.append(descriptor)
            steps.append((token, path, source_kind, header))
        request = {
            "cache_schema": FINGERPRINT_RESULT_CACHE_SCHEMA,
            "fingerprint_kind": "semantic",
            "fingerprint_schema": TRAINING_DATA_FINGERPRINT_SCHEMA,
            "start": start,
            "end": end,
            "sources": descriptors,
        }
        return request, steps

    def _legacy_request(
        self,
        project_root: Path,
        sources: Sequence[tuple[str, Path]],
        start: str,
        end: str,
    ) -> tuple[dict[str, Any], list[tuple[tuple[Any, ...], Path, str, bytes]]]:
        normalized = sorted(
            (
                str(source_kind),
                Path(path),
                _source_identity(Path(path), project_root),
            )
            for source_kind, path in sources
        )
        descriptors: list[dict[str, Any]] = []
        steps: list[tuple[tuple[Any, ...], Path, str, bytes]] = []
        for source_kind, path, identity in normalized:
            content_sha256 = self.digest_registry.digest(path)
            descriptor = {
                "kind": source_kind,
                "source": identity,
                "content_sha256": content_sha256,
            }
            header = _json_bytes({"kind": source_kind, "source": identity})
            token = (source_kind, identity, content_sha256)
            descriptors.append(descriptor)
            steps.append((token, path, source_kind, header))
        request = {
            "cache_schema": FINGERPRINT_RESULT_CACHE_SCHEMA,
            "fingerprint_kind": "legacy",
            "fingerprint_schema": LEGACY_TRAINING_DATA_FINGERPRINT_SCHEMA,
            "start": start,
            "end": end,
            "sources": descriptors,
        }
        return request, steps

    def semantic_many(
        self,
        topologies: Mapping[str, Sequence[tuple[str, Path]]],
        *,
        start: str,
        end: str,
    ) -> dict[str, str]:
        return self._resolve_many(
            kind="semantic",
            schema=TRAINING_DATA_FINGERPRINT_SCHEMA,
            requests={
                name: self._semantic_request(tuple(sources), start, end)
                for name, sources in topologies.items()
            },
            start=start,
            end=end,
        )

    def legacy_many(
        self,
        project_root: str | Path,
        topologies: Mapping[str, Sequence[tuple[str, Path]]],
        *,
        start: str,
        end: str,
    ) -> dict[str, str]:
        root = Path(project_root)
        return self._resolve_many(
            kind="legacy",
            schema=LEGACY_TRAINING_DATA_FINGERPRINT_SCHEMA,
            requests={
                name: self._legacy_request(root, tuple(sources), start, end)
                for name, sources in topologies.items()
            },
            start=start,
            end=end,
        )

    def _resolve_many(
        self,
        *,
        kind: str,
        schema: str,
        requests: Mapping[
            str,
            tuple[dict[str, Any], list[tuple[tuple[Any, ...], Path, str, bytes]]],
        ],
        start: str,
        end: str,
    ) -> dict[str, str]:
        if not requests:
            return {}
        results: dict[str, str] = {}
        missing: dict[
            str,
            tuple[
                str,
                dict[str, Any],
                list[tuple[tuple[Any, ...], Path, str, bytes]],
            ],
        ] = {}
        equivalent_request_names: dict[str, list[str]] = {}
        for name, (request, steps) in requests.items():
            if not steps:
                raise ValueError("At least one training-data source is required")
            request_key = _request_hash(request)
            value = self._read_entry(request_key, request)
            if value is not None:
                results[name] = value
                continue
            equivalent_request_names.setdefault(request_key, []).append(name)
            missing.setdefault(request_key, (request_key, request, steps))

        if missing:
            computed = self._compute_missing_batch(
                missing,
                schema=schema,
                start=date.fromisoformat(start),
                end=date.fromisoformat(end),
            )
            for request_key, (_, request, _) in missing.items():
                value = computed[request_key]
                self._write_entry(request_key, request, value)
                for name in equivalent_request_names[request_key]:
                    results[name] = value
            if kind == "semantic":
                self.semantic_computations += len(missing)
            else:
                self.legacy_computations += len(missing)
        return results

    def _compute_missing_batch(
        self,
        requests: Mapping[
            str,
            tuple[
                str,
                dict[str, Any],
                list[tuple[tuple[Any, ...], Path, str, bytes]],
            ],
        ],
        *,
        schema: str,
        start: date,
        end: date,
    ) -> dict[str, str]:
        root = _FingerprintTrieNode()
        for request_key, (_, _, steps) in requests.items():
            node = root
            for token, path, source_kind, header in steps:
                node = node.children.setdefault(
                    token,
                    _FingerprintTrieNode(
                        path=path,
                        source_kind=source_kind,
                        header=header,
                    ),
                )
            node.result_names.append(request_key)

        results: dict[str, str] = {}
        initial = hashlib.sha256()
        initial.update((schema + "\n").encode())

        def visit(node: _FingerprintTrieNode, digest: Any) -> None:
            for child in node.children.values():
                child_digest = digest.copy()
                child_digest.update(child.header)
                child_digest.update(b"\n")
                assert child.path is not None and child.source_kind is not None
                self.canonical_source_reads += 1
                for row in _canonical_source_rows(
                    child.path, child.source_kind, start, end
                ):
                    child_digest.update(json.dumps(row, separators=(",", ":")).encode())
                    child_digest.update(b"\n")
                value = child_digest.hexdigest()
                for request_key in child.result_names:
                    results[request_key] = value
                visit(child, child_digest)

        visit(root, initial)
        return results

    def raw_file_set(
        self,
        project_root: str | Path,
        paths: Iterable[str | Path],
    ) -> str:
        root = Path(project_root).resolve()
        ordered = sorted(Path(path) for path in paths)
        if not ordered:
            raise ValueError("At least one raw source file is required")
        descriptors = []
        for path in ordered:
            try:
                identity = path.resolve().relative_to(root).as_posix()
            except ValueError:
                identity = path.name
            descriptors.append(
                {
                    "identity": identity,
                    "content_sha256": self.digest_registry.digest(path),
                }
            )
        request = {
            "cache_schema": FINGERPRINT_RESULT_CACHE_SCHEMA,
            "fingerprint_kind": "raw_file_set",
            "fingerprint_schema": RAW_FILE_SET_FINGERPRINT_SCHEMA,
            "sources": descriptors,
        }
        request_key = _request_hash(request)
        value = self._read_entry(request_key, request)
        if value is not None:
            return value
        digest = hashlib.sha256()
        for path, descriptor in zip(ordered, descriptors):
            digest.update(str(descriptor["identity"]).encode())
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            self.digest_registry.assert_unchanged(path)
        value = digest.hexdigest()
        self.raw_file_set_computations += 1
        self._write_entry(request_key, request, value)
        return value

    def statistics(self) -> dict[str, int]:
        return {
            **self.digest_registry.statistics(),
            "fingerprint_memory_hits": self.memory_hits,
            "fingerprint_persistent_hits": self.persistent_hits,
            "semantic_fingerprint_computations": self.semantic_computations,
            "legacy_fingerprint_computations": self.legacy_computations,
            "raw_file_set_computations": self.raw_file_set_computations,
            "canonical_source_reads": self.canonical_source_reads,
        }


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
        relative = path.resolve().relative_to(project_root.resolve())
    except ValueError:
        # External data roots remain portable across machines.
        return path.name
    parts = relative.parts
    for index in range(len(parts) - 2):
        if parts[index : index + 2] == (
            "data",
            "synthetic",
        ) and SYNTHETIC_PERIOD_PATTERN.fullmatch(parts[index + 2]):
            # The generated-period directory is data catalog structure, not
            # training content. A further directory means the optional named
            # ticker-group layer. Omitting these catalog layers preserves cache
            # compatibility across both supported layouts.
            has_ticker_group = len(parts) - (index + 3) >= 3
            resume_at = index + 4 if has_ticker_group else index + 3
            parts = (*parts[: index + 2], *parts[resume_at:])
            break
    return Path(*parts).as_posix()


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
    digest.update((LEGACY_TRAINING_DATA_FINGERPRINT_SCHEMA + "\n").encode())
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


def fingerprint_semantic_training_sources(
    *,
    sources: Iterable[tuple[str, Path]],
    start: str,
    end: str,
) -> str:
    """Hash the ordered data topology without filesystem or model identities.

    Source order remains significant because a seeded ``MultiPathEnv`` samples
    path indexes.  Moving an equivalent ordered dataset to another directory,
    however, must not change its training identity.
    """

    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end)
    normalized_sources = [(str(kind), Path(path)) for kind, path in sources]
    if not normalized_sources:
        raise ValueError("At least one training-data source is required")

    slots: dict[str, int] = {}
    digest = hashlib.sha256()
    digest.update((TRAINING_DATA_FINGERPRINT_SCHEMA + "\n").encode())
    for source_kind, path in normalized_sources:
        slot = slots.get(source_kind, 0)
        slots[source_kind] = slot + 1
        digest.update(
            json.dumps(
                {"kind": source_kind, "slot": slot},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        digest.update(b"\n")
        for row in _canonical_source_rows(path, source_kind, start_date, end_date):
            digest.update(json.dumps(row, separators=(",", ":")).encode())
            digest.update(b"\n")
    return digest.hexdigest()
