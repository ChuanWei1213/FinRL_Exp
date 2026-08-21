from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

import training_cache_fingerprint as fingerprint_module
from training_cache_fingerprint import SourceDigestRegistry
from training_cache_fingerprint import TrainingFingerprintStore
from training_cache_fingerprint import fingerprint_semantic_training_sources
from training_cache_fingerprint import fingerprint_training_sources


START = "2024-01-01"
END = "2024-02-01"


def _write_sources(root: Path) -> tuple[Path, Path, Path]:
    real = root / "data" / "real" / "AAA.csv"
    synthetic_1 = root / "first" / "path_001.csv"
    synthetic_2 = root / "first" / "path_002.csv"
    real.parent.mkdir(parents=True)
    synthetic_1.parent.mkdir(parents=True)
    real.write_text(
        "date,open,high,low,close\n"
        "2024-01-02,10,11,9,10\n"
        "2024-01-03,11,12,10,11\n",
        encoding="utf-8",
    )
    rows = pd.DataFrame(
        [
            {"date": "2024-01-02", "tic": "AAA", "close": 10, "high": 11, "low": 9},
            {"date": "2024-01-03", "tic": "AAA", "close": 11, "high": 12, "low": 10},
        ]
    )
    rows.to_csv(synthetic_1, index=False)
    (rows * {"date": 1, "tic": 1, "close": 2, "high": 2, "low": 2}).to_csv(
        synthetic_2, index=False
    )
    return real, synthetic_1, synthetic_2


def test_batch_fingerprints_are_byte_exact_and_share_prefixes(tmp_path):
    real, synthetic_1, synthetic_2 = _write_sources(tmp_path)
    one = [("real", real), ("synthetic", synthetic_1)]
    two = [*one, ("synthetic", synthetic_2)]
    store = TrainingFingerprintStore(None)

    semantic = store.semantic_many(
        {"one": one, "two": two, "same_as_two": two}, start=START, end=END
    )
    legacy = store.legacy_many(tmp_path, {"two": two}, start=START, end=END)

    assert semantic["one"] == fingerprint_semantic_training_sources(
        sources=one, start=START, end=END
    )
    assert semantic["two"] == fingerprint_semantic_training_sources(
        sources=two, start=START, end=END
    )
    assert semantic["same_as_two"] == semantic["two"]
    assert legacy["two"] == fingerprint_training_sources(
        project_root=tmp_path, sources=two, start=START, end=END
    )
    statistics = store.statistics()
    assert statistics["raw_files_hashed"] == 3
    assert statistics["semantic_fingerprint_computations"] == 2
    assert statistics["canonical_source_reads"] == 6


def test_persistent_hit_does_not_parse_canonical_csv_rows(tmp_path, monkeypatch):
    real, synthetic_1, synthetic_2 = _write_sources(tmp_path)
    sources = [("real", real), ("synthetic", synthetic_1), ("synthetic", synthetic_2)]
    cache_root = tmp_path / "fingerprint_cache"
    cold = TrainingFingerprintStore(cache_root)
    expected_semantic = cold.semantic_many(
        {"training": sources}, start=START, end=END
    )["training"]
    expected_legacy = cold.legacy_many(
        tmp_path, {"training": sources}, start=START, end=END
    )["training"]

    monkeypatch.setattr(
        fingerprint_module,
        "_canonical_source_rows",
        lambda *args, **kwargs: pytest.fail("persistent hit parsed canonical CSV rows"),
    )
    warm = TrainingFingerprintStore(cache_root)

    assert warm.semantic_many({"training": sources}, start=START, end=END) == {
        "training": expected_semantic
    }
    assert warm.legacy_many(
        tmp_path, {"training": sources}, start=START, end=END
    ) == {"training": expected_legacy}
    assert warm.statistics()["fingerprint_persistent_hits"] == 2
    assert warm.statistics()["canonical_source_reads"] == 0


def test_full_content_sha_invalidates_same_size_and_mtime(tmp_path):
    _, synthetic, _ = _write_sources(tmp_path)
    cache_root = tmp_path / "fingerprint_cache"
    sources = [("synthetic", synthetic)]
    first = TrainingFingerprintStore(cache_root).semantic_many(
        {"training": sources}, start=START, end=END
    )["training"]
    original_stat = synthetic.stat()
    original = synthetic.read_text(encoding="utf-8")
    changed = original.replace(",10,11,9\n", ",20,11,9\n", 1)
    assert len(changed) == len(original)
    synthetic.write_text(changed, encoding="utf-8")
    os.utime(synthetic, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    second_store = TrainingFingerprintStore(cache_root)
    second = second_store.semantic_many(
        {"training": sources}, start=START, end=END
    )["training"]

    assert second != first
    assert second_store.statistics()["fingerprint_persistent_hits"] == 0
    assert second_store.statistics()["semantic_fingerprint_computations"] == 1


def test_digest_registry_rejects_source_changed_during_suite(tmp_path):
    _, synthetic, _ = _write_sources(tmp_path)
    registry = SourceDigestRegistry()
    registry.digest(synthetic)
    original = synthetic.read_text(encoding="utf-8")
    synthetic.write_text(original.replace(",10,11,9\n", ",20,11,9\n", 1))

    with pytest.raises(RuntimeError, match="changed during experiment suite"):
        registry.digest(synthetic)


def test_corrupt_persistent_entry_recomputes_safely(tmp_path, monkeypatch):
    _, synthetic, _ = _write_sources(tmp_path)
    cache_root = tmp_path / "fingerprint_cache"
    sources = [("synthetic", synthetic)]
    TrainingFingerprintStore(cache_root).semantic_many(
        {"training": sources}, start=START, end=END
    )
    entry = next(cache_root.glob("*/*.json"))
    entry.write_text("{not-json", encoding="utf-8")
    original_parser = fingerprint_module._canonical_source_rows
    calls = 0

    def tracked_parser(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_parser(*args, **kwargs)

    monkeypatch.setattr(fingerprint_module, "_canonical_source_rows", tracked_parser)
    repaired = TrainingFingerprintStore(cache_root)
    repaired.semantic_many({"training": sources}, start=START, end=END)

    assert calls == 1
    assert repaired.statistics()["semantic_fingerprint_computations"] == 1

