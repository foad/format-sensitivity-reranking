"""Tests for fsr.corpus.manifest."""

from __future__ import annotations

import hashlib
import json

import pytest

from fsr.corpus.config import DEFAULT
from fsr.corpus.manifest import (
    SCHEMA_VERSION,
    UNKNOWN_VERSION,
    Manifest,
    Output,
    Stage,
    digest_file,
    package_version,
)

CONFIG = DEFAULT.as_dict()


def written(tmp_path, name="out.json", payload=b'{"records": []}'):
    """Write a file and return its path."""
    path = tmp_path / name
    path.write_bytes(payload)
    return path


def manifest_with(stage="fetch", records=3):
    """Return a manifest carrying one recorded stage."""
    m = Manifest(config=CONFIG)
    m.record(stage, [Output(path="nq/a.json", sha256="ab", bytes=2, records=records)])
    return m


class TestDigestFile:
    def test_matches_hashlib(self, tmp_path):
        path = written(tmp_path, payload=b"content")
        assert digest_file(path) == hashlib.sha256(b"content").hexdigest()

    def test_digests_a_file_larger_than_one_chunk(self, tmp_path):
        payload = b"x" * (1 << 21)
        path = written(tmp_path, payload=payload)
        assert digest_file(path) == hashlib.sha256(payload).hexdigest()

    def test_digests_an_empty_file(self, tmp_path):
        assert (
            digest_file(written(tmp_path, payload=b"")) == hashlib.sha256().hexdigest()
        )

    def test_differs_for_differing_content(self, tmp_path):
        a = written(tmp_path, "a.json", b"one")
        b = written(tmp_path, "b.json", b"two")
        assert digest_file(a) != digest_file(b)


class TestPackageVersion:
    def test_returns_a_string(self):
        assert isinstance(package_version(), str)

    def test_falls_back_when_the_package_is_absent(self, monkeypatch):
        import fsr.corpus.manifest as mod

        def missing(_name):
            raise mod.PackageNotFoundError

        monkeypatch.setattr(mod, "version", missing)
        assert mod.package_version() == UNKNOWN_VERSION


class TestOutput:
    def test_describes_a_written_file(self, tmp_path):
        nested = tmp_path / "nq"
        nested.mkdir()
        path = written(nested, "matched_train.json", b"payload")
        out = Output.describe(path, tmp_path, records=7)
        assert out.path == "nq/matched_train.json"
        assert out.sha256 == hashlib.sha256(b"payload").hexdigest()
        assert out.bytes == len(b"payload")
        assert out.records == 7

    def test_records_are_optional(self, tmp_path):
        assert Output.describe(written(tmp_path), tmp_path).records is None


class TestRecord:
    def test_appends_a_stage(self):
        m = manifest_with()
        assert [s.name for s in m.stages] == ["fetch"]

    def test_appends_further_stages_in_order(self):
        m = manifest_with()
        m.record("parse", [])
        m.record("split", [])
        assert [s.name for s in m.stages] == ["fetch", "parse", "split"]

    def test_replaces_a_repeated_stage(self):
        m = manifest_with(records=3)
        m.record("fetch", [Output(path="nq/a.json", sha256="cd", bytes=4, records=9)])
        assert len(m.stages) == 1
        assert m.stage("fetch").outputs[0].records == 9

    def test_a_replaced_stage_keeps_its_position(self):
        m = manifest_with()
        m.record("parse", [])
        m.record("fetch", [])
        assert [s.name for s in m.stages] == ["fetch", "parse"]


class TestLookup:
    def test_finds_a_recorded_stage(self):
        assert manifest_with().stage("fetch").name == "fetch"

    def test_returns_none_for_an_absent_stage(self):
        assert manifest_with().stage("split") is None

    def test_reports_whether_a_stage_has_run(self):
        m = manifest_with()
        assert m.has("fetch")
        assert not m.has("split")


class TestSerialisation:
    def test_round_trips(self, tmp_path):
        m = manifest_with()
        m.save(tmp_path / "manifest.json")
        assert Manifest.load(tmp_path / "manifest.json").to_dict() == m.to_dict()

    def test_writes_indented_json_with_a_trailing_newline(self, tmp_path):
        path = tmp_path / "manifest.json"
        manifest_with().save(path)
        text = path.read_text()
        assert text.endswith("\n")
        assert "\n  " in text

    def test_creates_parent_directories(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "manifest.json"
        manifest_with().save(path)
        assert path.exists()

    def test_carries_the_config(self, tmp_path):
        path = tmp_path / "manifest.json"
        manifest_with().save(path)
        assert json.loads(path.read_text())["config"]["split_seed"] == 42

    def test_carries_no_timestamp(self, tmp_path):
        path = tmp_path / "manifest.json"
        manifest_with().save(path)
        text = path.read_text().lower()
        assert "time" not in text and "date" not in text

    def test_two_identical_builds_produce_identical_files(self, tmp_path):
        first, second = tmp_path / "a.json", tmp_path / "b.json"
        manifest_with().save(first)
        manifest_with().save(second)
        assert first.read_bytes() == second.read_bytes()

    def test_rejects_an_unsupported_schema(self):
        data = manifest_with().to_dict()
        data["schema"] = SCHEMA_VERSION + 1
        with pytest.raises(ValueError, match="unsupported manifest schema"):
            Manifest.from_dict(data)

    def test_tolerates_a_missing_version_field(self):
        data = manifest_with().to_dict()
        del data["fsr_version"]
        assert Manifest.from_dict(data).fsr_version == UNKNOWN_VERSION

    def test_restores_output_fields(self):
        restored = Manifest.from_dict(manifest_with().to_dict())
        out = restored.stage("fetch").outputs[0]
        assert (out.path, out.sha256, out.bytes, out.records) == (
            "nq/a.json",
            "ab",
            2,
            3,
        )


class TestLoadOrNew:
    def test_starts_a_manifest_when_the_file_is_absent(self, tmp_path):
        m = Manifest.load_or_new(tmp_path / "absent.json", CONFIG)
        assert m.stages == []
        assert m.config == CONFIG

    def test_reads_an_existing_manifest(self, tmp_path):
        path = tmp_path / "manifest.json"
        manifest_with().save(path)
        assert Manifest.load_or_new(path, CONFIG).has("fetch")

    def test_an_existing_manifest_keeps_its_own_config(self, tmp_path):
        path = tmp_path / "manifest.json"
        m = Manifest(config={"split_seed": 7})
        m.save(path)
        assert Manifest.load_or_new(path, CONFIG).config == {"split_seed": 7}


class TestStage:
    def test_defaults_to_no_outputs(self):
        assert Stage(name="fetch").outputs == []
