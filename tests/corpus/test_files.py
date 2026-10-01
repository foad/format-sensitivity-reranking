"""Tests for fsr.corpus.files."""

from __future__ import annotations

import hashlib

import pytest

from fsr.corpus.files import digest_file, write_atomic


def written(tmp_path, name="out.json", payload=b'{"records": []}'):
    """Write a file and return its path."""
    path = tmp_path / name
    path.write_bytes(payload)
    return path


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


class TestWriteAtomic:
    def test_writes_the_text(self, tmp_path):
        path = tmp_path / "a.json"
        write_atomic(path, '{"x": 1}')
        assert path.read_text() == '{"x": 1}'

    def test_replaces_an_existing_file(self, tmp_path):
        path = written(tmp_path, payload=b"old")
        write_atomic(path, "new")
        assert path.read_text() == "new"

    def test_creates_the_parent_directory(self, tmp_path):
        path = tmp_path / "splits" / "a.json"
        write_atomic(path, "{}")
        assert path.read_text() == "{}"

    def test_leaves_no_temporary_file(self, tmp_path):
        write_atomic(tmp_path / "a.json", "{}")
        assert [p.name for p in tmp_path.iterdir()] == ["a.json"]

    def test_writes_utf8(self, tmp_path):
        path = tmp_path / "a.json"
        write_atomic(path, "caf\u00e9")
        assert path.read_bytes() == "caf\u00e9".encode()

    def test_a_failed_write_leaves_nothing_behind(self, tmp_path, monkeypatch):
        def fail(*_args, **_kwargs):
            raise OSError("disk full")

        monkeypatch.setattr("os.replace", fail)
        with pytest.raises(OSError, match="disk full"):
            write_atomic(tmp_path / "a.json", "{}")
        assert list(tmp_path.iterdir()) == []

    def test_a_failed_write_keeps_the_previous_file(self, tmp_path, monkeypatch):
        path = written(tmp_path, payload=b"old")

        def fail(*_args, **_kwargs):
            raise OSError("disk full")

        monkeypatch.setattr("os.replace", fail)
        with pytest.raises(OSError):
            write_atomic(path, "new")
        assert path.read_text() == "old"
