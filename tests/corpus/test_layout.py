"""Tests for fsr.corpus.layout."""

from __future__ import annotations

from pathlib import Path

from fsr.corpus.layout import SPLIT_SUBDIR, split_dir


class TestSplitDir:
    def test_sits_under_the_data_root(self):
        assert split_dir(Path("data/nq")) == Path("data/nq") / SPLIT_SUBDIR
