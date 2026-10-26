"""Tests for fsr.standardisation.layout."""

from __future__ import annotations

from pathlib import Path

from fsr.standardisation.layout import (
    candidates_path,
    standardisation_dir,
    tensor_meta_path,
    tensor_path,
)

ROOT = Path("data/nq")


class TestTensorPath:
    def test_names_the_split_model_and_arm(self):
        path = tensor_path(ROOT, "test", "jina_v2", "5fmt_lam0.1")
        assert path.name == "test_scores_jina_v2_5fmt_lam0.1.npy"

    def test_sits_under_the_study_directory(self):
        path = tensor_path(ROOT, "test", "jina_v2", "base")
        assert path.parent == standardisation_dir(ROOT)

    def test_the_study_directory_sits_under_h2(self):
        assert standardisation_dir(ROOT) == ROOT / "h2" / "standardisation"

    def test_two_arms_of_one_model_do_not_collide(self):
        first = tensor_path(ROOT, "test", "jina_v2", "base")
        second = tensor_path(ROOT, "test", "jina_v2", "5fmt_lam0")
        assert first != second


class TestTensorMetaPath:
    def test_sits_beside_the_scores(self):
        scores = tensor_path(ROOT, "test", "jina_v2", "base")
        meta = tensor_meta_path(ROOT, "test", "jina_v2", "base")
        assert meta == scores.with_suffix(".json")


class TestCandidatesPath:
    def test_reads_the_lists_the_answer_axis_built(self):
        """The study must score the queries the main H2 result scores."""
        assert candidates_path(ROOT, "test") == ROOT / "h1" / "test_candidates.json"

    def test_a_capped_build_has_its_own_lists(self):
        path = candidates_path(ROOT, "test", limit=20)
        assert path.name == "test_candidates_limit20.json"
