"""Tests for fsr.corpus.validate."""

from __future__ import annotations

import json

from fsr.corpus import validate as mod
from fsr.corpus.manifest import Manifest, Output


def record(idx, title=None, flags=None):
    return {
        "id": str(idx),
        "title": title or f"Article {idx}",
        "quality_flags": flags or [],
    }


def splits_of(train, dev, test, nq_val=()):
    return {
        "train": list(train),
        "dev": list(dev),
        "test": list(test),
        "nq_val": list(nq_val),
    }


class TestCheckQualityGate:
    def test_passes_clean_records(self):
        assert mod.check_quality_gate("train", [record(1), record(2)]) == []

    def test_reports_a_flagged_record(self):
        problems = mod.check_quality_gate("dev", [record(1, flags=["body_too_short"])])
        assert len(problems) == 1
        assert "dev: 1 records carry a quality flag" in problems[0]

    def test_samples_the_flagged_ids(self):
        flagged = [record(i, flags=["too_few_pairs"]) for i in range(10)]
        problem = mod.check_quality_gate("train", flagged)[0]
        assert "and 7 more" in problem


class TestCheckTitlesDisjoint:
    def test_passes_disjoint_splits(self):
        splits = splits_of([record(1)], [record(2)], [record(3)])
        assert mod.check_titles_disjoint(splits) == []

    def test_reports_a_shared_title(self):
        splits = splits_of([record(1)], [record(2, title="Article 1")], [record(3)])
        problems = mod.check_titles_disjoint(splits)
        assert problems == ["train and dev share 1 titles (Article 1)"]

    def test_reports_each_overlapping_pair(self):
        shared = [record(1)]
        splits = splits_of(shared, shared, shared)
        assert len(mod.check_titles_disjoint(splits)) == 3

    def test_ignores_the_validation_split(self):
        splits = splits_of([record(1)], [record(2)], [record(3)], [record(1)])
        assert mod.check_titles_disjoint(splits) == []

    def test_checks_only_the_named_splits(self):
        splits = splits_of([record(1)], [record(1)], [record(3)])
        assert mod.check_titles_disjoint(splits, ("train", "test")) == []


class TestCheckIdsUnique:
    def test_passes_unique_ids(self):
        splits = splits_of([record(1)], [record(2)], [record(3)], [record(4)])
        assert mod.check_ids_unique(splits) == []

    def test_reports_an_id_in_two_splits(self):
        splits = splits_of([record(1)], [record(1)], [record(3)])
        problems = mod.check_ids_unique(splits)
        assert len(problems) == 1
        assert "1 record ids appear twice" in problems[0]
        assert "train and dev" in problems[0]

    def test_reports_a_repeat_inside_one_split(self):
        splits = splits_of([record(1), record(1)], [], [])
        assert len(mod.check_ids_unique(splits)) == 1


class TestCheckMeta:
    def _splits(self):
        return splits_of([record(1), record(2)], [record(3)], [record(4)], [record(5)])

    def _meta(self, **overrides):
        meta = {
            "n_records_train": 2,
            "n_records_dev": 1,
            "n_records_test": 1,
            "nq_val_n_records": 1,
        }
        return {**meta, **overrides}

    def test_passes_matching_counts(self):
        assert mod.check_meta(self._meta(), self._splits()) == []

    def test_reports_a_disagreeing_count(self):
        problems = mod.check_meta(self._meta(n_records_dev=9), self._splits())
        assert problems == ["meta n_records_dev is 9, but dev holds 1"]

    def test_reports_the_validation_count(self):
        problems = mod.check_meta(self._meta(nq_val_n_records=4), self._splits())
        assert "nq_val_n_records" in problems[0]

    def test_tolerates_a_missing_key(self):
        meta = self._meta()
        del meta["n_records_test"]
        assert mod.check_meta(meta, self._splits()) == []


def corpus():
    return [record(i) for i in range(1, 6)]


def queries():
    return [record(1), record(2)]


class TestCheckNegatives:
    def _payload(self, negatives, cache_k=2):
        return {"cache_k": cache_k, "negatives": negatives}

    def test_passes_well_formed_negatives(self):
        payload = self._payload({"1": ["2", "3"], "2": ["1", "4"]})
        assert mod.check_negatives(payload, queries(), corpus()) == []

    def test_reports_a_query_with_no_negatives(self):
        payload = self._payload({"1": ["2", "3"]})
        problems = mod.check_negatives(payload, queries(), corpus())
        assert problems == ["1 queries have no negatives (2)"]

    def test_reports_a_short_list(self):
        payload = self._payload({"1": ["2"], "2": ["1", "4"]})
        problems = mod.check_negatives(payload, queries(), corpus())
        assert "do not hold 2 negatives" in problems[0]

    def test_reports_a_repeated_negative(self):
        payload = self._payload({"1": ["2", "2"], "2": ["1", "4"]})
        problems = mod.check_negatives(payload, queries(), corpus())
        assert "repeat a negative" in problems[0]

    def test_reports_a_negative_outside_the_corpus(self):
        payload = self._payload({"1": ["2", "99"], "2": ["1", "4"]})
        problems = mod.check_negatives(payload, queries(), corpus())
        assert "outside the corpus" in problems[0]

    def test_reports_a_negative_from_the_query_article(self):
        extended = [*corpus(), record(9, title="Article 1")]
        payload = self._payload({"1": ["2", "9"], "2": ["1", "4"]})
        problems = mod.check_negatives(payload, queries(), extended)
        assert "from their own article" in problems[0]

    def test_reports_every_kind_of_problem(self):
        payload = self._payload({"1": ["2", "2"], "3": ["99", "99"]})
        problems = mod.check_negatives(payload, queries(), corpus())
        assert len(problems) == 3


class TestCheckManifestDigests:
    def _manifest(self, tmp_path, name="a.json", body='{"x": 1}'):
        path = tmp_path / name
        path.write_text(body)
        manifest = Manifest(config={})
        manifest.record("parse", [Output.describe(path, tmp_path, 1)])
        return manifest, path

    def test_passes_unchanged_outputs(self, tmp_path):
        manifest, _ = self._manifest(tmp_path)
        assert mod.check_manifest_digests(manifest, tmp_path) == []

    def test_reports_a_missing_output(self, tmp_path):
        manifest, path = self._manifest(tmp_path)
        path.unlink()
        assert mod.check_manifest_digests(manifest, tmp_path) == [
            "parse: a.json is missing"
        ]

    def test_reports_a_changed_output(self, tmp_path):
        manifest, path = self._manifest(tmp_path)
        path.write_text(json.dumps({"x": 2}))
        assert mod.check_manifest_digests(manifest, tmp_path) == [
            "parse: a.json changed after the build"
        ]

    def test_checks_every_stage(self, tmp_path):
        manifest, first = self._manifest(tmp_path)
        second = tmp_path / "b.json"
        second.write_text("{}")
        manifest.record("split", [Output.describe(second, tmp_path)])
        first.unlink()
        second.unlink()
        assert len(mod.check_manifest_digests(manifest, tmp_path)) == 2
