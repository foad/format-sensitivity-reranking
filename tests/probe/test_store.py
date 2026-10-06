"""Tests for fsr.probe.store."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from fsr.h2_layout import feature_meta_path, feature_path, feature_score_path
from fsr.probe.store import FeatureStore, load_store

MODEL = "minilm_l6"
FORMATS = ["yaml", "json", "toml"]
N, C, D = 4, 4, 3


def write_cache(root, split="train", formats=None, n_records=N, n_candidates=C):
    names = formats or FORMATS
    shape = (n_records, n_candidates, len(names), D)
    features = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    path = feature_path(root, split, MODEL)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, features)
    np.save(
        feature_score_path(root, split, MODEL),
        features.sum(axis=-1).astype(np.float32),
    )
    feature_meta_path(root, split, MODEL).write_text(
        json.dumps(
            {
                "model": MODEL,
                "split": split,
                "formats": names,
                "record_ids": [f"r{i}" for i in range(n_records)],
            }
        )
    )
    return features


@pytest.fixture
def store(tmp_path):
    write_cache(tmp_path)
    return load_store(tmp_path, "train", MODEL)


class TestLoadStore:
    def test_reads_what_the_capture_wrote(self, tmp_path):
        written = write_cache(tmp_path)
        loaded = load_store(tmp_path, "train", MODEL)
        assert np.array_equal(np.asarray(loaded.features), written)
        assert loaded.model == MODEL
        assert loaded.split == "train"
        assert loaded.formats == FORMATS

    def test_memory_maps_the_representations(self, tmp_path):
        write_cache(tmp_path)
        assert isinstance(load_store(tmp_path, "train", MODEL).features, np.memmap)

    def test_refuses_a_cache_that_is_not_there(self, tmp_path):
        with pytest.raises(SystemExit, match="no cached representations"):
            load_store(tmp_path, "train", MODEL)

    def test_refuses_a_cache_with_no_description(self, tmp_path):
        write_cache(tmp_path)
        feature_meta_path(tmp_path, "train", MODEL).unlink()
        with pytest.raises(SystemExit, match="no cached representations"):
            load_store(tmp_path, "train", MODEL)


class TestShape:
    def test_counts_the_records(self, store):
        assert len(store) == N

    def test_reports_the_representation_width(self, store):
        assert store.dim == D

    def test_counts_the_negatives_without_the_gold(self, store):
        assert store.n_negatives == C - 1


class TestFormatColumns:
    def test_locates_each_name(self, store):
        assert store.format_columns(["toml", "yaml"]) == [2, 0]

    def test_refuses_a_format_the_cache_does_not_hold(self, store):
        with pytest.raises(ValueError, match="does not include 'markdown'"):
            store.format_columns(["markdown"])


class TestBatch:
    def test_shapes_match_what_the_loss_expects(self, store):
        gold, negatives = store.batch([0, 1], FORMATS, n_negatives=2)
        assert gold.shape == (2, len(FORMATS), D)
        assert negatives.shape == (2, len(FORMATS), 2, D)

    def test_the_gold_is_the_first_candidate(self, tmp_path):
        written = write_cache(tmp_path)
        loaded = load_store(tmp_path, "train", MODEL)
        gold, _ = loaded.batch([1], FORMATS, n_negatives=1)
        assert np.array_equal(gold.numpy()[0], written[1, 0])

    def test_the_negatives_follow_the_gold_in_order(self, tmp_path):
        written = write_cache(tmp_path)
        loaded = load_store(tmp_path, "train", MODEL)
        _, negatives = loaded.batch([0], FORMATS, n_negatives=2)
        assert np.array_equal(negatives.numpy()[0, 0, 0], written[0, 1, 0])
        assert np.array_equal(negatives.numpy()[0, 0, 1], written[0, 2, 0])

    def test_takes_the_formats_in_the_order_asked_for(self, tmp_path):
        written = write_cache(tmp_path)
        loaded = load_store(tmp_path, "train", MODEL)
        gold, _ = loaded.batch([0], ["toml", "yaml"], n_negatives=1)
        assert np.array_equal(gold.numpy()[0, 0], written[0, 0, 2])
        assert np.array_equal(gold.numpy()[0, 1], written[0, 0, 0])

    def test_a_subset_of_formats_narrows_the_format_axis(self, store):
        gold, negatives = store.batch([0], ["yaml"], n_negatives=1)
        assert gold.shape[1] == 1
        assert negatives.shape[1] == 1

    def test_returns_float_tensors(self, store):
        gold, negatives = store.batch([0], FORMATS, n_negatives=1)
        assert gold.dtype == torch.float32
        assert negatives.dtype == torch.float32

    def test_refuses_more_negatives_than_it_cached(self, store):
        with pytest.raises(ValueError, match="holds 3 negatives"):
            store.batch([0], FORMATS, n_negatives=9)


class TestGoldScores:
    def test_returns_one_score_per_record_and_format(self, store):
        assert store.gold_scores(FORMATS).shape == (N, len(FORMATS))

    def test_matches_the_captured_score(self, tmp_path):
        written = write_cache(tmp_path)
        loaded = load_store(tmp_path, "train", MODEL)
        assert np.allclose(
            loaded.gold_scores(["yaml"])[:, 0], written[:, 0, 0].sum(axis=-1)
        )


class TestDirectConstruction:
    def test_holds_what_it_was_given(self):
        store = FeatureStore(
            features=np.zeros((1, 2, 1, D), dtype=np.float32),
            scores=np.zeros((1, 2, 1), dtype=np.float32),
            record_ids=["a"],
            formats=["yaml"],
            model=MODEL,
            split="dev",
        )
        assert len(store) == 1
        assert store.n_negatives == 1
