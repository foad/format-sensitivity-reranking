"""Tests for fsr.corpus.config."""

from __future__ import annotations

import dataclasses

import pytest

from fsr.corpus.config import DEFAULT, NQ_DATASET, CorpusConfig


class TestPublishedDefaults:
    @pytest.mark.parametrize(
        "field,expected",
        [
            ("dataset", NQ_DATASET),
            ("body_chars", 3000),
            ("min_paragraph_chars", 30),
            ("max_key_chars", 80),
            ("max_value_chars", 400),
            ("min_pairs", 3),
            ("min_body_chars", 100),
            ("min_infobox_chars", 200),
            ("split_seed", 42),
            ("frac_train", 0.80),
            ("frac_dev", 0.10),
            ("negatives_seed", 42),
            ("cache_k", 25),
        ],
    )
    def test_the_published_value(self, field, expected):
        assert getattr(DEFAULT, field) == expected

    def test_the_test_share_is_the_remainder(self):
        assert DEFAULT.frac_test == pytest.approx(0.10)

    def test_the_three_shares_sum_to_one(self):
        total = DEFAULT.frac_train + DEFAULT.frac_dev + DEFAULT.frac_test
        assert total == pytest.approx(1.0)


class TestImmutability:
    def test_a_field_cannot_be_reassigned(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            DEFAULT.body_chars = 1

    def test_an_override_leaves_the_default_untouched(self):
        CorpusConfig(body_chars=500)
        assert DEFAULT.body_chars == 3000


class TestValidation:
    @pytest.mark.parametrize("value", [0.0, 1.0, -0.1, 1.5])
    def test_rejects_a_train_share_outside_the_unit_interval(self, value):
        with pytest.raises(ValueError, match="frac_train must be between 0 and 1"):
            CorpusConfig(frac_train=value)

    @pytest.mark.parametrize("value", [0.0, 1.0, -0.1])
    def test_rejects_a_dev_share_outside_the_unit_interval(self, value):
        with pytest.raises(ValueError, match="frac_dev must be between 0 and 1"):
            CorpusConfig(frac_dev=value)

    def test_rejects_shares_that_leave_no_test_split(self):
        with pytest.raises(ValueError, match="must leave a test split"):
            CorpusConfig(frac_train=0.9, frac_dev=0.1)

    def test_accepts_shares_that_leave_a_test_split(self):
        assert CorpusConfig(frac_train=0.5, frac_dev=0.25).frac_test == pytest.approx(
            0.25
        )


class TestSerialisation:
    def test_includes_every_field(self):
        as_dict = DEFAULT.as_dict()
        for field in dataclasses.fields(CorpusConfig):
            assert field.name in as_dict

    def test_includes_the_derived_test_share(self):
        assert DEFAULT.as_dict()["frac_test"] == pytest.approx(0.10)

    def test_round_trips_through_the_constructor(self):
        as_dict = DEFAULT.as_dict()
        del as_dict["frac_test"]
        assert CorpusConfig(**as_dict) == DEFAULT

    def test_reflects_an_override(self):
        assert CorpusConfig(cache_k=5).as_dict()["cache_k"] == 5
