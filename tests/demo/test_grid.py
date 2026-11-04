"""Tests for fsr.demo.grid."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from fsr.demo import grid as mod
from fsr.demo.results import keep_rate
from fsr.formats import FORMAT_NAMES


def version(title, body):
    return {"pairs": [["title", title]], "body": body}


PASSAGES = [
    {
        "id": "key",
        "source": "changelog",
        "optional": False,
        "versions": [version("limit raised", "now 200"), version("limit", "now 200")],
    },
    {
        "id": "stale",
        "source": "tickets",
        "optional": True,
        "versions": [version("limit", "still 50")],
    },
    {
        "id": "far",
        "source": "wiki",
        "optional": False,
        "versions": [version("regions", "dublin"), version("regions", "sydney")],
    },
]


@pytest.fixture
def catalogue(tmp_path):
    root = tmp_path / "demo"
    mod.grid_dir(root).mkdir(parents=True)
    raw = {
        "passages": PASSAGES,
        "wordings": [
            {"text": "is the limit still 50", "stale_premise": True},
            {"text": "what is the limit", "stale_premise": False},
        ],
        "key": "key",
        "k": 1,
        "near_cut": ["key", "stale"],
        "case_wording": 0,
        "case_state": {"key": 0, "stale": 0, "far": 1},
    }
    mod.catalogue_path(root).write_text(json.dumps(raw))
    return mod.load_catalogue(root)


class TestCatalogue:
    def test_texts_in_score_order(self, catalogue):
        assert catalogue.texts() == [
            ("key", 0),
            ("key", 1),
            ("stale", 0),
            ("far", 0),
            ("far", 1),
        ]

    def test_optional_passages_can_be_absent(self, catalogue):
        assert catalogue.levels("stale") == [0, mod.ABSENT]
        assert catalogue.levels("far") == [0, 1]


class TestStates:
    def test_crosses_the_near_passages_twice_then_the_original(self, catalogue):
        got = list(mod.states(catalogue))
        assert len(got) == 2 * 2 * 2 + 1
        assert [label for label, _ in got[:2]] == [mod.REFERENCE, mod.DRAWN]
        assert got[-1] == (mod.ORIGINAL, {"key": 0, "stale": mod.ABSENT, "far": 0})

    def test_reference_keeps_the_far_passages_at_the_case(self, catalogue):
        for label, state in mod.states(catalogue):
            if label == mod.REFERENCE:
                assert state["far"] == 1

    def test_the_draw_is_seeded(self, catalogue):
        assert list(mod.states(catalogue, seed=3)) == list(
            mod.states(catalogue, seed=3)
        )


class TestRetrieval:
    def test_leaves_out_absent_passages(self, catalogue):
        _, docs = mod.retrieve(catalogue, {"key": 0, "stale": mod.ABSENT, "far": 0})
        assert docs == [("key", 0), ("far", 0)]

    def test_ranks_by_bm25(self, catalogue):
        index, docs = mod.retrieve(catalogue, {"key": 0, "stale": 0, "far": 0})
        got = mod.top(index, docs, "still 50", n=2)
        assert got[0] == ("stale", 0)


class TestKeepRates:
    def test_matches_the_single_pool_rate(self):
        rng = np.random.default_rng(1)
        sources = ["changelog", "tickets", "wiki", "api_reference"]
        stack = rng.random((3, 4, len(FORMAT_NAMES)))
        got = mod.keep_rates(stack, [sources] * 3, [0, 1, 2], k=2)
        want = [keep_rate(s, sources, i, 2) for i, s in enumerate(stack)]
        np.testing.assert_allclose(got, want)


def variants(values):
    """A table crossing two wordings with three states, one value per cell."""
    cells = np.asarray(values, dtype=float).reshape(2, 3)
    return pd.DataFrame(
        [
            {"wording": w, "state": s, "a": cells[w, s], "b": cells[w, s] + 0.1}
            for w in range(2)
            for s in range(3)
        ]
    )


class TestBootstrap:
    def test_lays_values_out_by_wording_and_state(self):
        table = variants(range(6))
        np.testing.assert_array_equal(
            mod.crossed(table, table["a"]), [[0, 1, 2], [3, 4, 5]]
        )

    def test_refuses_a_grid_with_gaps(self):
        table = variants(range(6)).iloc[1:]
        with pytest.raises(ValueError, match="cross every wording"):
            mod.crossed(table, table["a"])

    def test_constant_cells_give_a_point_interval(self):
        mean, low, high = mod.crossed_bootstrap(np.full((4, 5), 0.3), n=50)
        assert (mean, low, high) == pytest.approx((0.3, 0.3, 0.3))

    def test_interval_brackets_the_mean(self):
        cells = np.random.default_rng(2).random((6, 8))
        mean, low, high = mod.crossed_bootstrap(cells, n=300)
        assert low < mean < high

    def test_summary_rows(self):
        table = variants([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        got = mod.summary(table, ["a", "b"], control="a", winner="b", n=50)
        assert got.loc["b minus a", "mean"] == pytest.approx(0.1)
        assert got.loc["share of variants with b above a", "mean"] == 1
        assert got.loc["share of variants with b below a", "mean"] == 0


class TestTables:
    def test_per_wording_means(self, catalogue):
        table = pd.DataFrame(
            {"wording": [0, 0, 1, 1], "state": [0, 1, 0, 1], "a": [0.2, 0.4, 1.0, 0.0]}
        )
        got = mod.per_wording(table, catalogue, ["a"])
        assert list(got.index) == ["is the limit still 50", "what is the limit"]
        assert list(got["stale_premise"]) == [True, False]
        assert got["a"].tolist() == pytest.approx([0.3, 0.5])

    def test_versions_show_only_what_differs(self, catalogue):
        got = mod.versions_table(catalogue, "key")
        assert list(got.columns) == ["title"]
        assert list(got.index) == [1, 2]
        assert list(got["title"]) == ["limit raised", "limit"]


class TestLiveWordings:
    def test_keeps_wordings_live_in_more_than_the_threshold(self):
        table = pd.DataFrame(
            {
                "wording": [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
                "a": [0, 0, 0, 0.1, 0.2, 0, 0.3, 0, 1, 1, 1, 0],
                "b": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            }
        )
        assert mod.live_wordings(table, ["a", "b"], threshold=0.5) == [2]
        assert mod.live_wordings(table, ["a", "b"], threshold=0.25) == [1, 2]


class TestSample:
    def test_draws_the_same_number_from_each_wording(self):
        table = pd.DataFrame({"wording": [0] * 10 + [1] * 30, "x": range(40)})
        got = mod.sample_variants(table, per_wording=4, seed=1)
        assert got["wording"].value_counts().to_dict() == {0: 4, 1: 4}
        assert got.equals(mod.sample_variants(table, per_wording=4, seed=1))
