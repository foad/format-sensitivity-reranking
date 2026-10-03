"""Tests for fsr.h1_results."""

from __future__ import annotations

import json

import pytest

from fsr import h1_results as mod
from fsr.models.registry import by_slug

FORMATS = ["yaml", "json", "toml"]
SLUGS = ["minilm_l6", "bge_base"]


def cross_entry(max_d=0.9):
    return {
        "stats": {
            "per_format": {
                f: {"mean": i / 10, "std": 1.0, "median": 0.0}
                for i, f in enumerate(FORMATS)
            },
            "pairwise": [
                {
                    "pair": "yaml - json",
                    "mean_diff": 0.3,
                    "cohen_d": max_d,
                    "wilcoxon_p": 0.0,
                },
                {
                    "pair": "json - toml",
                    "mean_diff": -0.1,
                    "cohen_d": -0.2,
                    "wilcoxon_p": 0.1,
                },
            ],
            "rank": [
                {"pair": "yaml vs json", "spearman_rho": 0.9, "flip_rate_pct": 12.0}
            ],
            "summary": {
                "max_abs_cohen_d": max_d,
                "max_d_pair": "yaml - json",
                "mean_abs_cohen_d": max_d / 2,
                "min_spearman_rho": 0.9,
                "min_rho_pair": "yaml vs json",
                "max_flip_rate_pct": 12.0,
                "max_flip_pair": "yaml vs json",
                "format_ranking_by_mean": FORMATS,
            },
        }
    }


def within_entry(inconsistency=17.5):
    return {
        "within_query": {
            "summary": {
                "n_queries": 10,
                "n_candidates": 4,
                "max_flip_rate_pct": 12.5,
                "max_flip_pair": "yaml vs json",
                "min_kendall_tau": 0.75,
                "min_tau_pair": "yaml vs json",
                "max_top1_changed_pct": 5.0,
                "max_top1_pair": "yaml vs json",
            }
        },
        "gold_top1": {
            "gold_top1_all_formats_pct": 74.0,
            "gold_top1_format_dependent_pct": 16.0,
            "gold_top1_never_pct": 10.0,
            "worst_pair_disagreement_pct": 8.0,
            "worst_pair": "yaml vs json",
            "per_format_top1_pct": {"yaml": 86.0, "json": 83.0, "toml": 82.0},
        },
        "conditional_inconsistency": {
            "n_answerable": 9,
            "inconsistent": 2,
            "inconsistency_pct": inconsistency,
            "answerable_pct": 90.0,
        },
        "scale_diagnostic": {"delta": 0.5, "s_within": 2.75, "ratio": 0.18},
    }


def bootstrap_entry():
    """Return a within-query entry carrying the per-query vectors."""
    entry = within_entry()
    entry["reciprocal_ranks"] = {
        f: [1.0, 0.5, 1.0, 0.25, 1.0, 0.5] for f in mod.FORMAT_NAMES
    }
    entry["reciprocal_ranks"]["json"] = [0.5, 0.5, 1.0, 0.25, 1.0, 1.0]
    entry["conditional_inconsistency"] = {
        "n_answerable": 5,
        "inconsistent": 2,
        "inconsistency_pct": 40.0,
        "answerable_pct": 83.0,
    }
    entry["per_query"] = {
        "yaml vs json": {
            "flip_rate_pct": [10.0, 20.0, 5.0, 0.0, 15.0, 30.0],
            "kendall_tau": [0.8, 0.7, 0.9, 1.0, 0.6, 0.5],
        }
    }
    return entry


def write(tmp_path, split, axis, mode, slugs=None, entry=None):
    """Write one result file per model and return the data root."""
    out = mod.results_dir(tmp_path)
    out.mkdir(parents=True, exist_ok=True)
    for slug in slugs or SLUGS:
        payload = {
            "split": split,
            "mode": mode,
            "results": {
                by_slug(slug).model_id: entry
                or (cross_entry() if axis == "cross" else within_entry())
            },
        }
        (out / f"{split}_{axis}_{mode}_{slug}.json").write_text(json.dumps(payload))
    return tmp_path


class TestPaths:
    def test_names_the_file(self, tmp_path):
        path = mod.result_path(tmp_path, "test", "cross", "with_body", "bge_base")
        assert path.name == "test_cross_with_body_bge_base.json"
        assert path.parent == tmp_path / mod.RESULTS_SUBDIR


class TestLoadAxis:
    def test_loads_every_model(self, tmp_path):
        root = write(tmp_path, "test", "cross", "with_body")
        assert list(mod.load_axis(root, "test", "cross", "with_body")) == [
            "minilm_l6",
            "bge_base",
        ]

    def test_keeps_roster_order(self, tmp_path):
        root = write(tmp_path, "test", "cross", "with_body", ["bge_base", "minilm_l6"])
        loaded = mod.load_axis(root, "test", "cross", "with_body")
        assert next(iter(loaded)) == "minilm_l6"

    def test_skips_an_absent_model(self, tmp_path):
        root = write(tmp_path, "test", "cross", "with_body", ["minilm_l6"])
        assert list(mod.load_axis(root, "test", "cross", "with_body")) == ["minilm_l6"]

    def test_an_absent_split_is_empty(self, tmp_path):
        assert mod.load_axis(tmp_path, "all", "cross", "with_body") == {}

    def test_rejects_a_file_holding_several_models(self, tmp_path):
        out = mod.results_dir(tmp_path)
        out.mkdir(parents=True)
        (out / "test_cross_with_body_minilm_l6.json").write_text(
            json.dumps({"results": {"a": cross_entry(), "b": cross_entry()}})
        )
        with pytest.raises(ValueError, match="holds 2 models"):
            mod.load_axis(tmp_path, "test", "cross", "with_body")

    def test_modes_are_separate(self, tmp_path):
        root = write(tmp_path, "test", "cross", "with_body")
        assert mod.load_axis(root, "test", "cross", "metadata_only") == {}


class TestAvailable:
    def test_reports_the_splits_present(self, tmp_path):
        write(tmp_path, "test", "within", "with_body")
        write(tmp_path, "nq_val", "within", "with_body")
        assert mod.available(tmp_path, "within", "with_body") == ["test", "nq_val"]

    def test_reports_nothing_when_absent(self, tmp_path):
        assert mod.available(tmp_path, "within", "with_body") == []


class TestScoreTable:
    def test_one_row_per_model_labelled(self, tmp_path):
        root = write(tmp_path, "test", "cross", "with_body")
        table = mod.score_table(mod.load_axis(root, "test", "cross", "with_body"))
        assert list(table.index) == ["MiniLM-L6", "bge-base"]
        assert table.index.name == "model"

    def test_carries_the_summary_statistics(self, tmp_path):
        root = write(tmp_path, "test", "cross", "with_body")
        row = mod.score_table(mod.load_axis(root, "test", "cross", "with_body")).iloc[0]
        assert row["max_abs_d"] == 0.9
        assert row["max_d_pair"] == "yaml - json"
        assert row["mean_abs_d"] == 0.45
        assert row["min_rho"] == 0.9


class TestPairwiseTable:
    def test_one_row_per_model_and_pair(self, tmp_path):
        root = write(tmp_path, "test", "cross", "with_body")
        table = mod.pairwise_table(mod.load_axis(root, "test", "cross", "with_body"))
        assert len(table) == 4
        assert set(table["model"]) == {"MiniLM-L6", "bge-base"}

    def test_splits_the_pair(self, tmp_path):
        root = write(tmp_path, "test", "cross", "with_body")
        table = mod.pairwise_table(mod.load_axis(root, "test", "cross", "with_body"))
        assert table.iloc[0]["left"] == "yaml"
        assert table.iloc[0]["right"] == "json"

    def test_carries_the_absolute_effect(self, tmp_path):
        root = write(tmp_path, "test", "cross", "with_body")
        table = mod.pairwise_table(mod.load_axis(root, "test", "cross", "with_body"))
        assert table.iloc[1]["cohen_d"] == -0.2
        assert table.iloc[1]["abs_d"] == 0.2


class TestAnswerTable:
    def test_carries_both_axes_of_the_breakdown(self, tmp_path):
        root = write(tmp_path, "test", "within", "with_body")
        row = mod.answer_table(mod.load_axis(root, "test", "within", "with_body")).iloc[
            0
        ]
        assert row["inconsistency_pct"] == 17.5
        assert row["answerable_pct"] == 90.0
        assert row["top1_format_dependent_pct"] == 16.0
        assert row["min_tau"] == 0.75
        assert row["delta_over_s"] == 0.18


class TestPerFormatTables:
    def test_top1_has_a_column_per_format(self, tmp_path):
        root = write(tmp_path, "test", "within", "with_body")
        table = mod.per_format_top1(mod.load_axis(root, "test", "within", "with_body"))
        assert list(table.columns) == [*FORMATS, "spread_pp"]

    def test_top1_reports_the_spread(self, tmp_path):
        root = write(tmp_path, "test", "within", "with_body")
        table = mod.per_format_top1(mod.load_axis(root, "test", "within", "with_body"))
        assert table.iloc[0]["spread_pp"] == pytest.approx(4.0)

    def test_scores_have_a_column_per_format(self, tmp_path):
        root = write(tmp_path, "test", "cross", "with_body")
        table = mod.per_format_scores(mod.load_axis(root, "test", "cross", "with_body"))
        assert list(table.columns) == FORMATS
        assert table.iloc[0]["yaml"] == 0.0


class TestBootstrapFrame:
    def test_reports_an_interval_per_model(self, tmp_path):
        root = write(tmp_path, "test", "within", "with_body", entry=bootstrap_entry())
        frame = mod.bootstrap_frame(
            mod.load_axis(root, "test", "within", "with_body"), n_boot=50
        )
        assert list(frame.index) == ["MiniLM-L6", "bge-base"]
        row = frame.iloc[0]
        assert row["format_dependent_lo"] <= row["format_dependent_pct"]
        assert row["format_dependent_pct"] <= row["format_dependent_hi"]

    def test_names_the_worst_pair(self, tmp_path):
        root = write(tmp_path, "test", "within", "with_body", entry=bootstrap_entry())
        frame = mod.bootstrap_frame(
            mod.load_axis(root, "test", "within", "with_body"), n_boot=50
        )
        assert frame.iloc[0]["worst_pair"] == "yaml vs json"


class TestIntervals:
    def test_score_interval_brackets_the_estimate(self, tmp_path):
        entry = cross_entry()
        entry["scores"] = {f: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0] for f in mod.FORMAT_NAMES}
        entry["scores"]["json"] = [1.0, 1.5, 2.5, 2.0, 5.0, 4.0]
        root = write(tmp_path, "test", "cross", "with_body", entry=entry)
        frame = mod.score_intervals(
            mod.load_axis(root, "test", "cross", "with_body"), n_boot=200
        )
        assert list(frame.columns) == ["max_abs_d", "low", "high"]
        assert frame.iloc[0]["low"] <= frame.iloc[0]["high"]

    def test_answer_interval_brackets_the_estimate(self, tmp_path):
        root = write(tmp_path, "test", "within", "with_body", entry=bootstrap_entry())
        frame = mod.answer_intervals(
            mod.load_axis(root, "test", "within", "with_body"), n_boot=200
        )
        row = frame.iloc[0]
        assert row["low"] <= row["inconsistency_pct"] <= row["high"]

    def test_answer_interval_is_labelled_in_roster_order(self, tmp_path):
        root = write(tmp_path, "test", "within", "with_body", entry=bootstrap_entry())
        frame = mod.answer_intervals(
            mod.load_axis(root, "test", "within", "with_body"), n_boot=100
        )
        assert list(frame.index) == ["MiniLM-L6", "bge-base"]
