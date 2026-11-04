"""Tests for fsr.demo.results."""

from __future__ import annotations

import json

import numpy as np
import pytest

from fsr.demo import results as mod
from fsr.formats import FORMAT_NAMES

YAML, JSON = FORMAT_NAMES.index("yaml"), FORMAT_NAMES.index("json")
SOURCES = ["changelog", "wiki", "tickets", "api_reference"]
IDS = ["key", "stale-a", "stale-b", "other"]


def scores(key_by_format, rest=(0.5, 0.4, 0.1)):
    """Scores of four passages: the key's per format, the others' flat."""
    out = np.empty((4, len(FORMAT_NAMES)), dtype=np.float32)
    out[0] = key_by_format
    for i, value in enumerate(rest, 1):
        out[i] = value
    return out


def write_case(root, arm_scores, k=2, assignment=None):
    case = mod.case_dir(root)
    case.mkdir(parents=True)
    meta = {
        "question": "q",
        "k": k,
        "key": "key",
        "expected": "new",
        "stale": "old",
        "assignment": assignment or dict.fromkeys(mod.SOURCES, "json"),
        "claims": {"key": "new", "stale-a": "old", "stale-b": "old"},
        "arms": list(arm_scores),
        "candidates": [
            {"passage_id": pid, "source": src}
            for pid, src in zip(IDS, SOURCES, strict=True)
        ],
    }
    (case / "case.json").write_text(json.dumps(meta))
    for arm, value in arm_scores.items():
        np.save(mod.scores_path(root, "case", arm), value)
    return mod.load_case(root)


# The key passage wins only in YAML under the untrained model, never under
# lam0, and always under lam1.
ONLY_YAML = [0.9 if f == YAML else 0.3 for f in range(len(FORMAT_NAMES))]
ARMS = {
    "base": scores(ONLY_YAML),
    "5fmt_lam0": scores([0.3] * len(FORMAT_NAMES)),
    "5fmt_lam1": scores([0.45] * len(FORMAT_NAMES)),
}


@pytest.fixture
def case(tmp_path):
    return write_case(tmp_path / "demo", ARMS)


class TestDeployed:
    def test_takes_each_passage_in_its_source_format(self):
        s = np.arange(4 * len(FORMAT_NAMES), dtype=float).reshape(4, -1)
        assignment = np.array([YAML, JSON, YAML, JSON])  # by mod.SOURCES order
        got = mod.deployed(s, SOURCES, assignment)
        want = [
            s[i, assignment[mod.SOURCES.index(src)]] for i, src in enumerate(SOURCES)
        ]
        np.testing.assert_array_equal(got, want)

    def test_stacks_over_assignments(self):
        s = np.random.default_rng(0).random((4, len(FORMAT_NAMES)))
        every = mod.all_assignments()
        assert every.shape == (625, 4)
        got = mod.deployed(s, SOURCES, every)
        assert got.shape == (625, 4)
        np.testing.assert_array_equal(got[7], mod.deployed(s, SOURCES, every[7]))


class TestCase:
    def test_reads_claims_and_key(self, case):
        assert case.key == 0
        assert case.claims == ["new", "old", "old", mod.OTHER]

    def test_orders_the_standards_from_best_to_worst(self, case):
        order = mod.standard_order(case)
        assert order[0] == "yaml"
        assert sorted(order) == sorted(FORMAT_NAMES)

    def test_shows_standardising_to_every_format_best_first(self, case):
        handlings = [c.handling for c in mod.conditions(case)]
        assert handlings == [mod.MIXED, *mod.standard_order(case), mod.MIXED, mod.MIXED]

    def test_ladder_marks_what_is_passed_on(self, case):
        shown = mod.conditions(case)
        table = mod.ladder(case, depth=3)
        assert len(table) == len(shown) * 3
        by = table.set_index(["condition", "rank"])
        labels = [c.label for c in shown]
        mixed, best, worst, control, winner = (labels[i] for i in (0, 1, -3, -2, -1))
        assert by.loc[(best, 1), "passage"] == "key"
        assert by.loc[(worst, 1), "passage"] == "stale-a"
        assert by.loc[(mixed, 1), "passage"] == "stale-a"
        assert not table[(table.condition == control) & table.key].passed_on.any()
        assert table[(table.condition == winner) & table.key].passed_on.all()

    def test_mixed_condition_uses_the_assignment(self, tmp_path):
        yaml_changelog = dict.fromkeys(mod.SOURCES, "json") | {"changelog": "yaml"}
        case = write_case(tmp_path / "demo", ARMS, assignment=yaml_changelog)
        table = mod.ladder(case, depth=1)
        assert table.iloc[0].passage == "key"
        assert table.iloc[0].format == "yaml"


class TestKeepRate:
    def test_share_of_assignments_passing_the_key_on(self, case):
        rates = mod.case_keep_rates(case)
        # Under base the key is passed on only when the changelog is in YAML.
        assert rates["base"] == pytest.approx(1 / len(FORMAT_NAMES))
        assert rates["5fmt_lam0"] == 0
        assert rates["5fmt_lam1"] == 1
