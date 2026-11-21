"""Tests for fsr.representation.arms."""

from __future__ import annotations

import json

import pytest

from fsr.formats import FORMAT_NAMES
from fsr.h2_layout import selection_path
from fsr.representation import arms as mod


def write_selection(root, model, **selection):
    path = selection_path(root, model, "lambda")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(selection))


class TestChosenWeight:
    def test_reads_the_winner(self, tmp_path):
        write_selection(tmp_path, "minilm_l6", status="selected", winner_lambda=0.3)
        assert mod.chosen_weight(tmp_path, "minilm_l6") == 0.3

    def test_reads_the_weight_of_the_model_it_borrows_from(self, tmp_path):
        write_selection(tmp_path, "minilm_l6", status="selected", winner_lambda=0.3)
        assert mod.chosen_weight(tmp_path, "minilm_l6_pooler") == 0.3

    def test_rejects_a_halted_selection(self, tmp_path):
        write_selection(tmp_path, "minilm_l6", status="halt")
        with pytest.raises(ValueError, match="no weight was chosen"):
            mod.chosen_weight(tmp_path, "minilm_l6")

    def test_rejects_a_missing_selection(self, tmp_path):
        with pytest.raises(ValueError, match="no weight selection"):
            mod.chosen_weight(tmp_path, "minilm_l6")


class TestTrainedArms:
    def test_one_arm_per_held_out_format(self, tmp_path):
        got = mod.trained_arms(tmp_path, "minilm_l6", ["0"], [0])
        assert got == [f"{fold}_lam0" for fold in FORMAT_NAMES]

    def test_names_the_seed_after_the_default(self, tmp_path):
        got = mod.trained_arms(tmp_path, "minilm_l6", ["0"], [2])
        assert all(name.endswith("_s2") for name in got)

    def test_resolves_the_winner(self, tmp_path):
        write_selection(tmp_path, "minilm_l6", status="selected", winner_lambda=0.3)
        got = mod.trained_arms(tmp_path, "minilm_l6", ["winner"], [0])
        assert got[0] == f"{FORMAT_NAMES[0]}_lam0.3"

    def test_orders_seed_then_weight_then_format(self, tmp_path):
        got = mod.trained_arms(tmp_path, "minilm_l6", ["0", "1"], [0, 1])
        n = len(FORMAT_NAMES)
        assert len(got) == 4 * n
        assert got[n] == f"{FORMAT_NAMES[0]}_lam1"
        assert got[2 * n] == f"{FORMAT_NAMES[0]}_lam0_s1"


class TestMain:
    def test_prints_one_arm_per_line(self, tmp_path, capsys):
        argv = ["--data-root", str(tmp_path), "--model", "minilm_l6"]
        assert mod.main([*argv, "--weights", "0"]) == 0
        lines = capsys.readouterr().out.splitlines()
        assert lines == mod.trained_arms(tmp_path, "minilm_l6", ["0"], [0])

    def test_fails_without_a_chosen_weight(self, tmp_path, capsys):
        argv = ["--data-root", str(tmp_path), "--model", "minilm_l6"]
        assert mod.main([*argv, "--weights", "winner"]) == 2
        assert "ERROR: no weight selection" in capsys.readouterr().err
