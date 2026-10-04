"""Tests for scripts.h2.select."""

from __future__ import annotations

import json

import pytest
from scripts.h2 import select as mod

from fsr.formats import FORMAT_NAMES
from fsr.h2_layout import BASE_ARM, h2_dir, result_path, selection_path
from fsr.selection import LAMBDA, RANK

SLUG = "minilm_l6"


def guardrail(ranks):
    return {"per_format_reciprocal_ranks": {f: list(ranks) for f in FORMAT_NAMES}}


def write_eval(path, ranks, max_abs_d, interval=(0.0, 1.0)):
    payload = {
        "mrr_guardrail": guardrail(ranks),
        "format_sensitivity": {"summary": {"max_abs_cohen_d": max_abs_d}},
    }
    if interval is not None:
        payload["max_abs_d_ci95"] = list(interval)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


@pytest.fixture
def data_root(tmp_path):
    root = tmp_path / "nq"
    write_eval(
        result_path(root, "dev", "cross", SLUG, BASE_ARM), [0.5] * 20, 0.80, (0.7, 0.9)
    )
    return root


def candidate(root, arm, ranks=None, max_abs_d=0.3, interval=(0.25, 0.35)):
    return write_eval(
        result_path(root, "dev", "cross", SLUG, arm),
        ranks or [0.5] * 20,
        max_abs_d,
        interval,
    )


def run(monkeypatch, data_root, *extra):
    monkeypatch.setattr(
        "sys.argv",
        ["select.py", "--model", SLUG, "--data-root", str(data_root), *extra],
    )
    return mod.main()


def chosen(data_root, sweep="lambda"):
    return json.loads(selection_path(data_root, SLUG, sweep).read_text())


class TestCandidatePaths:
    def test_finds_the_weight_candidates(self, data_root):
        candidate(data_root, "5fmt_lam0")
        candidate(data_root, "5fmt_lam0.1")
        found = mod.candidate_paths(data_root, SLUG, LAMBDA)
        assert [p.name for p in found] == [
            f"dev_cross_{SLUG}_5fmt_lam0.json",
            f"dev_cross_{SLUG}_5fmt_lam0.1.json",
        ]

    def test_leaves_out_the_rank_candidates(self, data_root):
        candidate(data_root, "5fmt_lam0.1")
        candidate(data_root, "5fmt_lam0.1_r8")
        found = mod.candidate_paths(data_root, SLUG, LAMBDA)
        assert all("_r8" not in p.name for p in found)

    def test_leaves_out_a_fold_candidate(self, data_root):
        candidate(data_root, "5fmt_lam0.1")
        candidate(data_root, "yaml_lam0.1")
        found = mod.candidate_paths(data_root, SLUG, LAMBDA)
        assert all("yaml" not in p.name for p in found)

    def test_orders_the_weights_as_numbers(self, data_root):
        for value in ("0", "0.01", "0.1", "1.0", "10.0"):
            candidate(data_root, f"5fmt_lam{value}")
        found = mod.candidate_paths(data_root, SLUG, LAMBDA)
        values = [p.name.rsplit("lam", 1)[1].removesuffix(".json") for p in found]
        assert values == ["0", "0.01", "0.1", "1.0", "10.0"]

    def test_orders_the_ranks_as_numbers(self, data_root):
        path = selection_path(data_root, SLUG, "lambda")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"winner_lambda": "0.1"}))
        for rank in ("4", "8", "16", "32"):
            candidate(data_root, f"5fmt_lam0.1_r{rank}")
        found = mod.candidate_paths(data_root, SLUG, RANK)
        ranks = [p.name.rsplit("_r", 1)[1].removesuffix(".json") for p in found]
        assert ranks == ["4", "8", "16", "32"]

    def test_the_rank_sweep_reads_the_chosen_weight(self, data_root):
        selection_path(data_root, SLUG, "lambda").parent.mkdir(parents=True)
        selection_path(data_root, SLUG, "lambda").write_text(
            json.dumps({"winner_lambda": "0.1"})
        )
        candidate(data_root, "5fmt_lam0.1_r8")
        candidate(data_root, "5fmt_lam0.1_r16")
        candidate(data_root, "5fmt_lam1_r8")
        found = mod.candidate_paths(data_root, SLUG, RANK)
        assert [p.name for p in found] == [
            f"dev_cross_{SLUG}_5fmt_lam0.1_r8.json",
            f"dev_cross_{SLUG}_5fmt_lam0.1_r16.json",
        ]

    def test_the_rank_sweep_refuses_before_a_weight_is_chosen(self, data_root):
        with pytest.raises(SystemExit, match="no weight chosen yet"):
            mod.candidate_paths(data_root, SLUG, RANK)

    def test_returns_nothing_when_no_candidate_exists(self, data_root):
        assert mod.candidate_paths(data_root, SLUG, LAMBDA) == []


class TestSelectWeight:
    def test_writes_the_selection(self, monkeypatch, data_root):
        candidate(data_root, "5fmt_lam0", max_abs_d=0.60, interval=(0.55, 0.65))
        candidate(data_root, "5fmt_lam0.1", max_abs_d=0.20, interval=(0.15, 0.25))
        assert run(monkeypatch, data_root) == 0
        assert chosen(data_root)["winner_lambda"] == "0.1"

    def test_names_the_model(self, monkeypatch, data_root):
        candidate(data_root, "5fmt_lam0.1", max_abs_d=0.20, interval=(0.15, 0.25))
        run(monkeypatch, data_root)
        assert chosen(data_root)["model"] == SLUG

    def test_records_the_swept_quantity(self, monkeypatch, data_root):
        candidate(data_root, "5fmt_lam0.1", max_abs_d=0.20, interval=(0.15, 0.25))
        run(monkeypatch, data_root)
        assert chosen(data_root)["swept"] == "lambda"

    def test_prints_the_winner(self, monkeypatch, data_root, capsys):
        candidate(data_root, "5fmt_lam0.1", max_abs_d=0.20, interval=(0.15, 0.25))
        run(monkeypatch, data_root)
        assert "Winner: lambda = 0.1" in capsys.readouterr().out

    def test_prints_one_row_per_candidate(self, monkeypatch, data_root, capsys):
        candidate(data_root, "5fmt_lam0", max_abs_d=0.60, interval=(0.55, 0.65))
        candidate(data_root, "5fmt_lam0.1", max_abs_d=0.20, interval=(0.15, 0.25))
        run(monkeypatch, data_root)
        printed = capsys.readouterr().out
        assert "0.1" in printed
        assert "yes" in printed


class TestTieBreak:
    def test_reports_the_tied_group(self, monkeypatch, data_root, capsys):
        for value, point in (("0", 0.60), ("0.1", 0.20), ("1", 0.40)):
            candidate(
                data_root, f"5fmt_lam{value}", max_abs_d=point, interval=(0.0, 1.0)
            )
        run(monkeypatch, data_root)
        printed = capsys.readouterr().out
        assert "3 candidates tie" in printed

    def test_states_the_value_the_sweep_takes(self, monkeypatch, data_root, capsys):
        for value, point in (("0", 0.60), ("0.1", 0.20), ("1", 0.40)):
            candidate(
                data_root, f"5fmt_lam{value}", max_abs_d=point, interval=(0.0, 1.0)
            )
        run(monkeypatch, data_root)
        assert "the sweep takes 0.1" in capsys.readouterr().out

    def test_the_tied_group_keeps_the_order_it_was_read_in(
        self, monkeypatch, data_root
    ):
        for value, point in (("0", 0.60), ("0.1", 0.20), ("1", 0.40)):
            candidate(
                data_root, f"5fmt_lam{value}", max_abs_d=point, interval=(0.0, 1.0)
            )
        run(monkeypatch, data_root)
        read_order = [
            p.name.rsplit("lam", 1)[1].removesuffix(".json")
            for p in mod.candidate_paths(data_root, SLUG, LAMBDA)
        ]
        assert chosen(data_root)["tied_lambdas"] == read_order

    def test_a_three_way_weight_tie_takes_the_median_not_the_smallest(
        self, monkeypatch, data_root
    ):
        for value, point in (("0", 0.60), ("0.1", 0.20), ("1", 0.40)):
            candidate(
                data_root, f"5fmt_lam{value}", max_abs_d=point, interval=(0.0, 1.0)
            )
        run(monkeypatch, data_root)
        out = chosen(data_root)
        assert set(out["tied_lambdas"]) == {"0", "0.1", "1"}
        assert out["winner_lambda"] == "0.1"

    def test_the_message_does_not_claim_the_smallest(
        self, monkeypatch, data_root, capsys
    ):
        for value, point in (("0", 0.60), ("0.1", 0.20), ("1", 0.40)):
            candidate(
                data_root, f"5fmt_lam{value}", max_abs_d=point, interval=(0.0, 1.0)
            )
        run(monkeypatch, data_root)
        assert "smallest" not in capsys.readouterr().out


class TestSelectRank:
    @pytest.fixture
    def with_weight(self, data_root):
        path = selection_path(data_root, SLUG, "lambda")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"winner_lambda": "0.1"}))
        return data_root

    def test_takes_the_smallest_of_a_tied_group(self, monkeypatch, with_weight):
        for rank, point in (("4", 0.60), ("8", 0.20), ("16", 0.40)):
            candidate(
                with_weight,
                f"5fmt_lam0.1_r{rank}",
                max_abs_d=point,
                interval=(0.0, 1.0),
            )
        assert run(monkeypatch, with_weight, "--sweep", "rank") == 0
        out = chosen(with_weight, "rank")
        assert out["winner_rank"] == "4"
        assert set(out["tied_ranks"]) == {"4", "8", "16"}

    def test_writes_to_its_own_file(self, monkeypatch, with_weight):
        candidate(with_weight, "5fmt_lam0.1_r8", max_abs_d=0.2, interval=(0.1, 0.3))
        run(monkeypatch, with_weight, "--sweep", "rank")
        assert selection_path(with_weight, SLUG, "rank").exists()
        assert (
            not selection_path(with_weight, SLUG, "lambda")
            .read_text()
            .startswith('{"status"')
        )

    def test_names_the_swept_quantity(self, monkeypatch, with_weight):
        candidate(with_weight, "5fmt_lam0.1_r8", max_abs_d=0.2, interval=(0.1, 0.3))
        run(monkeypatch, with_weight, "--sweep", "rank")
        assert chosen(with_weight, "rank")["swept"] == "rank"


class TestHalt:
    def test_exits_non_zero_when_nothing_keeps_ranking_quality(
        self, monkeypatch, tmp_path
    ):
        root = tmp_path / "nq"
        write_eval(
            result_path(root, "dev", "cross", SLUG, BASE_ARM),
            [0.9] * 20,
            0.8,
            (0.7, 0.9),
        )
        candidate(root, "5fmt_lam1", ranks=[0.1] * 20, max_abs_d=0.2)
        assert run(monkeypatch, root) == mod.HALT

    def test_still_writes_the_selection(self, monkeypatch, tmp_path):
        root = tmp_path / "nq"
        write_eval(
            result_path(root, "dev", "cross", SLUG, BASE_ARM),
            [0.9] * 20,
            0.8,
            (0.7, 0.9),
        )
        candidate(root, "5fmt_lam1", ranks=[0.1] * 20, max_abs_d=0.2)
        run(monkeypatch, root)
        assert chosen(root)["status"] == "halt"

    def test_says_phase_two_cannot_start(self, monkeypatch, tmp_path, capsys):
        root = tmp_path / "nq"
        write_eval(
            result_path(root, "dev", "cross", SLUG, BASE_ARM),
            [0.9] * 20,
            0.8,
            (0.7, 0.9),
        )
        candidate(root, "5fmt_lam1", ranks=[0.1] * 20, max_abs_d=0.2)
        run(monkeypatch, root)
        assert "Phase 2 cannot start" in capsys.readouterr().err


class TestMissingInputs:
    def test_refuses_without_a_baseline(self, monkeypatch, tmp_path):
        root = tmp_path / "nq"
        h2_dir(root).mkdir(parents=True)
        with pytest.raises(SystemExit, match="no baseline evaluation"):
            run(monkeypatch, root)

    def test_refuses_without_candidates(self, monkeypatch, data_root):
        with pytest.raises(SystemExit, match="no candidate evaluations"):
            run(monkeypatch, data_root)


class TestSkipAndForce:
    def test_skips_a_finished_selection(self, monkeypatch, data_root, capsys):
        candidate(data_root, "5fmt_lam0.1", max_abs_d=0.20, interval=(0.15, 0.25))
        run(monkeypatch, data_root)
        capsys.readouterr()
        assert run(monkeypatch, data_root) == 0
        assert "Skipping" in capsys.readouterr().out

    def test_force_chooses_again(self, monkeypatch, data_root, capsys):
        candidate(data_root, "5fmt_lam0.1", max_abs_d=0.20, interval=(0.15, 0.25))
        run(monkeypatch, data_root)
        capsys.readouterr()
        run(monkeypatch, data_root, "--force")
        assert "Skipping" not in capsys.readouterr().out


class TestMissingInterval:
    def test_names_a_candidate_with_no_interval(self, monkeypatch, data_root, capsys):
        candidate(data_root, "5fmt_lam0.1", max_abs_d=0.20, interval=None)
        run(monkeypatch, data_root)
        assert "ties only itself" in capsys.readouterr().out
