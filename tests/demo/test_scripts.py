"""Tests for the scripts under scripts/demo."""

from __future__ import annotations

import json

import numpy as np
import pytest
from scripts.demo import build_scenarios, loop_worker, score
from scripts.demo import select as select_script

from fsr.demo.layout import (
    inbox_dir,
    outbox_dir,
    scenarios_path,
    selection_path,
    tensor_meta_path,
    topk_path,
)
from fsr.demo.store import load_tensor, save_tensor
from fsr.formats import FORMAT_NAMES
from fsr.h2_layout import adapter_dir
from fsr.h2_layout import selection_path as h2_selection_path
from fsr.reporting import parse_progress
from fsr.standardisation.store import ScoreTensor
from tests.fakes import WordTokenizer

SLUG = "bge_base"
N = 4


def run(monkeypatch, module, *argv):
    monkeypatch.setattr("sys.argv", [f"{module.__name__}.py", *argv])
    module.main()


@pytest.fixture
def demo_root(tmp_path, monkeypatch):
    root = tmp_path / "demo"
    run(monkeypatch, build_scenarios, "--demo-root", str(root), "--n", str(N))
    return root


class TestBuild:
    def test_writes_the_pools(self, demo_root):
        stored = json.loads(scenarios_path(demo_root).read_text())
        assert stored["n"] == N
        assert len(stored["scenarios"]) == N

    def test_skips_present_pools(self, monkeypatch, demo_root, capsys):
        run(monkeypatch, build_scenarios, "--demo-root", str(demo_root), "--n", "2")
        assert "already present" in capsys.readouterr().out
        assert json.loads(scenarios_path(demo_root).read_text())["n"] == N

    def test_force_draws_again(self, monkeypatch, demo_root):
        run(
            monkeypatch,
            build_scenarios,
            "--demo-root",
            str(demo_root),
            "--n",
            "2",
            "--force",
        )
        assert json.loads(scenarios_path(demo_root).read_text())["n"] == 2


@pytest.fixture
def stub_model(monkeypatch):
    """Keep the scoring away from the network and the GPU."""
    monkeypatch.setattr(score, "load_tokenizer", lambda *_a, **_k: WordTokenizer())
    monkeypatch.setattr(score, "load_model", lambda *_a, **_k: object())
    monkeypatch.setattr(
        "fsr.standardisation.scoring.score_batch",
        lambda _m, _t, pairs, *_a, **_k: [float(len(p[1])) for p in pairs],
    )


def score_args(demo_root, data_root, *extra):
    return [
        "--model",
        SLUG,
        "--demo-root",
        str(demo_root),
        "--data-root",
        str(data_root),
        *extra,
    ]


@pytest.mark.usefixtures("stub_model")
class TestScore:
    def test_writes_one_tensor_per_condition(self, monkeypatch, demo_root, tmp_path):
        run(monkeypatch, score, *score_args(demo_root, tmp_path / "nq"))
        read = load_tensor(demo_root, SLUG, "base")
        assert read.scores.shape == (N, 10, len(FORMAT_NAMES))
        assert read.query_ids == [f"demo{i:05d}" for i in range(N)]

    def test_describes_the_condition(self, monkeypatch, demo_root, tmp_path):
        run(monkeypatch, score, *score_args(demo_root, tmp_path / "nq"))
        meta = json.loads(tensor_meta_path(demo_root, SLUG, "base").read_text())
        assert meta["adapter"] is None
        assert meta["split"] == "demo"

    def test_counts_every_format(self, monkeypatch, demo_root, tmp_path):
        path = tmp_path / "p.progress"
        run(
            monkeypatch,
            score,
            *score_args(demo_root, tmp_path / "nq", "--progress-file", str(path)),
        )
        assert parse_progress(path.read_text())["done"] == str(len(FORMAT_NAMES))

    def test_loads_the_adapter_from_the_corpus_directory(
        self, monkeypatch, demo_root, tmp_path
    ):
        seen = {}
        monkeypatch.setattr(
            score, "load_model", lambda *_a, **kw: seen.update(kw) or object()
        )
        adapter = adapter_dir(tmp_path / "nq", SLUG, "5fmt_lam1")
        adapter.mkdir(parents=True)
        run(
            monkeypatch,
            score,
            *score_args(demo_root, tmp_path / "nq", "--arm", "5fmt_lam1"),
        )
        assert seen["lora_adapter_path"] == str(adapter)

    def test_refuses_an_arm_with_no_adapter(self, monkeypatch, demo_root, tmp_path):
        with pytest.raises(SystemExit, match="no adapter at"):
            run(
                monkeypatch,
                score,
                *score_args(demo_root, tmp_path / "nq", "--arm", "5fmt_lam1"),
            )

    def test_refuses_without_pools(self, monkeypatch, tmp_path):
        with pytest.raises(SystemExit, match="no pools"):
            run(monkeypatch, score, *score_args(tmp_path / "none", tmp_path / "nq"))

    def test_refuses_pools_that_would_be_truncated(
        self, monkeypatch, demo_root, tmp_path
    ):
        monkeypatch.setattr(
            score, "overlong", lambda *_a, **_k: [("demo00000", 0, "yaml")]
        )
        with pytest.raises(SystemExit, match="would be truncated"):
            run(monkeypatch, score, *score_args(demo_root, tmp_path / "nq"))

    def test_skips_a_present_tensor(self, monkeypatch, demo_root, tmp_path, capsys):
        run(monkeypatch, score, *score_args(demo_root, tmp_path / "nq"))
        run(monkeypatch, score, *score_args(demo_root, tmp_path / "nq"))
        assert "is present" in capsys.readouterr().out


def write_selection(data_root, winner="1"):
    path = h2_selection_path(data_root, SLUG, "lambda")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"status": "selected", "winner_lambda": winner}))


def write_tensors(demo_root, gold_by_arm):
    """Score every pool alike, with three negatives that peak in markdown.

    The mixed assignment is set to yaml throughout, so a gold passage that
    leads only in markdown is lost under every handling yet rescued by
    re-rendering it alone.
    """
    path = scenarios_path(demo_root)
    stored = json.loads(path.read_text())
    for pool in stored["scenarios"]:
        pool["mixed"] = ["yaml"] * len(pool["candidates"])
    path.write_text(json.dumps(stored))
    pools = stored["scenarios"]
    for arm_name, gold in gold_by_arm.items():
        scores = np.full((N, 10, len(FORMAT_NAMES)), 0.5, dtype=np.float32)
        scores[:, 0] = gold
        scores[:, 1:4] = [0.6, 0.6, 0.6, 0.6, 0.95]
        save_tensor(
            demo_root,
            ScoreTensor(
                scores=scores,
                query_ids=[p["id"] for p in pools],
                formats=list(FORMAT_NAMES),
                model=SLUG,
                arm=arm_name,
                split="demo",
            ),
        )


class TestSelect:
    def select_args(self, demo_root, data_root, *extra):
        return ["--demo-root", str(demo_root), "--data-root", str(data_root), *extra]

    def test_writes_the_selection_and_exports_the_first_match(
        self, monkeypatch, demo_root, tmp_path
    ):
        data_root = tmp_path / "nq"
        write_selection(data_root)
        losing = np.array([0.55, 0.1, 0.1, 0.1, 0.9])
        write_tensors(
            demo_root, {"base": losing, "5fmt_lam0": losing, "5fmt_lam1": 0.99}
        )
        run(monkeypatch, select_script, *self.select_args(demo_root, data_root))
        result = json.loads(selection_path(demo_root, SLUG).read_text())
        assert len(result["matches"]) == N
        first = result["matches"][0]["id"]
        assert topk_path(demo_root, SLUG, first).exists()

    def test_says_when_nothing_matched(self, monkeypatch, demo_root, tmp_path, capsys):
        data_root = tmp_path / "nq"
        write_selection(data_root)
        write_tensors(demo_root, {"base": 0.99, "5fmt_lam0": 0.99, "5fmt_lam1": 0.99})
        run(monkeypatch, select_script, *self.select_args(demo_root, data_root))
        assert "No pool matched" in capsys.readouterr().out

    def test_exports_a_named_pool(self, monkeypatch, demo_root, tmp_path):
        data_root = tmp_path / "nq"
        write_selection(data_root)
        write_tensors(demo_root, {"base": 0.99, "5fmt_lam0": 0.99, "5fmt_lam1": 0.99})
        run(
            monkeypatch,
            select_script,
            *self.select_args(demo_root, data_root, "--scenario", "demo00002"),
        )
        assert topk_path(demo_root, SLUG, "demo00002").exists()

    def test_refuses_a_zero_weight(self, monkeypatch, demo_root, tmp_path):
        data_root = tmp_path / "nq"
        write_selection(data_root, winner="0")
        with pytest.raises(SystemExit, match="weight of zero"):
            run(monkeypatch, select_script, *self.select_args(demo_root, data_root))


class TestLoopWorker:
    @pytest.fixture
    def stub_worker(self, monkeypatch):
        """Keep the worker away from the network and the GPU."""
        monkeypatch.setattr(
            loop_worker, "load_tokenizer", lambda *_a, **_k: WordTokenizer()
        )
        monkeypatch.setattr(loop_worker, "load_model", lambda *_a, **_k: object())
        monkeypatch.setattr(
            loop_worker,
            "score_batch",
            lambda _m, _t, pairs, *_a, **_k: [float(len(p[1])) for p in pairs],
        )

    def worker_args(self, demo_root, data_root, *extra):
        return [
            "--demo-root",
            str(demo_root),
            "--data-root",
            str(data_root),
            "--arms",
            "base,5fmt_lam1",
            "--idle-minutes",
            "0",
            "--poll",
            "0",
            *extra,
        ]

    @pytest.mark.usefixtures("stub_worker")
    def test_scores_a_pushed_job_and_stops_idle(self, monkeypatch, tmp_path):
        demo, nq = tmp_path / "demo", tmp_path / "nq"
        adapter_dir(nq, SLUG, "5fmt_lam1").mkdir(parents=True)
        inbox = inbox_dir(demo)
        inbox.mkdir(parents=True)
        pools = [
            {
                "id": "p0",
                "question": "how big",
                "candidates": [{"pairs": [["title", "x"]], "truncated_body": "b"}],
            }
        ]
        (inbox / "r1.json").write_text(json.dumps({"pools": pools}))
        (inbox / "r1.ready").write_text("")
        progress = tmp_path / "w.progress"
        run(
            monkeypatch,
            loop_worker,
            *self.worker_args(demo, nq, "--progress-file", str(progress)),
        )
        outbox = outbox_dir(demo)
        assert (outbox / "r1.done").exists()
        with np.load(outbox / "r1.npz") as saved:
            assert set(saved.files) == {"scores_base", "scores_5fmt_lam1"}
        assert parse_progress(progress.read_text())["done"] == "2"

    def test_refuses_an_arm_with_no_adapter(self, monkeypatch, tmp_path):
        with pytest.raises(SystemExit, match="no adapter at"):
            run(
                monkeypatch,
                loop_worker,
                *self.worker_args(tmp_path / "demo", tmp_path / "nq"),
            )
