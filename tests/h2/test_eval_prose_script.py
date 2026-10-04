"""Tests for scripts.h2.eval_prose."""

from __future__ import annotations

import json

import pytest
from scripts.h2 import eval_prose as mod
from tests.fakes import HashingPairTokenizer, ScoringModel

from fsr.corpus.layout import prose_eval_path
from fsr.corpus.prose import PROSE_NEGATIVES
from fsr.h2_layout import adapter_dir, result_path
from fsr.models.registry import by_slug
from fsr.reporting import parse_progress

SLUG = "minilm_l6"
ARM = "5fmt_lam0.1"
K = 3
N = 12


def ranking_record(rid):
    return {
        "id": f"r{rid}",
        "title": f"Article {rid}",
        "question": f"who built bridge {rid}",
        "positive": {
            "id": f"r{rid}",
            "title": f"Article {rid}",
            "text": f"The bridge {rid} opened in 1946.",
        },
        "negatives": [
            {
                "id": f"n{rid}-{j}",
                "title": f"Other {j}",
                "text": f"Unrelated passage {j} about lichen.",
            }
            for j in range(K)
        ],
        "short_answers": ["1946"],
    }


@pytest.fixture
def data_root(tmp_path):
    root = tmp_path / "nq"
    path = prose_eval_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "n_records": N,
                "k_negatives": K,
                "records": [ranking_record(i) for i in range(N)],
            }
        )
    )
    return root


@pytest.fixture
def fake_model(monkeypatch):
    """Replace the loaders, so no download and no device are needed."""
    seen: dict = {}

    def fake_load_model(model_id, device, **kwargs):
        seen["model_id"] = model_id
        seen["device"] = device
        seen.update(kwargs)
        return ScoringModel(4)

    monkeypatch.setattr(mod, "load_model", fake_load_model)
    monkeypatch.setattr(mod, "load_tokenizer", lambda _id: HashingPairTokenizer(4))
    return seen


def run(monkeypatch, data_root, *extra, model=SLUG):
    monkeypatch.setattr(
        "sys.argv",
        [
            "eval_prose.py",
            "--model",
            model,
            "--data-root",
            str(data_root),
            "--batch-size",
            "8",
            *extra,
        ],
    )
    mod.main()


def payload(data_root, arm="base"):
    return json.loads(result_path(data_root, "prose", "mrr", SLUG, arm).read_text())


class TestRankingPairs:
    def test_puts_the_gold_first_in_each_record(self):
        pairs = mod.ranking_pairs([ranking_record(0)])
        assert pairs[0][1].startswith("The bridge")

    def test_follows_the_gold_with_the_negatives(self):
        pairs = mod.ranking_pairs([ranking_record(0)])
        assert all("lichen" in p[1] for p in pairs[1:])

    def test_builds_one_block_per_record(self):
        pairs = mod.ranking_pairs([ranking_record(i) for i in range(4)])
        assert len(pairs) == 4 * (K + 1)

    def test_repeats_the_question_across_the_block(self):
        pairs = mod.ranking_pairs([ranking_record(0)])
        assert len({p[0] for p in pairs}) == 1


class TestScoreProse:
    def test_returns_the_gold_and_the_negatives(self):
        records = [ranking_record(i) for i in range(4)]
        gold, negatives = mod.score_prose(
            ScoringModel(4), HashingPairTokenizer(4), records, K, 8, "cpu"
        )
        assert gold.shape == (4,)
        assert negatives.shape == (4, K)

    def test_refuses_a_record_with_the_wrong_negative_count(self):
        records = [ranking_record(0)]
        with pytest.raises(ValueError, match="expected"):
            mod.score_prose(
                ScoringModel(4), HashingPairTokenizer(4), records, K + 1, 8, "cpu"
            )

    def test_runs_without_a_counter(self):
        records = [ranking_record(i) for i in range(2)]
        gold, _ = mod.score_prose(
            ScoringModel(4), HashingPairTokenizer(4), records, K, 4, "cpu"
        )
        assert len(gold) == 2


class TestRun:
    @pytest.mark.usefixtures("fake_model")
    def test_writes_the_baseline_result(self, monkeypatch, data_root):
        run(monkeypatch, data_root, "--baseline")
        assert payload(data_root)["arm"] == "base"

    @pytest.mark.usefixtures("fake_model")
    def test_reports_a_reciprocal_rank_for_every_record(self, monkeypatch, data_root):
        run(monkeypatch, data_root, "--baseline")
        out = payload(data_root)
        assert len(out["reciprocal_ranks"]) == N
        assert len(out["record_ids"]) == N

    @pytest.mark.usefixtures("fake_model")
    def test_reports_the_mean_reciprocal_rank(self, monkeypatch, data_root):
        run(monkeypatch, data_root, "--baseline")
        out = payload(data_root)
        assert out["mrr"] == pytest.approx(
            sum(out["reciprocal_ranks"]) / len(out["reciprocal_ranks"])
        )

    @pytest.mark.usefixtures("fake_model")
    def test_records_the_negative_count(self, monkeypatch, data_root):
        run(monkeypatch, data_root, "--baseline")
        assert payload(data_root)["k_negatives"] == K

    @pytest.mark.usefixtures("fake_model")
    def test_records_the_gold_and_negative_scores(self, monkeypatch, data_root):
        run(monkeypatch, data_root, "--baseline")
        out = payload(data_root)
        assert len(out["gold_scores"]) == N
        assert len(out["neg_score_means"]) == N

    def test_loads_no_adapter_for_the_baseline(
        self, monkeypatch, data_root, fake_model
    ):
        run(monkeypatch, data_root, "--baseline")
        assert fake_model["lora_adapter_path"] is None

    def test_loads_the_adapter_of_the_arm(self, monkeypatch, data_root, fake_model):
        made = adapter_dir(data_root, SLUG, ARM)
        made.mkdir(parents=True)
        run(monkeypatch, data_root, "--arm", ARM)
        assert fake_model["lora_adapter_path"] == str(made)

    @pytest.mark.usefixtures("fake_model")
    def test_writes_the_arm_result(self, monkeypatch, data_root):
        adapter_dir(data_root, SLUG, ARM).mkdir(parents=True)
        run(monkeypatch, data_root, "--arm", ARM)
        assert payload(data_root, ARM)["arm"] == ARM

    @pytest.mark.usefixtures("fake_model")
    def test_refuses_an_arm_that_was_never_trained(self, monkeypatch, data_root):
        with pytest.raises(SystemExit, match="no adapter at"):
            run(monkeypatch, data_root, "--arm", ARM)

    def test_takes_the_head_choice_from_the_registry(
        self, monkeypatch, data_root, fake_model
    ):
        run(monkeypatch, data_root, "--baseline")
        assert fake_model["tanh_head"] is False

    def test_the_tanh_variant_needs_no_flag(self, monkeypatch, data_root, fake_model):
        monkeypatch.setattr(
            "sys.argv",
            [
                "eval_prose.py",
                "--model",
                "mxbai_v1_tanh",
                "--data-root",
                str(data_root),
                "--batch-size",
                "8",
                "--baseline",
            ],
        )
        mod.main()
        assert fake_model["tanh_head"] is True

    @pytest.mark.usefixtures("fake_model")
    def test_caps_the_records_on_request(self, monkeypatch, data_root):
        run(monkeypatch, data_root, "--baseline", "--limit", "4")
        out = payload(data_root)
        assert out["n_records_kept"] == 4
        assert out["n_records_input"] == N

    def test_takes_the_batch_size_from_the_registry(self):
        args = mod.build_parser().parse_args(["--model", SLUG, "--baseline"])
        assert args.batch_size is None
        assert by_slug(SLUG).eval_batch == 32


class TestArmSelection:
    @pytest.mark.usefixtures("fake_model")
    def test_refuses_neither(self, monkeypatch, data_root):
        with pytest.raises(SystemExit, match="not both or neither"):
            run(monkeypatch, data_root)

    @pytest.mark.usefixtures("fake_model")
    def test_refuses_both(self, monkeypatch, data_root):
        with pytest.raises(SystemExit, match="not both or neither"):
            run(monkeypatch, data_root, "--baseline", "--arm", ARM)


class TestMissingSource:
    @pytest.mark.usefixtures("fake_model")
    def test_refuses_without_the_ranking_set(self, monkeypatch, tmp_path):
        with pytest.raises(SystemExit, match="build the corpus first"):
            run(monkeypatch, tmp_path / "nq", "--baseline")


class TestUnexpectedNegativeCount:
    @pytest.mark.usefixtures("fake_model")
    def test_says_so_without_stopping(self, monkeypatch, data_root, capsys):
        run(monkeypatch, data_root, "--baseline")
        assert f"carries {K} negatives" in capsys.readouterr().out

    @pytest.mark.usefixtures("fake_model")
    def test_stays_quiet_at_the_published_count(self, monkeypatch, tmp_path, capsys):
        root = tmp_path / "nq"
        path = prose_eval_path(root)
        path.parent.mkdir(parents=True, exist_ok=True)
        records = []
        for i in range(4):
            record = ranking_record(i)
            record["negatives"] = [
                {
                    "id": f"n{i}-{j}",
                    "title": f"Other {j}",
                    "text": f"Unrelated passage {j}.",
                }
                for j in range(PROSE_NEGATIVES)
            ]
            records.append(record)
        path.write_text(
            json.dumps(
                {
                    "n_records": len(records),
                    "k_negatives": PROSE_NEGATIVES,
                    "records": records,
                }
            )
        )
        run(monkeypatch, root, "--baseline")
        assert "carries" not in capsys.readouterr().out


class TestProgress:
    @pytest.mark.usefixtures("fake_model")
    def test_counts_every_batch(self, monkeypatch, data_root, tmp_path):
        path = tmp_path / "job.progress"
        run(monkeypatch, data_root, "--baseline", "--progress-file", str(path))
        fields = parse_progress(path.read_text())
        assert fields["total"] == str(-(-N * (K + 1) // 8))
        assert fields["done"] == fields["total"]

    @pytest.mark.usefixtures("fake_model")
    def test_writes_no_progress_file_by_default(self, monkeypatch, data_root, tmp_path):
        run(monkeypatch, data_root, "--baseline")
        assert not list(tmp_path.glob("*.progress"))


class TestSkipAndForce:
    @pytest.mark.usefixtures("fake_model")
    def test_skips_a_finished_run(self, monkeypatch, data_root, capsys):
        run(monkeypatch, data_root, "--baseline")
        capsys.readouterr()
        run(monkeypatch, data_root, "--baseline")
        assert "Skipping" in capsys.readouterr().out

    @pytest.mark.usefixtures("fake_model")
    def test_force_ranks_again(self, monkeypatch, data_root, capsys):
        run(monkeypatch, data_root, "--baseline")
        capsys.readouterr()
        run(monkeypatch, data_root, "--baseline", "--force")
        assert "Skipping" not in capsys.readouterr().out
