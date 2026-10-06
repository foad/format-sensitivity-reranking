"""Tests for scripts.h2.capture_features."""

from __future__ import annotations

import json

import numpy as np
import pytest
from scripts.h2 import capture_features as mod

from fsr.corpus.layout import NEGATIVES_NAME, split_dir
from fsr.formats import FORMAT_NAMES
from fsr.h2_layout import feature_meta_path, feature_path, feature_score_path
from fsr.reporting import parse_progress
from tests.fakes import HashingPairTokenizer, ScoringModel

SLUG = "minilm_l6"
NEG = 3
WIDTH = 4


def record(rec_id):
    return {
        "id": str(rec_id),
        "title": f"Article {rec_id}",
        "question": f"who built bridge {rec_id}",
        "pairs": [["Born", "1946"], ["Role", "Engineer"]],
        "body": f"The bridge {rec_id} opened in 1946 and carries the road. " * 4,
        "quality_flags": [],
    }


@pytest.fixture
def data_root(tmp_path):
    root = tmp_path / "nq"
    splits = split_dir(root)
    splits.mkdir(parents=True)
    ids = {
        "train": [f"t{i}" for i in range(6)],
        "dev": [f"d{i}" for i in range(4)],
        "test": [f"s{i}" for i in range(2)],
        "nq_val": [f"v{i}" for i in range(2)],
    }
    for name, names in ids.items():
        (splits / f"{name}.json").write_text(
            json.dumps({"records": [record(i) for i in names]})
        )
    pool = ids["train"] + ids["dev"] + ids["test"]
    (splits / NEGATIVES_NAME).write_text(
        json.dumps(
            {
                "negatives": {
                    rid: [other for other in pool if other != rid][:NEG] for rid in pool
                }
            }
        )
    )
    return root


@pytest.fixture
def fake_model(monkeypatch):
    """Replace the loaders, so no download and no device are needed."""
    seen: dict = {}

    scorer = ScoringModel(WIDTH)

    def fake_load_model(model_id, _device, **kwargs):
        seen["model_id"] = model_id
        seen["model"] = scorer
        seen.update(kwargs)
        return scorer

    monkeypatch.setattr(mod, "load_model", fake_load_model)
    monkeypatch.setattr(mod, "load_tokenizer", lambda _id: HashingPairTokenizer(WIDTH))
    return seen


def run(monkeypatch, data_root, *extra, split="train"):
    monkeypatch.setattr(
        "sys.argv",
        [
            "capture_features.py",
            "--model",
            SLUG,
            "--split",
            split,
            "--data-root",
            str(data_root),
            "--negatives",
            str(NEG),
            "--budget-models",
            SLUG,
            *extra,
        ],
    )
    mod.main()


def meta(data_root, split="train"):
    return json.loads(feature_meta_path(data_root, split, SLUG).read_text())


class TestCandidatePairs:
    def test_puts_the_gold_first_in_each_group(self):
        class Rec:
            def __init__(self):
                self.question = "who"
                self.pairs = [("Born", "1946")]
                self.truncated_body = "body"
                self.neg_pairs_list = [[("Born", "1800")], [("Born", "1900")]]
                self.neg_bodies_truncated = ["one", "two"]

        pairs = mod.candidate_pairs([Rec()], "yaml")
        assert len(pairs) == 3
        assert "1946" in pairs[0][1]
        assert "1800" in pairs[1][1]

    def test_keeps_the_record_order(self):
        class Rec:
            def __init__(self, q):
                self.question = q
                self.pairs = [("Born", "1946")]
                self.truncated_body = "body"
                self.neg_pairs_list = [[("Born", "1800")]]
                self.neg_bodies_truncated = ["one"]

        pairs = mod.candidate_pairs([Rec("a"), Rec("b")], "yaml")
        assert [p[0] for p in pairs] == ["a", "a", "b", "b"]


class TestRun:
    @pytest.mark.usefixtures("fake_model")
    def test_writes_one_vector_per_candidate_and_format(self, monkeypatch, data_root):
        run(monkeypatch, data_root)
        features = np.load(feature_path(data_root, "train", SLUG))
        n = meta(data_root)["n_records"]
        assert features.shape == (n, 1 + NEG, len(FORMAT_NAMES), WIDTH)

    @pytest.mark.usefixtures("fake_model")
    def test_writes_a_score_for_every_vector(self, monkeypatch, data_root):
        run(monkeypatch, data_root)
        features = np.load(feature_path(data_root, "train", SLUG))
        scores = np.load(feature_score_path(data_root, "train", SLUG))
        assert scores.shape == features.shape[:3]

    def test_the_vectors_reproduce_the_scores(self, monkeypatch, data_root, fake_model):
        """The cache is only useful if a head replays the model's own score."""
        run(monkeypatch, data_root)
        features = np.load(feature_path(data_root, "train", SLUG))
        scores = np.load(feature_score_path(data_root, "train", SLUG))
        head = fake_model["model"].classifier
        weight = head.weight.detach().numpy()
        bias = head.bias.detach().numpy()
        replayed = features.reshape(-1, WIDTH) @ weight.T + bias
        assert np.allclose(replayed.ravel(), scores.ravel(), atol=1e-5)

    @pytest.mark.usefixtures("fake_model")
    def test_describes_what_it_cached(self, monkeypatch, data_root):
        run(monkeypatch, data_root)
        payload = meta(data_root)
        assert payload["model"] == SLUG
        assert payload["split"] == "train"
        assert payload["formats"] == list(FORMAT_NAMES)
        assert payload["n_negatives"] == NEG
        assert payload["feature_dim"] == WIDTH
        assert len(payload["record_ids"]) == payload["n_records"]

    @pytest.mark.usefixtures("fake_model")
    def test_covers_the_dev_split(self, monkeypatch, data_root):
        run(monkeypatch, data_root, split="dev")
        assert feature_path(data_root, "dev", SLUG).exists()
        assert meta(data_root, "dev")["split"] == "dev"

    @pytest.mark.usefixtures("fake_model")
    def test_caps_the_records_on_request(self, monkeypatch, data_root):
        run(monkeypatch, data_root, "--limit", "2")
        assert meta(data_root)["n_records"] <= 2

    @pytest.mark.usefixtures("fake_model")
    def test_records_the_budget_roster(self, monkeypatch, data_root):
        run(monkeypatch, data_root)
        assert meta(data_root)["budget_models"] == [
            "cross-encoder/ms-marco-MiniLM-L6-v2"
        ]

    def test_takes_the_tanh_head_from_the_registry(
        self, monkeypatch, data_root, fake_model
    ):
        run(monkeypatch, data_root)
        assert fake_model["tanh_head"] is False

    def test_uses_the_fused_kernel_by_default(self, monkeypatch, data_root, fake_model):
        run(monkeypatch, data_root)
        assert fake_model["eager_attn"] is False


class TestProgress:
    @pytest.mark.usefixtures("fake_model")
    def test_reports_one_unit_per_format(self, monkeypatch, data_root, tmp_path):
        path = tmp_path / "p.progress"
        run(monkeypatch, data_root, "--progress-file", str(path))
        state = parse_progress(path.read_text())
        assert state["done"] == str(len(FORMAT_NAMES))
        assert state["total"] == str(len(FORMAT_NAMES))


class TestSkipAndForce:
    @pytest.mark.usefixtures("fake_model")
    def test_skips_a_present_cache(self, monkeypatch, data_root, capsys):
        run(monkeypatch, data_root)
        run(monkeypatch, data_root)
        assert "is present" in capsys.readouterr().out

    @pytest.mark.usefixtures("fake_model")
    def test_force_caches_again(self, monkeypatch, data_root, capsys):
        run(monkeypatch, data_root)
        run(monkeypatch, data_root, "--force")
        assert "is present" not in capsys.readouterr().out


class TestBudgetModels:
    def test_refuses_when_a_tokenizer_does_not_load(self, monkeypatch):
        def fail(_mid):
            raise OSError("no network")

        monkeypatch.setattr(mod, "load_tokenizer", fail)
        with pytest.raises(SystemExit, match="did not load"):
            mod.load_budget_tokenizers(["minilm_l6"])

    def test_loads_each_identifier_once(self, monkeypatch):
        seen = []
        monkeypatch.setattr(
            mod, "load_tokenizer", lambda mid: seen.append(mid) or object()
        )
        mod.load_budget_tokenizers(["minilm_l6", "minilm_l6"])
        assert len(seen) == 1


class TestNoRecords:
    @pytest.mark.usefixtures("fake_model")
    def test_refuses_when_nothing_survives(self, monkeypatch, data_root):
        monkeypatch.setattr(mod, "prepare_train_records", lambda *_a, **_k: ([], 7))
        with pytest.raises(SystemExit, match="nothing to cache"):
            run(monkeypatch, data_root)
