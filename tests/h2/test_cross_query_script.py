"""Tests for scripts.h2.cross_query."""

from __future__ import annotations

import json

import numpy as np
import pytest
from scripts.h2 import cross_query as mod

from fsr.corpus.layout import NEGATIVES_NAME, split_dir
from fsr.formats import FORMAT_NAMES
from fsr.h2_layout import BASE_ARM, adapter_dir, result_path
from fsr.models.registry import BASE_MODEL_IDS, by_slug
from fsr.reporting import parse_progress
from tests.fakes import HashingPairTokenizer, ScoringModel

SLUG = "minilm_l6"
NEG = mod.MRR_NEG_COUNT


def record(rec_id):
    return {
        "id": str(rec_id),
        "title": f"Article {rec_id}",
        "question": f"who built bridge {rec_id}",
        "pairs": [["Born", "1946"], ["Role", "Engineer"]],
        "body": f"The bridge {rec_id} opened in 1946 and carries the road here. " * 6,
        "quality_flags": [],
    }


@pytest.fixture
def data_root(tmp_path):
    root = tmp_path / "nq"
    splits = split_dir(root)
    splits.mkdir(parents=True)
    ids = {
        "train": [f"t{i}" for i in range(NEG + 4)],
        "dev": [f"d{i}" for i in range(4)],
        "test": [f"s{i}" for i in range(4)],
        "nq_val": [f"v{i}" for i in range(2)],
    }
    for name, names in ids.items():
        (splits / f"{name}.json").write_text(
            json.dumps({"records": [record(i) for i in names]})
        )
    pool = ids["train"] + ids["dev"] + ids["test"]
    (splits / NEGATIVES_NAME).write_text(
        json.dumps(
            {"negatives": {rid: [o for o in pool if o != rid][:NEG] for rid in pool}}
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
            "cross_query.py",
            "--model",
            model,
            "--data-root",
            str(data_root),
            "--batch-size",
            "8",
            "--n-boot",
            "50",
            *extra,
        ],
    )
    mod.main()


def payload(data_root, arm=BASE_ARM, split="test", slug=SLUG):
    return json.loads(result_path(data_root, split, "cross", slug, arm).read_text())


def make_adapter(data_root, arm_name, slug=SLUG):
    path = adapter_dir(data_root, slug, arm_name)
    path.mkdir(parents=True)
    return path


class TestArmOf:
    def test_a_baseline_run_names_the_untrained_arm(self):
        args = mod.build_parser().parse_args(["--model", SLUG, "--baseline"])
        assert mod.arm_of(args) == BASE_ARM

    def test_a_trained_run_names_the_fold_and_the_weight(self):
        args = mod.build_parser().parse_args(
            ["--model", SLUG, "--held-out-format", "yaml", "--lambda-inv", "0.1"]
        )
        assert mod.arm_of(args) == "yaml_lam0.1"

    def test_the_five_format_phase_has_no_fold(self):
        args = mod.build_parser().parse_args(["--model", SLUG, "--lambda-inv", "0"])
        assert mod.arm_of(args) == "5fmt_lam0"

    def test_a_rank_sweep_carries_the_rank(self):
        args = mod.build_parser().parse_args(
            ["--model", SLUG, "--lambda-inv", "0.1", "--rank-tag", "8"]
        )
        assert mod.arm_of(args) == "5fmt_lam0.1_r8"


class TestRendering:
    def test_renders_the_budgeted_positive(self):
        prepared = {"pairs": [("Born", "1946")], "truncated_body": "Short body."}
        assert "Short body." in mod.render_positive(prepared, "yaml")

    def test_renders_every_format(self):
        prepared = {"pairs": [("Born", "1946")], "truncated_body": "Body."}
        rendered = {mod.render_positive(prepared, f) for f in FORMAT_NAMES}
        assert len(rendered) == len(FORMAT_NAMES)

    def test_cuts_a_negative_to_the_query_budget(self):
        negative = {"pairs": [("Born", "1801")], "body": "One two three four. " * 40}
        out = mod.render_negative(negative, "yaml", 20, HashingPairTokenizer(4))
        assert len(out.split()) < 60

    def test_gives_a_negative_with_no_body_an_empty_one(self):
        negative = {"pairs": [("Born", "1801")], "body": ""}
        assert mod.render_negative(negative, "yaml", 40, HashingPairTokenizer(4))


class TestDefaults:
    def test_takes_the_batch_size_from_the_registry(self):
        args = mod.build_parser().parse_args(["--model", SLUG])
        assert args.batch_size is None
        assert by_slug(SLUG).eval_batch == 32

    def test_evaluates_the_test_split_by_default(self):
        assert mod.build_parser().parse_args(["--model", SLUG]).split == "test"

    def test_offers_only_the_two_phase_splits(self):
        assert mod.EVAL_SPLITS == ("dev", "test")

    def test_rejects_a_split_the_phase_does_not_use(self):
        with pytest.raises(SystemExit):
            mod.build_parser().parse_args(["--model", SLUG, "--split", "nq_val"])

    def test_keeps_the_published_negative_count(self):
        assert mod.MRR_NEG_COUNT == 15


class TestBaselineRun:
    @pytest.mark.usefixtures("fake_model")
    def test_writes_the_untrained_result(self, monkeypatch, data_root):
        run(monkeypatch, data_root, "--baseline")
        assert payload(data_root)["arm"] == BASE_ARM

    @pytest.mark.usefixtures("fake_model")
    def test_loads_no_adapter(self, monkeypatch, data_root, fake_model):
        run(monkeypatch, data_root, "--baseline")
        assert fake_model["lora_adapter_path"] is None

    @pytest.mark.usefixtures("fake_model")
    def test_records_no_adapter_in_the_payload(self, monkeypatch, data_root):
        run(monkeypatch, data_root, "--baseline")
        assert payload(data_root)["adapter"] is None

    @pytest.mark.usefixtures("fake_model")
    def test_names_the_model_by_its_slug(self, monkeypatch, data_root):
        run(monkeypatch, data_root, "--baseline")
        assert payload(data_root)["model"] == SLUG

    @pytest.mark.usefixtures("fake_model")
    def test_covers_the_dev_split_too(self, monkeypatch, data_root):
        run(monkeypatch, data_root, "--baseline", "--split", "dev")
        assert payload(data_root, split="dev")["split"] == "dev"


class TestTrainedRun:
    @pytest.mark.usefixtures("fake_model")
    def test_reads_the_adapter_of_the_arm(self, monkeypatch, data_root, fake_model):
        made = make_adapter(data_root, "yaml_lam0.1")
        run(monkeypatch, data_root, "--held-out-format", "yaml", "--lambda-inv", "0.1")
        assert fake_model["lora_adapter_path"] == str(made)

    @pytest.mark.usefixtures("fake_model")
    def test_writes_the_result_under_the_same_arm(self, monkeypatch, data_root):
        make_adapter(data_root, "yaml_lam0.1")
        run(monkeypatch, data_root, "--held-out-format", "yaml", "--lambda-inv", "0.1")
        assert payload(data_root, "yaml_lam0.1")["arm"] == "yaml_lam0.1"

    @pytest.mark.usefixtures("fake_model")
    def test_refuses_an_arm_that_was_never_trained(self, monkeypatch, data_root):
        with pytest.raises(SystemExit, match="no adapter at"):
            run(monkeypatch, data_root, "--lambda-inv", "0.1")

    @pytest.mark.usefixtures("fake_model")
    def test_passes_the_registry_head_choice(self, monkeypatch, data_root, fake_model):
        run(monkeypatch, data_root, "--baseline")
        assert fake_model["tanh_head"] is False

    @pytest.mark.usefixtures("fake_model")
    def test_an_explicit_head_flag_wins(self, monkeypatch, data_root, fake_model):
        run(monkeypatch, data_root, "--baseline", "--tanh-head")
        assert fake_model["tanh_head"] is True

    @pytest.mark.usefixtures("fake_model")
    def test_the_tanh_variant_needs_no_flag(self, monkeypatch, data_root, fake_model):
        run(monkeypatch, data_root, "--baseline", model="mxbai_v1_tanh")
        assert fake_model["tanh_head"] is True


class TestPayload:
    @pytest.fixture(autouse=True)
    def _written(self, request, monkeypatch, data_root):
        request.getfixturevalue("fake_model")
        run(monkeypatch, data_root, "--baseline")
        self.out = payload(data_root)

    def test_holds_the_score_axis_summary(self):
        assert "max_abs_cohen_d" in self.out["format_sensitivity"]["summary"]

    def test_holds_an_interval_on_the_largest_effect(self):
        low, high = self.out["max_abs_d_ci95"]
        assert low <= self.out["format_sensitivity"]["summary"]["max_abs_cohen_d"]
        assert high >= low

    def test_holds_the_raw_scores_of_every_format(self):
        assert set(self.out["scores_per_fmt"]) == set(FORMAT_NAMES)

    def test_holds_the_record_identifiers(self):
        assert self.out["record_ids"]
        assert len(self.out["record_ids"]) == self.out["n_records_kept"]

    def test_holds_the_body_budget_statistics(self):
        assert "dropped" in self.out["body_budget_stats"]

    def test_holds_the_ranking_guardrail(self):
        assert set(self.out["mrr_guardrail"]["per_format_mrr"]) == set(FORMAT_NAMES)

    def test_the_guardrail_names_its_records(self):
        assert self.out["mrr_guardrail"]["record_ids"]

    def test_the_guardrail_reports_the_worst_format(self):
        guard = self.out["mrr_guardrail"]
        assert guard["min_mrr"] == min(guard["per_format_mrr"].values())

    def test_the_guardrail_keeps_the_published_negative_count(self):
        assert self.out["mrr_guardrail"]["neg_count_per_query"] == NEG


class TestScoringWithoutACounter:
    def test_scores_the_formats(self):
        prepared = [
            {
                "question": "who",
                "pairs": [("Born", "1946")],
                "truncated_body": "Body.",
            }
        ]
        scores = mod.score_formats(
            ScoringModel(4), HashingPairTokenizer(4), prepared, 4, "cpu"
        )
        assert set(scores) == set(FORMAT_NAMES)

    @staticmethod
    def guardrail_inputs():
        prepared = [
            {
                "id": "a",
                "question": "who",
                "pairs": [("Born", "1946")],
                "truncated_body": "Body.",
                "body_budget_tokens": 40,
                "tightest_tokeniser": "tight",
            }
        ]
        corpus = {
            f"n{i}": {"id": f"n{i}", "pairs": [("Born", "1801")], "body": "Other."}
            for i in range(NEG)
        }
        return prepared, corpus

    def test_scores_the_guardrail(self):
        prepared, corpus = self.guardrail_inputs()
        mrr, rr = mod.score_guardrail(
            ScoringModel(4),
            HashingPairTokenizer(4),
            prepared,
            {"a": list(corpus)},
            corpus,
            {"tight": HashingPairTokenizer(4)},
            4,
            "cpu",
        )
        assert set(mrr) == set(FORMAT_NAMES)
        assert all(len(v) == 1 for v in rr.values())

    def test_cuts_negatives_with_the_budget_tokenizer(self, monkeypatch):
        prepared, corpus = self.guardrail_inputs()
        budget = HashingPairTokenizer(4)
        scoring = HashingPairTokenizer(8)
        seen = []
        original = mod.render_negative

        def spy(record, format_name, token_budget, budget_tokenizer):
            seen.append(budget_tokenizer)
            return original(record, format_name, token_budget, budget_tokenizer)

        monkeypatch.setattr(mod, "render_negative", spy)
        mod.score_guardrail(
            ScoringModel(8),
            scoring,
            prepared,
            {"a": list(corpus)},
            corpus,
            {"tight": budget},
            4,
            "cpu",
        )
        assert seen
        assert all(t is budget for t in seen)


class TestRecordsWithoutNegatives:
    @pytest.mark.usefixtures("fake_model")
    def test_reports_the_records_it_cannot_score(self, monkeypatch, data_root, capsys):
        path = split_dir(data_root) / NEGATIVES_NAME
        negatives = json.loads(path.read_text())["negatives"]
        dropped = sorted(k for k in negatives if k.startswith("s"))[0]
        del negatives[dropped]
        path.write_text(json.dumps({"negatives": negatives}))
        run(monkeypatch, data_root, "--baseline")
        assert "1 records have no negatives" in capsys.readouterr().out

    @pytest.mark.usefixtures("fake_model")
    def test_the_guardrail_covers_fewer_records_than_the_score_axis(
        self, monkeypatch, data_root
    ):
        path = split_dir(data_root) / NEGATIVES_NAME
        negatives = json.loads(path.read_text())["negatives"]
        del negatives[sorted(k for k in negatives if k.startswith("s"))[0]]
        path.write_text(json.dumps({"negatives": negatives}))
        run(monkeypatch, data_root, "--baseline")
        out = payload(data_root)
        assert len(out["mrr_guardrail"]["record_ids"]) < len(out["record_ids"])


class TestGuardrailPayload:
    def test_pools_every_format_for_the_interval(self):
        rr = {name: [0.5, 1.0] for name in FORMAT_NAMES}
        out = mod.guardrail_payload({n: 0.75 for n in FORMAT_NAMES}, rr, ["a"], 0)
        low, high = out["all_format_rr_mean_ci95"]
        assert low <= 0.75 <= high

    def test_reports_the_mean_over_formats(self):
        mrr = {name: float(i) for i, name in enumerate(FORMAT_NAMES)}
        rr = {name: [0.5] for name in FORMAT_NAMES}
        out = mod.guardrail_payload(mrr, rr, [], 0)
        assert out["mean_mrr"] == np.mean(list(mrr.values()))


class TestSkipMrr:
    @pytest.mark.usefixtures("fake_model")
    def test_writes_no_guardrail_on_request(self, monkeypatch, data_root):
        run(monkeypatch, data_root, "--baseline", "--no-mrr")
        assert payload(data_root)["mrr_guardrail"] is None

    @pytest.mark.usefixtures("fake_model")
    def test_still_writes_the_score_axis(self, monkeypatch, data_root):
        run(monkeypatch, data_root, "--baseline", "--no-mrr")
        assert payload(data_root)["scores_per_fmt"]


class TestSkipAndForce:
    @pytest.mark.usefixtures("fake_model")
    def test_skips_a_finished_evaluation(self, monkeypatch, data_root, capsys):
        run(monkeypatch, data_root, "--baseline")
        capsys.readouterr()
        run(monkeypatch, data_root, "--baseline")
        assert "Skipping" in capsys.readouterr().out

    @pytest.mark.usefixtures("fake_model")
    def test_force_evaluates_again(self, monkeypatch, data_root, capsys):
        run(monkeypatch, data_root, "--baseline")
        capsys.readouterr()
        run(monkeypatch, data_root, "--baseline", "--force")
        assert "Skipping" not in capsys.readouterr().out


class TestProgress:
    @pytest.mark.usefixtures("fake_model")
    def test_counts_every_format_twice_with_the_guardrail(
        self, monkeypatch, data_root, tmp_path
    ):
        path = tmp_path / "job.progress"
        run(monkeypatch, data_root, "--baseline", "--progress-file", str(path))
        fields = parse_progress(path.read_text())
        assert fields["total"] == str(2 * len(FORMAT_NAMES))
        assert fields["done"] == fields["total"]

    @pytest.mark.usefixtures("fake_model")
    def test_counts_every_format_once_without_it(
        self, monkeypatch, data_root, tmp_path
    ):
        path = tmp_path / "job.progress"
        run(
            monkeypatch,
            data_root,
            "--baseline",
            "--no-mrr",
            "--progress-file",
            str(path),
        )
        assert parse_progress(path.read_text())["total"] == str(len(FORMAT_NAMES))


class TestEmptySplit:
    @pytest.mark.usefixtures("fake_model")
    def test_stops_when_no_record_survives_the_budget(self, monkeypatch, data_root):
        monkeypatch.setattr(
            mod, "prepare_records_with_body", lambda *_a, **_k: ([], {"dropped": 4})
        )
        with pytest.raises(SystemExit, match="nothing to score"):
            run(monkeypatch, data_root, "--baseline")


class TestLimit:
    @pytest.mark.usefixtures("fake_model")
    def test_caps_the_records_on_request(self, monkeypatch, data_root):
        run(monkeypatch, data_root, "--baseline", "--limit", "2")
        assert payload(data_root)["n_records_input"] == 2


class TestResampleCount:
    def test_defaults_to_the_published_count(self):
        from fsr.metrics import DEFAULT_N_BOOT

        args = mod.build_parser().parse_args(["--model", SLUG])
        assert args.n_boot == DEFAULT_N_BOOT == 10_000

    @pytest.mark.usefixtures("fake_model")
    def test_records_the_count_it_used(self, monkeypatch, data_root):
        run(monkeypatch, data_root, "--baseline")
        assert payload(data_root)["n_boot"] == 50


PARSE_ARGS = ["--model", SLUG, "--baseline"]


class TestBudgetModels:
    def test_defaults_to_the_whole_roster(self):
        args = mod.build_parser().parse_args(PARSE_ARGS)
        assert args.budget_models == list(BASE_MODEL_IDS)

    def test_the_default_is_more_than_the_model_itself(self):
        args = mod.build_parser().parse_args(PARSE_ARGS)
        assert len(args.budget_models) > 1

    def test_loads_every_tokenizer(self, monkeypatch):
        seen = []
        monkeypatch.setattr(
            mod, "load_tokenizer", lambda mid: seen.append(mid) or object()
        )
        loaded = mod.load_budget_tokenizers(["minilm_l6", "bge_base"])
        assert len(loaded) == 2
        assert seen == [
            "cross-encoder/ms-marco-MiniLM-L6-v2",
            "BAAI/bge-reranker-base",
        ]

    def test_loads_a_repeated_model_once(self, monkeypatch):
        seen = []
        monkeypatch.setattr(
            mod, "load_tokenizer", lambda mid: seen.append(mid) or object()
        )
        mod.load_budget_tokenizers(["minilm_l6", "minilm_l6"])
        assert len(seen) == 1

    def test_refuses_when_a_tokenizer_does_not_load(self, monkeypatch):
        def fail(_mid):
            raise OSError("no network")

        monkeypatch.setattr(mod, "load_tokenizer", fail)
        with pytest.raises(SystemExit, match="did not load"):
            mod.load_budget_tokenizers(["minilm_l6"])

    def test_the_refusal_says_why_it_matters(self, monkeypatch):
        def fail(_mid):
            raise OSError("no network")

        monkeypatch.setattr(mod, "load_tokenizer", fail)
        with pytest.raises(SystemExit, match="whole roster"):
            mod.load_budget_tokenizers(["minilm_l6"])
