"""Tests for scripts.h1.cross_query."""

from __future__ import annotations

import json

import pytest
import torch
from scripts.h1 import cross_query as mod
from tests.fakes import LogitModel, PairTokenizer

from fsr.corpus.layout import split_dir
from fsr.models.registry import BASE_MODEL_IDS, BASE_MODELS
from fsr.reporting import ProgressCounter, parse_progress, shorten

FORMAT_COUNT = 5


def record(rec_id="1", body="budgeted body text"):
    return {
        "id": rec_id,
        "question": f"question {rec_id}",
        "pairs": [("Born", "1946"), ("Role", "Engineer")],
        "truncated_body": body,
        "body_budget_tokens": 400,
    }


class TestTryLoadTokenizer:
    def test_returns_none_and_reports_a_failure(self, monkeypatch, capsys):
        def fail(*_args, **_kwargs):
            raise OSError("no such model")

        monkeypatch.setattr(mod.AutoTokenizer, "from_pretrained", fail)
        assert mod.try_load_tokenizer("missing/model", True) is None
        assert "tokenizer load failed" in capsys.readouterr().out

    def test_returns_the_tokenizer_on_success(self, monkeypatch):
        monkeypatch.setattr(
            mod.AutoTokenizer, "from_pretrained", lambda *_a, **_k: "TOKENIZER"
        )
        assert mod.try_load_tokenizer("some/model", True) == "TOKENIZER"

    def test_forwards_the_trust_flag(self, monkeypatch):
        seen = {}

        def capture(_name, **kwargs):
            seen.update(kwargs)
            return "TOKENIZER"

        monkeypatch.setattr(mod.AutoTokenizer, "from_pretrained", capture)
        mod.try_load_tokenizer("some/model", False)
        assert seen["trust_remote_code"] is False


class TestTryLoadModel:
    class Loaded:
        """A model stub that records the device and eval calls."""

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            self.evaluated = True
            return self

    def test_moves_the_model_to_the_device_and_evaluates(self, monkeypatch):
        monkeypatch.setattr(
            mod.AutoModelForSequenceClassification,
            "from_pretrained",
            lambda *_a, **_k: self.Loaded(),
        )
        model = mod.try_load_model("some/model", "cpu", True)
        assert model.device == "cpu"
        assert model.evaluated is True

    def test_requests_eager_attention_when_asked(self, monkeypatch):
        seen = {}

        def capture(_name, **kwargs):
            seen.update(kwargs)
            return self.Loaded()

        monkeypatch.setattr(
            mod.AutoModelForSequenceClassification, "from_pretrained", capture
        )
        mod.try_load_model("some/model", "cpu", True, force_eager_attn=True)
        assert seen["attn_implementation"] == "eager"

    def test_retries_without_eager_attention_on_a_type_error(self, monkeypatch):
        calls = []

        def capture(_name, **kwargs):
            calls.append(dict(kwargs))
            if "attn_implementation" in kwargs:
                raise TypeError("unexpected keyword")
            return self.Loaded()

        monkeypatch.setattr(
            mod.AutoModelForSequenceClassification, "from_pretrained", capture
        )
        assert mod.try_load_model("m", "cpu", True, force_eager_attn=True) is not None
        assert len(calls) == 2
        assert "attn_implementation" not in calls[1]

    def test_returns_none_when_the_retry_also_fails(self, monkeypatch, capsys):
        def capture(_name, **kwargs):
            if "attn_implementation" in kwargs:
                raise TypeError("unexpected keyword")
            raise OSError("corrupt weights")

        monkeypatch.setattr(
            mod.AutoModelForSequenceClassification, "from_pretrained", capture
        )
        assert mod.try_load_model("m", "cpu", True, force_eager_attn=True) is None
        assert "model load failed" in capsys.readouterr().out

    def test_returns_none_on_a_non_type_error(self, monkeypatch, capsys):
        def fail(*_args, **_kwargs):
            raise OSError("no such model")

        monkeypatch.setattr(
            mod.AutoModelForSequenceClassification, "from_pretrained", fail
        )
        assert mod.try_load_model("m", "cpu", True) is None
        assert "model load failed" in capsys.readouterr().out


class TestBudgetSummary:
    def test_summarises_the_budgets(self):
        records = [{"body_budget_tokens": b} for b in (10, 20, 30, 40)]
        assert mod.budget_summary(records) == {
            "budget_min": 10,
            "budget_median": 25,
            "budget_max": 40,
        }

    def test_truncates_a_fractional_median(self):
        records = [{"body_budget_tokens": b} for b in (10, 11)]
        assert mod.budget_summary(records)["budget_median"] == 10

    def test_reports_zeros_for_no_records(self):
        assert mod.budget_summary([]) == {
            "budget_min": 0,
            "budget_median": 0,
            "budget_max": 0,
        }


class TestBuildPairs:
    def test_includes_the_body_in_with_body_mode(self):
        pairs = mod.build_pairs([record()], mod.FORMATS["yaml"], "with_body")
        assert "budgeted body text" in pairs[0][1]

    def test_omits_the_body_in_metadata_only_mode(self):
        pairs = mod.build_pairs([record()], mod.FORMATS["yaml"], "metadata_only")
        assert "budgeted body text" not in pairs[0][1]
        assert "1946" in pairs[0][1]

    def test_pairs_the_question_with_the_passage(self):
        pairs = mod.build_pairs([record("7")], mod.FORMATS["yaml"], "with_body")
        assert pairs[0][0] == "question 7"

    def test_tolerates_a_record_without_a_truncated_body(self):
        rec = {k: v for k, v in record().items() if k != "truncated_body"}
        pairs = mod.build_pairs([rec], mod.FORMATS["yaml"], "with_body")
        assert pairs[0][1].endswith("---\n")


def write_split(root, name, records):
    """Write one split file under the corpus directory."""
    path = split_dir(root) / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"records": records}))
    return path


class TestRunMode:
    def _args(self, tmp_path, **over):
        base = {
            "split": "test",
            "out_dir": tmp_path,
            "in_dir": tmp_path,
            "batch_size": 4,
            "seed": 0,
            "no_trust_remote_code": False,
            "eager_attn": False,
            "out_tag": "",
        }
        base.update(over)
        return pytest.importorskip("argparse").Namespace(**base)

    def _patch_loaders(self, monkeypatch, model=None):
        monkeypatch.setattr(
            mod, "try_load_model", lambda *_a, **_k: model or LogitModel()
        )

    def test_writes_results_for_each_model(self, monkeypatch, tmp_path):
        self._patch_loaders(monkeypatch)
        records = [record(str(i)) for i in range(6)]
        toks = {"model/a": PairTokenizer(), "model/b": PairTokenizer()}
        mod.run_mode(
            "metadata_only",
            records,
            list(toks),
            toks,
            self._args(tmp_path),
            "cpu",
            None,
            None,
            ProgressCounter(10),
        )
        data = json.loads((tmp_path / "test_cross_metadata_only.json").read_text())
        assert data["models_probed"] == ["model/a", "model/b"]
        assert data["models_failed"] == []
        assert len(data["results"]["model/a"]["scores"]) == FORMAT_COUNT
        assert "summary" in data["results"]["model/a"]["stats"]

    def test_records_a_missing_tokenizer_as_a_failure(self, monkeypatch, tmp_path):
        self._patch_loaders(monkeypatch)
        mod.run_mode(
            "metadata_only",
            [record()],
            ["model/a"],
            {},
            self._args(tmp_path),
            "cpu",
            None,
            None,
            ProgressCounter(10),
        )
        data = json.loads((tmp_path / "test_cross_metadata_only.json").read_text())
        assert data["models_failed"][0]["stage"] == "tokenizer_load"

    def test_records_a_model_load_failure(self, monkeypatch, tmp_path):
        monkeypatch.setattr(mod, "try_load_model", lambda *_a, **_k: None)
        toks = {"model/a": PairTokenizer()}
        mod.run_mode(
            "metadata_only",
            [record()],
            list(toks),
            toks,
            self._args(tmp_path),
            "cpu",
            None,
            None,
            ProgressCounter(10),
        )
        data = json.loads((tmp_path / "test_cross_metadata_only.json").read_text())
        assert data["models_failed"][0]["stage"] == "model_load"

    def test_records_a_scoring_failure_without_stopping(self, monkeypatch, tmp_path):
        self._patch_loaders(monkeypatch)

        def boom(*_args, **_kwargs):
            raise RuntimeError("cuda oom")

        monkeypatch.setattr(mod, "score_batch", boom)
        toks = {"model/a": PairTokenizer(), "model/b": PairTokenizer()}
        mod.run_mode(
            "metadata_only",
            [record()],
            list(toks),
            toks,
            self._args(tmp_path),
            "cpu",
            None,
            None,
            ProgressCounter(10),
        )
        data = json.loads((tmp_path / "test_cross_metadata_only.json").read_text())
        assert len(data["models_failed"]) == 2
        assert data["models_failed"][0]["error"].startswith("RuntimeError")

    def test_with_body_mode_records_the_budget_metadata(self, monkeypatch, tmp_path):
        self._patch_loaders(monkeypatch)
        toks = {"model/a": PairTokenizer()}
        mod.run_mode(
            "with_body",
            [record()],
            list(toks),
            toks,
            self._args(tmp_path),
            "cpu",
            {"model/a": 1},
            {"dropped": 0, "budget_min": 400},
            ProgressCounter(10),
        )
        data = json.loads((tmp_path / "test_cross_with_body.json").read_text())
        assert data["tightest_tokeniser_counts"] == {"model/a": 1}
        assert data["drop_stats"]["dropped"] == 0

    def test_metadata_only_mode_records_the_budget_metadata(
        self, monkeypatch, tmp_path
    ):
        self._patch_loaders(monkeypatch)
        toks = {"model/a": PairTokenizer()}
        mod.run_mode(
            "metadata_only",
            [record("1"), record("2")],
            list(toks),
            toks,
            self._args(tmp_path),
            "cpu",
            {"model/a": 2},
            {"dropped": 3, "budget_min": 400},
            ProgressCounter(10),
        )
        data = json.loads((tmp_path / "test_cross_metadata_only.json").read_text())
        assert data["tightest_tokeniser_counts"] == {"model/a": 2}
        assert data["drop_stats"]["dropped"] == 3

    def test_records_the_identifier_of_every_record_scored(self, monkeypatch, tmp_path):
        self._patch_loaders(monkeypatch)
        toks = {"model/a": PairTokenizer()}
        mod.run_mode(
            "with_body",
            [record("7"), record("8")],
            list(toks),
            toks,
            self._args(tmp_path),
            "cpu",
            {"model/a": 2},
            {"dropped": 0, "budget_min": 400},
            ProgressCounter(10),
        )
        data = json.loads((tmp_path / "test_cross_with_body.json").read_text())
        assert data["record_ids"] == ["7", "8"]
        assert data["n_records"] == len(data["record_ids"])

    def test_releases_gpu_memory_after_each_model(self, monkeypatch, tmp_path):
        calls = []
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append(1))
        self._patch_loaders(monkeypatch)
        toks = {"model/a": PairTokenizer(), "model/b": PairTokenizer()}
        mod.run_mode(
            "metadata_only",
            [record("1"), record("2")],
            list(toks),
            toks,
            self._args(tmp_path),
            "cuda",
            None,
            None,
            ProgressCounter(10),
        )
        assert len(calls) == len(toks)

    def test_does_not_touch_gpu_memory_on_a_cpu_run(self, monkeypatch, tmp_path):
        calls = []
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append(1))
        self._patch_loaders(monkeypatch)
        toks = {"model/a": PairTokenizer()}
        mod.run_mode(
            "metadata_only",
            [record("1"), record("2")],
            list(toks),
            toks,
            self._args(tmp_path),
            "cpu",
            None,
            None,
            ProgressCounter(10),
        )
        assert calls == []

    def test_truncates_a_long_model_name_in_the_summary(
        self, monkeypatch, tmp_path, capsys
    ):
        self._patch_loaders(monkeypatch)
        long_name = "vendor/" + "x" * 60
        toks = {long_name: PairTokenizer()}
        mod.run_mode(
            "metadata_only",
            [record("1"), record("2")],
            list(toks),
            toks,
            self._args(tmp_path),
            "cpu",
            None,
            None,
            ProgressCounter(10),
        )
        out = capsys.readouterr().out
        assert shorten(long_name) in out
        assert out.count(long_name) == 1


class TestBuildParser:
    def test_defaults_to_both_modes_and_the_full_roster(self):
        args = mod.build_parser().parse_args([])
        assert args.mode == "both"
        assert args.models == list(BASE_MODEL_IDS)
        assert args.split == "test"

    def test_accepts_an_explicit_model_list(self):
        args = mod.build_parser().parse_args(["--models", "a/b", "c/d"])
        assert args.models == ["a/b", "c/d"]

    def test_rejects_an_unknown_split(self):
        with pytest.raises(SystemExit):
            mod.build_parser().parse_args(["--split", "validation"])

    def test_accepts_every_split_and_all(self):
        for split in ("train", "dev", "test", "nq_val", "all"):
            assert mod.build_parser().parse_args(["--split", split]).split == split


class TestMain:
    def _corpus(self, tmp_path, n=3):
        records = [
            {
                "id": str(i),
                "question": f"q{i}",
                "pairs": [("Born", "1946")],
                "body": "body prose " * 30,
                "quality_flags": [],
            }
            for i in range(n)
        ]
        write_split(tmp_path, "test", records)
        return tmp_path

    def _patch(self, monkeypatch, _tmp_path):
        monkeypatch.setattr(
            mod, "try_load_tokenizer", lambda *_a, **_k: PairTokenizer()
        )
        monkeypatch.setattr(mod, "try_load_model", lambda *_a, **_k: LogitModel())
        # Scores must vary, or every rank correlation is undefined.
        monkeypatch.setattr(
            mod, "score_batch", lambda *a, **_k: [i * 0.1 for i in range(len(a[2]))]
        )
        monkeypatch.setattr(
            mod,
            "prepare_records_with_body",
            lambda recs, _toks: (
                [{**r, "truncated_body": "b", "body_budget_tokens": 400} for r in recs],
                {"dropped": 0, "tightest_counts": {"m/a": len(recs)}},
            ),
        )

    def test_runs_metadata_only_and_writes_output(self, monkeypatch, tmp_path):
        self._corpus(tmp_path)
        self._patch(monkeypatch, tmp_path)
        monkeypatch.setattr(
            "sys.argv",
            [
                "prog",
                "--mode",
                "metadata_only",
                "--models",
                "m/a",
                "--in-dir",
                str(tmp_path),
                "--out-dir",
                str(tmp_path),
            ],
        )
        mod.main()
        data = json.loads((tmp_path / "test_cross_metadata_only.json").read_text())
        assert data["n_records"] == 3

    def test_both_modes_score_one_record_set(self, monkeypatch, tmp_path):
        self._corpus(tmp_path, n=5)
        self._patch(monkeypatch, tmp_path)
        # The budget keeps three of the five records.
        monkeypatch.setattr(
            mod,
            "prepare_records_with_body",
            lambda recs, _toks: (
                [
                    {**r, "truncated_body": "b", "body_budget_tokens": 400}
                    for r in recs[:3]
                ],
                {"dropped": 2, "tightest_counts": {"m/a": 3}},
            ),
        )
        monkeypatch.setattr(
            "sys.argv",
            [
                "prog",
                "--mode",
                "both",
                "--models",
                "m/a",
                "--in-dir",
                str(tmp_path),
                "--out-dir",
                str(tmp_path),
            ],
        )
        mod.main()
        payloads = {
            mode: json.loads((tmp_path / f"test_cross_{mode}.json").read_text())
            for mode in ("with_body", "metadata_only")
        }
        assert payloads["with_body"]["record_ids"] == ["0", "1", "2"]
        assert (
            payloads["with_body"]["record_ids"]
            == payloads["metadata_only"]["record_ids"]
        )
        assert {p["n_records"] for p in payloads.values()} == {3}
        assert {p["drop_stats"]["dropped"] for p in payloads.values()} == {2}

    def test_keeps_the_progress_of_the_whole_job(self, monkeypatch, tmp_path):
        self._corpus(tmp_path)
        self._patch(monkeypatch, tmp_path)
        progress = tmp_path / "job.progress"
        monkeypatch.setattr(
            "sys.argv",
            [
                "prog",
                "--mode",
                "both",
                "--models",
                "m/a",
                "--in-dir",
                str(tmp_path),
                "--out-dir",
                str(tmp_path),
                "--progress-file",
                str(progress),
            ],
        )
        mod.main()
        fields = parse_progress(progress.read_text())
        assert (fields["done"], fields["total"]) == ("10", "10")
        assert fields["label"] == "with_body/markdown"
        assert fields["failed"] == "0"

    def test_keeps_no_progress_file_unless_asked(self, monkeypatch, tmp_path):
        self._corpus(tmp_path)
        self._patch(monkeypatch, tmp_path)
        monkeypatch.setattr(
            "sys.argv",
            [
                "prog",
                "--mode",
                "with_body",
                "--models",
                "m/a",
                "--in-dir",
                str(tmp_path),
                "--out-dir",
                str(tmp_path),
            ],
        )
        mod.main()
        assert list(tmp_path.glob("*.progress")) == []

    def test_keeps_the_progress_out_of_the_log(self, monkeypatch, tmp_path, capsys):
        self._corpus(tmp_path)
        self._patch(monkeypatch, tmp_path)
        monkeypatch.setattr(
            "sys.argv",
            [
                "prog",
                "--mode",
                "with_body",
                "--models",
                "m/a",
                "--in-dir",
                str(tmp_path),
                "--out-dir",
                str(tmp_path),
                "--progress-file",
                str(tmp_path / "job.progress"),
            ],
        )
        mod.main()
        out = capsys.readouterr().out
        assert "done=" not in out
        assert "total=" not in out

    def test_reports_the_total_time_of_the_pass(self, monkeypatch, tmp_path, capsys):
        self._corpus(tmp_path)
        self._patch(monkeypatch, tmp_path)
        monkeypatch.setattr(
            "sys.argv",
            [
                "prog",
                "--mode",
                "with_body",
                "--models",
                "m/a",
                "--in-dir",
                str(tmp_path),
                "--out-dir",
                str(tmp_path),
            ],
        )
        mod.main()
        assert "Pass complete in 0:00:" in capsys.readouterr().out

    def test_with_body_mode_reports_the_budget_distribution(
        self, monkeypatch, tmp_path, capsys
    ):
        self._corpus(tmp_path)
        self._patch(monkeypatch, tmp_path)
        monkeypatch.setattr(
            "sys.argv",
            [
                "prog",
                "--mode",
                "with_body",
                "--models",
                "m/a",
                "--in-dir",
                str(tmp_path),
                "--out-dir",
                str(tmp_path),
            ],
        )
        mod.main()
        out = capsys.readouterr().out
        assert "body budget tokens" in out
        assert "tightest tokenizer distribution" in out

    def test_the_limit_caps_the_record_count(self, monkeypatch, tmp_path):
        self._corpus(tmp_path, n=8)
        self._patch(monkeypatch, tmp_path)
        monkeypatch.setattr(
            "sys.argv",
            [
                "prog",
                "--mode",
                "metadata_only",
                "--models",
                "m/a",
                "--limit",
                "2",
                "--in-dir",
                str(tmp_path),
                "--out-dir",
                str(tmp_path),
            ],
        )
        mod.main()
        data = json.loads((tmp_path / "test_cross_metadata_only.json").read_text())
        assert data["n_records"] == 2

    def test_smoke_test_mode_caps_records_and_batch(
        self, monkeypatch, tmp_path, capsys
    ):
        self._corpus(tmp_path, n=40)
        self._patch(monkeypatch, tmp_path)
        monkeypatch.setattr(
            "sys.argv",
            [
                "prog",
                "--mode",
                "metadata_only",
                "--models",
                "m/a",
                "--smoke-test",
                "--in-dir",
                str(tmp_path),
                "--out-dir",
                str(tmp_path),
            ],
        )
        mod.main()
        assert "SMOKE TEST MODE" in capsys.readouterr().out
        data = json.loads((tmp_path / "test_cross_metadata_only.json").read_text())
        assert data["n_records"] == mod.SMOKE_TEST_RECORDS

    def test_exits_when_no_tokenizer_loads(self, monkeypatch, tmp_path):
        self._corpus(tmp_path)
        self._patch(monkeypatch, tmp_path)
        monkeypatch.setattr(mod, "try_load_tokenizer", lambda *_a, **_k: None)
        monkeypatch.setattr(
            "sys.argv",
            [
                "prog",
                "--models",
                "m/a",
                "--in-dir",
                str(tmp_path),
                "--out-dir",
                str(tmp_path),
            ],
        )
        with pytest.raises(SystemExit):
            mod.main()

    def test_skips_a_mode_with_no_eligible_records(self, monkeypatch, tmp_path, capsys):
        write_split(tmp_path, "test", [])
        self._patch(monkeypatch, tmp_path)
        monkeypatch.setattr(
            "sys.argv",
            [
                "prog",
                "--mode",
                "metadata_only",
                "--models",
                "m/a",
                "--in-dir",
                str(tmp_path),
                "--out-dir",
                str(tmp_path),
            ],
        )
        mod.main()
        assert "No eligible records" in capsys.readouterr().out


class TestOutTag:
    def test_names_the_file_without_a_tag(self, monkeypatch, tmp_path):
        TestRunMode()._patch_loaders(monkeypatch)
        mod.run_mode(
            "metadata_only",
            [record("1")],
            ["model/a"],
            {"model/a": PairTokenizer()},
            TestRunMode()._args(tmp_path),
            "cpu",
            None,
            None,
            ProgressCounter(10),
        )
        assert (tmp_path / "test_cross_metadata_only.json").exists()

    def test_a_tag_separates_parallel_runs(self, monkeypatch, tmp_path):
        TestRunMode()._patch_loaders(monkeypatch)
        mod.run_mode(
            "metadata_only",
            [record("1")],
            ["model/a"],
            {"model/a": PairTokenizer()},
            TestRunMode()._args(tmp_path, out_tag="bge_base"),
            "cpu",
            None,
            None,
            ProgressCounter(10),
        )
        assert (tmp_path / "test_cross_metadata_only_bge_base.json").exists()


class TestResolveModels:
    def test_passes_an_identifier_through(self):
        assert mod.resolve_models(["BAAI/bge-reranker-base"]) == [
            "BAAI/bge-reranker-base"
        ]

    def test_resolves_a_slug(self):
        assert mod.resolve_models(["bge_base"]) == ["BAAI/bge-reranker-base"]

    def test_rejects_an_unknown_slug(self):
        with pytest.raises(ValueError, match="unknown model"):
            mod.resolve_models(["nope"])


class TestListModels:
    def test_prints_every_slug_and_stops(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(
            "sys.argv", ["prog", "--out-dir", str(tmp_path), "--list-models"]
        )
        mod.main()
        assert capsys.readouterr().out.split() == [m.slug for m in BASE_MODELS]


class TestBudgetModels:
    def test_defaults_to_the_whole_roster(self):
        args = mod.build_parser().parse_args([])
        assert args.budget_models == list(BASE_MODEL_IDS)

    def test_one_scored_model_still_budgets_across_the_roster(
        self, monkeypatch, tmp_path
    ):
        seen = {}

        def capture(records, tokenizers):
            seen["budget"] = sorted(tokenizers)
            return records, {"tightest_counts": {}, "dropped": 0}

        monkeypatch.setattr(mod, "prepare_records_with_body", capture)
        monkeypatch.setattr(
            mod, "try_load_tokenizer", lambda _name, _t: PairTokenizer()
        )
        monkeypatch.setattr(mod, "try_load_model", lambda *_a, **_k: LogitModel())
        write_split(tmp_path, "test", [record(str(i)) for i in range(4)])
        monkeypatch.setattr(
            "sys.argv",
            [
                "prog",
                "--in-dir",
                str(tmp_path),
                "--out-dir",
                str(tmp_path),
                "--mode",
                "with_body",
                "--models",
                "bge_base",
            ],
        )
        mod.main()
        assert seen["budget"] == sorted(BASE_MODEL_IDS)

    def test_stops_when_a_budget_tokenizer_is_missing(self, monkeypatch, tmp_path):
        def load(name, _trust):
            return None if "bge-reranker-v2-m3" in name else PairTokenizer()

        monkeypatch.setattr(mod, "try_load_tokenizer", load)
        write_split(tmp_path, "test", [record("1")])
        monkeypatch.setattr(
            "sys.argv",
            ["prog", "--in-dir", str(tmp_path), "--out-dir", str(tmp_path)],
        )
        with pytest.raises(SystemExit, match="budget tokenizer"):
            mod.main()
