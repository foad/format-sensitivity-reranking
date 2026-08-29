"""Tests for scripts.h1.nq_h1_measurement."""

from __future__ import annotations

import json

import pytest
import torch
from scripts.h1 import nq_h1_measurement as mod
from tests.fakes import LogitModel, PairTokenizer

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
        def fail(*args, **kwargs):
            raise OSError("no such model")

        monkeypatch.setattr(mod.AutoTokenizer, "from_pretrained", fail)
        assert mod.try_load_tokenizer("missing/model", True) is None
        assert "tokenizer load failed" in capsys.readouterr().out

    def test_returns_the_tokenizer_on_success(self, monkeypatch):
        monkeypatch.setattr(
            mod.AutoTokenizer, "from_pretrained", lambda *a, **k: "TOKENIZER"
        )
        assert mod.try_load_tokenizer("some/model", True) == "TOKENIZER"

    def test_forwards_the_trust_flag(self, monkeypatch):
        seen = {}

        def capture(name, **kwargs):
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
            lambda *a, **k: self.Loaded(),
        )
        model = mod.try_load_model("some/model", "cpu", True)
        assert model.device == "cpu"
        assert model.evaluated is True

    def test_requests_eager_attention_when_asked(self, monkeypatch):
        seen = {}

        def capture(name, **kwargs):
            seen.update(kwargs)
            return self.Loaded()

        monkeypatch.setattr(
            mod.AutoModelForSequenceClassification, "from_pretrained", capture
        )
        mod.try_load_model("some/model", "cpu", True, force_eager_attn=True)
        assert seen["attn_implementation"] == "eager"

    def test_retries_without_eager_attention_on_a_type_error(self, monkeypatch):
        calls = []

        def capture(name, **kwargs):
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
        def capture(name, **kwargs):
            if "attn_implementation" in kwargs:
                raise TypeError("unexpected keyword")
            raise OSError("corrupt weights")

        monkeypatch.setattr(
            mod.AutoModelForSequenceClassification, "from_pretrained", capture
        )
        assert mod.try_load_model("m", "cpu", True, force_eager_attn=True) is None
        assert "model load failed" in capsys.readouterr().out

    def test_returns_none_on_a_non_type_error(self, monkeypatch, capsys):
        def fail(*args, **kwargs):
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


class TestRunMode:
    def _args(self, tmp_path, **over):
        base = {
            "split": "train",
            "out_dir": tmp_path,
            "in_dir": tmp_path,
            "batch_size": 4,
            "seed": 0,
            "no_trust_remote_code": False,
            "eager_attn": False,
        }
        base.update(over)
        return pytest.importorskip("argparse").Namespace(**base)

    def _patch_loaders(self, monkeypatch, model=None):
        monkeypatch.setattr(
            mod, "try_load_model", lambda *a, **k: model or LogitModel()
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
        )
        data = json.loads((tmp_path / "train_metadata_only.json").read_text())
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
        )
        data = json.loads((tmp_path / "train_metadata_only.json").read_text())
        assert data["models_failed"][0]["stage"] == "tokenizer_load"

    def test_records_a_model_load_failure(self, monkeypatch, tmp_path):
        monkeypatch.setattr(mod, "try_load_model", lambda *a, **k: None)
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
        )
        data = json.loads((tmp_path / "train_metadata_only.json").read_text())
        assert data["models_failed"][0]["stage"] == "model_load"

    def test_records_a_scoring_failure_without_stopping(self, monkeypatch, tmp_path):
        self._patch_loaders(monkeypatch)

        def boom(*args, **kwargs):
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
        )
        data = json.loads((tmp_path / "train_metadata_only.json").read_text())
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
        )
        data = json.loads((tmp_path / "train_with_body.json").read_text())
        assert data["tightest_tokeniser_counts"] == {"model/a": 1}
        assert data["drop_stats"]["dropped"] == 0

    def test_metadata_only_mode_omits_the_budget_metadata(self, monkeypatch, tmp_path):
        self._patch_loaders(monkeypatch)
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
        )
        data = json.loads((tmp_path / "train_metadata_only.json").read_text())
        assert "tightest_tokeniser_counts" not in data
        assert "drop_stats" not in data

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
        )
        assert "..." in capsys.readouterr().out


class TestBuildParser:
    def test_defaults_to_both_modes_and_the_full_roster(self):
        args = mod.build_parser().parse_args([])
        assert args.mode == "both"
        assert args.models == mod.DEFAULT_MODELS
        assert args.split == "train"

    def test_accepts_an_explicit_model_list(self):
        args = mod.build_parser().parse_args(["--models", "a/b", "c/d"])
        assert args.models == ["a/b", "c/d"]

    def test_rejects_an_unknown_split(self):
        with pytest.raises(SystemExit):
            mod.build_parser().parse_args(["--split", "test"])


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
        records.append({**records[0], "id": "bad", "quality_flags": ["too_few_pairs"]})
        (tmp_path / "parsed_train.json").write_text(json.dumps({"records": records}))
        return tmp_path

    def _patch(self, monkeypatch, tmp_path):
        monkeypatch.setattr(mod, "try_load_tokenizer", lambda *a, **k: PairTokenizer())
        monkeypatch.setattr(mod, "try_load_model", lambda *a, **k: LogitModel())
        # Scores must vary, or every rank correlation is undefined.
        monkeypatch.setattr(
            mod, "score_batch", lambda *a, **k: [i * 0.1 for i in range(len(a[2]))]
        )
        monkeypatch.setattr(
            mod,
            "prepare_records_with_body",
            lambda recs, toks: (
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
        data = json.loads((tmp_path / "train_metadata_only.json").read_text())
        assert data["n_records"] == 3

    def test_excludes_quality_flagged_records(self, monkeypatch, tmp_path):
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
        data = json.loads((tmp_path / "train_metadata_only.json").read_text())
        assert "bad" not in json.dumps(data["results"])

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
        data = json.loads((tmp_path / "train_metadata_only.json").read_text())
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
        data = json.loads((tmp_path / "train_metadata_only.json").read_text())
        assert data["n_records"] == mod.SMOKE_TEST_RECORDS

    def test_exits_when_no_tokenizer_loads(self, monkeypatch, tmp_path):
        self._corpus(tmp_path)
        self._patch(monkeypatch, tmp_path)
        monkeypatch.setattr(mod, "try_load_tokenizer", lambda *a, **k: None)
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
        (tmp_path / "parsed_train.json").write_text(json.dumps({"records": []}))
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
