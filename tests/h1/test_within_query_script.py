"""Tests for scripts.h1.within_query."""

from __future__ import annotations

import json

import numpy as np
import pytest
from scripts.h1 import within_query as mod
from tests.fakes import LogitModel, PairTokenizer, WordTokenizer

from fsr.corpus.layout import split_dir
from fsr.formats import FORMAT_NAMES
from fsr.models.registry import BASE_MODEL_IDS, BASE_MODELS

SENTENCE = "The bridge opened in 1946 and carries the road across the river. "
TOKENIZERS = {"word": WordTokenizer()}


def record(rec_id, flagged=False):
    return {
        "id": str(rec_id),
        "title": f"Article {rec_id}",
        "question": f"who built bridge {rec_id}",
        "pairs": [["Born", "1946"], ["Role", "Engineer"]],
        "body": SENTENCE * 20,
        "quality_flags": ["too_few_pairs"] if flagged else [],
    }


def corpus_tree(tmp_path, n=12, n_queries=3, n_negatives=3):
    """Write the parsed corpus and the split files, and return the data root."""
    records = [record(i) for i in range(n)]
    (tmp_path / "parsed_train.json").write_text(
        json.dumps({"records": [*records, record(900, flagged=True)]})
    )
    out_dir = split_dir(tmp_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train", "dev", "test"):
        (out_dir / f"{name}.json").write_text(
            json.dumps({"records": records[:n_queries]})
        )
    (out_dir / "nq_val.json").write_text(json.dumps({"records": [record(500)]}))
    (out_dir / "bm25_negatives.json").write_text(
        json.dumps(
            {
                "negatives": {
                    str(q): [str(i) for i in range(n) if i != q][:n_negatives]
                    for q in range(n_queries)
                }
            }
        )
    )
    return tmp_path


def matrices(n_queries=4, n_candidates=3, seed=0):
    rng = np.random.default_rng(seed)
    return {f: rng.normal(size=(n_queries, n_candidates)) for f in FORMAT_NAMES}


def prepared_queries(n_queries=2, n_candidates=3):
    return [
        {
            "id": str(q),
            "question": "who built it",
            "candidates": [
                {
                    "id": f"{q}-{c}",
                    "pairs": [["Born", "1946"]],
                    "truncated_body": "body",
                }
                for c in range(n_candidates)
            ],
        }
        for q in range(n_queries)
    ]


class TestLoaders:
    def test_loads_a_split(self, tmp_path):
        root = corpus_tree(tmp_path)
        assert len(mod.load_split(split_dir(root), "test")) == 3

    def test_loads_the_negatives(self, tmp_path):
        root = corpus_tree(tmp_path)
        negatives = mod.load_negatives(split_dir(root))
        assert len(negatives["0"]) == 3

    def test_the_corpus_index_drops_flagged_records(self, tmp_path):
        root = corpus_tree(tmp_path)
        assert "900" not in mod.load_corpus_index(root, split_dir(root))

    def test_the_corpus_index_adds_the_validation_split(self, tmp_path):
        root = corpus_tree(tmp_path)
        assert "500" in mod.load_corpus_index(root, split_dir(root))


class TestScoreAllFormats:
    def test_returns_one_matrix_per_format(self):
        out = mod.score_all_formats(
            LogitModel(), PairTokenizer(), prepared_queries(), 4, "cpu", verbose=False
        )
        assert set(out) == set(FORMAT_NAMES)

    def test_each_matrix_is_queries_by_candidates(self):
        out = mod.score_all_formats(
            LogitModel(),
            PairTokenizer(),
            prepared_queries(3, 4),
            4,
            "cpu",
            verbose=False,
        )
        assert all(m.shape == (3, 4) for m in out.values())

    def test_reports_each_format_when_asked(self, capsys):
        mod.score_all_formats(
            LogitModel(), PairTokenizer(), prepared_queries(), 4, "cpu"
        )
        printed = capsys.readouterr().out
        assert all(f in printed for f in FORMAT_NAMES)

    def test_stays_quiet_otherwise(self, capsys):
        mod.score_all_formats(
            LogitModel(), PairTokenizer(), prepared_queries(), 4, "cpu", verbose=False
        )
        assert capsys.readouterr().out == ""


class TestSummariseModel:
    def test_reports_every_statistic(self):
        entry = mod.summarise_model(matrices())
        for key in (
            "within_query",
            "per_query",
            "mrr_per_format",
            "reciprocal_ranks",
            "gold_top1",
            "conditional_inconsistency",
            "gold_scores_per_fmt",
            "scale_diagnostic",
        ):
            assert key in entry

    def test_keeps_the_gold_column(self):
        source = matrices(n_queries=4)
        entry = mod.summarise_model(source)
        assert entry["gold_scores_per_fmt"]["yaml"] == source["yaml"][:, 0].tolist()

    def test_omits_the_matrices_by_default(self):
        assert "matrices" not in mod.summarise_model(matrices())

    def test_keeps_the_matrices_when_asked(self):
        entry = mod.summarise_model(matrices(), store_matrices=True)
        assert set(entry["matrices"]) == set(FORMAT_NAMES)

    def test_reports_one_mrr_per_format(self):
        assert set(mod.summarise_model(matrices())["mrr_per_format"]) == set(
            FORMAT_NAMES
        )


class TestReporting:
    def test_reports_one_model(self, capsys):
        mod.report_model(mod.summarise_model(matrices()))
        printed = capsys.readouterr().out
        assert "max within-query flip %" in printed
        assert "MRR per format" in printed

    def test_reports_the_cross_model_table(self, capsys):
        entry = mod.summarise_model(matrices())
        mod.report_summary({"model/a": entry}, [], 3, "test", 4)
        printed = capsys.readouterr().out
        assert "WITHIN-QUERY SUMMARY" in printed
        assert "model/a" in printed

    def test_shortens_a_long_model_name(self, capsys):
        entry = mod.summarise_model(matrices())
        mod.report_summary({"x" * 60: entry}, [], 3, "test", 4)
        assert "…" in capsys.readouterr().out

    def test_reports_failures(self, capsys):
        failures = [{"model": "model/b", "stage": "scoring", "error": "boom"}]
        mod.report_summary({}, failures, 3, "test", 4)
        printed = capsys.readouterr().out
        assert "FAILURES (1)" in printed
        assert "boom" in printed


class TestLoadTokenizers:
    def test_loads_each_tokenizer(self, monkeypatch):
        monkeypatch.setattr(mod, "load_tokenizer", lambda name: f"tok:{name}")
        assert mod.load_tokenizers(["a", "b"]) == {"a": "tok:a", "b": "tok:b"}

    def test_stops_when_one_fails(self, monkeypatch):
        def load(name):
            if name == "b":
                raise RuntimeError("no such model")
            return "tok"

        monkeypatch.setattr(mod, "load_tokenizer", load)
        with pytest.raises(SystemExit, match="Not every tokenizer loaded"):
            mod.load_tokenizers(["a", "b"])


class TestPrepareQueries:
    def _args(self, tmp_path, **overrides):
        args = mod.build_parser().parse_args(["--data-root", str(tmp_path)])
        args.out_dir = tmp_path / mod.OUT_SUBDIR
        args.out_dir.mkdir(parents=True, exist_ok=True)
        args.negatives = 3
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    def test_builds_and_caches(self, tmp_path):
        root = corpus_tree(tmp_path)
        args = self._args(root)
        prepared, stats = mod.prepare_queries(args, split_dir(root), TOKENIZERS)
        assert stats["n_kept"] == len(prepared) == 3
        assert (args.out_dir / "test_candidates.json").exists()

    def test_reuses_the_cache(self, tmp_path, capsys):
        root = corpus_tree(tmp_path)
        args = self._args(root)
        mod.prepare_queries(args, split_dir(root), TOKENIZERS)
        capsys.readouterr()
        mod.prepare_queries(args, split_dir(root), TOKENIZERS)
        assert "Reusing candidate lists" in capsys.readouterr().out

    def test_force_rebuilds_the_cache(self, tmp_path, capsys):
        root = corpus_tree(tmp_path)
        args = self._args(root)
        mod.prepare_queries(args, split_dir(root), TOKENIZERS)
        capsys.readouterr()
        args.force = True
        mod.prepare_queries(args, split_dir(root), TOKENIZERS)
        assert "Building candidate lists" in capsys.readouterr().out

    def test_the_limit_names_the_cache(self, tmp_path):
        root = corpus_tree(tmp_path)
        args = self._args(root, limit=2)
        mod.prepare_queries(args, split_dir(root), TOKENIZERS)
        assert (args.out_dir / "test_candidates_limit2.json").exists()

    def test_the_limit_caps_the_queries(self, tmp_path):
        root = corpus_tree(tmp_path)
        prepared, _ = mod.prepare_queries(
            self._args(root, limit=2), split_dir(root), TOKENIZERS
        )
        assert len(prepared) == 2

    def test_stops_when_nothing_survives(self, tmp_path):
        root = corpus_tree(tmp_path)
        args = self._args(root, negatives=50)
        with pytest.raises(SystemExit, match="No queries survived"):
            mod.prepare_queries(args, split_dir(root), TOKENIZERS)


class TestBuildParser:
    def test_defaults(self):
        args = mod.build_parser().parse_args([])
        assert args.split == "test"
        assert args.models == list(BASE_MODEL_IDS)
        assert args.budget_models == list(BASE_MODEL_IDS)
        assert args.negatives == mod.DEFAULT_NEGATIVES
        assert args.overflow_sample == 100
        assert args.limit == 0
        assert args.force is False
        assert args.store_matrices is False
        assert args.out_dir is None

    def test_parses_an_adapter_run(self):
        args = mod.build_parser().parse_args(
            ["--models", "m", "--lora-adapter", "a", "--tanh-head"]
        )
        assert args.models == ["m"]
        assert args.lora_adapter == "a"
        assert args.tanh_head is True


class DualTokenizer(WordTokenizer):
    """A tokenizer that budgets single texts and encodes batches for scoring."""

    def __call__(self, text, text_pair=None, **kwargs):
        """Return word counts for one text, or a tensor batch for a list."""
        if isinstance(text, list):
            return PairTokenizer()(text, text_pair, **kwargs)
        return super().__call__(text, text_pair, **kwargs)


def run_main(monkeypatch, tmp_path, *extra, model=None):
    """Run main over a small corpus with the model and tokenizers faked."""
    root = corpus_tree(tmp_path)
    monkeypatch.setattr(mod, "load_tokenizer", lambda _name: DualTokenizer())
    monkeypatch.setattr(mod, "load_model", lambda *_a, **_k: model or LogitModel())
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--data-root",
            str(root),
            "--models",
            "model/a",
            "--budget-models",
            "model/a",
            "--negatives",
            "3",
            "--overflow-sample",
            "2",
            *extra,
        ],
    )
    mod.main()
    return root / mod.OUT_SUBDIR / "test_within_query.json"


class TestMain:
    def test_writes_the_measurement(self, monkeypatch, tmp_path):
        payload = json.loads(run_main(monkeypatch, tmp_path).read_text())
        assert payload["split"] == "test"
        assert payload["models_probed"] == ["model/a"]
        assert payload["models_failed"] == []
        assert payload["n_queries"] == 3
        assert payload["query_ids"] == ["0", "1", "2"]
        assert payload["formats"] == list(FORMAT_NAMES)
        assert payload["neg_count_per_query"] == 3

    def test_records_the_preparation_statistics(self, monkeypatch, tmp_path):
        payload = json.loads(run_main(monkeypatch, tmp_path).read_text())
        assert payload["prep_stats"]["n_kept"] == 3
        assert payload["prep_stats"]["overflow"]["queries_sampled"] == 2

    def test_disables_the_overflow_check(self, monkeypatch, tmp_path):
        out = run_main(monkeypatch, tmp_path, "--overflow-sample", "0")
        assert json.loads(out.read_text())["prep_stats"]["overflow"] is None

    def test_keeps_the_matrices_when_asked(self, monkeypatch, tmp_path):
        out = run_main(monkeypatch, tmp_path, "--store-matrices")
        entry = json.loads(out.read_text())["results"]["model/a"]
        assert set(entry["matrices"]) == set(FORMAT_NAMES)

    def test_records_a_model_that_fails_to_load(self, monkeypatch, tmp_path):
        root = corpus_tree(tmp_path)
        monkeypatch.setattr(mod, "load_tokenizer", lambda _name: DualTokenizer())

        def fail(*_args, **_kwargs):
            raise RuntimeError("no weights")

        monkeypatch.setattr(mod, "load_model", fail)
        monkeypatch.setattr(
            "sys.argv",
            [
                "prog",
                "--data-root",
                str(root),
                "--models",
                "model/a",
                "--budget-models",
                "model/a",
                "--negatives",
                "3",
            ],
        )
        mod.main()
        payload = json.loads(
            (root / mod.OUT_SUBDIR / "test_within_query.json").read_text()
        )
        assert payload["models_failed"] == [
            {"model": "model/a", "stage": "model_load", "error": "no weights"}
        ]

    def test_records_a_model_that_fails_to_score(self, monkeypatch, tmp_path):
        def fail(*_args, **_kwargs):
            raise RuntimeError("bad shape")

        monkeypatch.setattr(mod, "score_all_formats", fail)
        out = run_main(monkeypatch, tmp_path)
        payload = json.loads(out.read_text())
        assert payload["models_failed"][0]["stage"] == "scoring"
        assert payload["results"] == {}

    def test_rejects_an_adapter_for_several_models(self, monkeypatch, tmp_path):
        root = corpus_tree(tmp_path)
        monkeypatch.setattr(
            "sys.argv",
            ["prog", "--data-root", str(root), "--lora-adapter", "a"],
        )
        with pytest.raises(SystemExit, match="one --models entry"):
            mod.main()

    def test_reports_the_adapter_and_the_head(self, monkeypatch, tmp_path, capsys):
        run_main(monkeypatch, tmp_path, "--lora-adapter", "a", "--tanh-head")
        printed = capsys.readouterr().out
        assert "adapter: a" in printed
        assert "dense->tanh->out_proj" in printed

    def test_honours_a_custom_output_directory(self, monkeypatch, tmp_path):
        out_dir = tmp_path / "elsewhere"
        run_main(monkeypatch, tmp_path, "--out-dir", str(out_dir))
        assert (out_dir / "test_within_query.json").exists()

    def test_honours_the_output_tag(self, monkeypatch, tmp_path):
        root = tmp_path
        run_main(monkeypatch, root, "--out-tag", "baseline")
        assert (root / mod.OUT_SUBDIR / "test_baseline.json").exists()

    def test_frees_the_cache_on_a_gpu(self, monkeypatch, tmp_path):
        freed = []
        monkeypatch.setattr(mod.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(mod.torch.cuda, "empty_cache", lambda: freed.append(1))
        run_main(monkeypatch, tmp_path)
        assert freed == [1]

    def test_leaves_the_cache_alone_on_a_cpu(self, monkeypatch, tmp_path):
        freed = []
        monkeypatch.setattr(mod.torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(mod.torch.cuda, "empty_cache", lambda: freed.append(1))
        run_main(monkeypatch, tmp_path)
        assert freed == []


class TestResolveModels:
    def test_passes_an_identifier_through(self):
        assert mod.resolve_models(["BAAI/bge-reranker-base"]) == [
            "BAAI/bge-reranker-base"
        ]

    def test_resolves_a_slug(self):
        assert mod.resolve_models(["bge_base"]) == ["BAAI/bge-reranker-base"]

    def test_keeps_the_order(self):
        assert mod.resolve_models(["jina_v2", "minilm_l6"]) == [
            "jinaai/jina-reranker-v2-base-multilingual",
            "cross-encoder/ms-marco-MiniLM-L6-v2",
        ]

    def test_rejects_an_unknown_slug(self):
        with pytest.raises(ValueError, match="unknown model"):
            mod.resolve_models(["nope"])


class TestListModels:
    def test_prints_every_slug_and_stops(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(
            "sys.argv", ["prog", "--data-root", str(tmp_path), "--list-models"]
        )
        mod.main()
        printed = capsys.readouterr().out.split()
        assert printed == [m.slug for m in BASE_MODELS]
        assert not (tmp_path / mod.OUT_SUBDIR).exists()


class TestPrepareOnly:
    def test_caches_the_candidates_and_stops(self, monkeypatch, tmp_path, capsys):
        root = corpus_tree(tmp_path)
        monkeypatch.setattr(mod, "load_tokenizer", lambda _name: DualTokenizer())
        monkeypatch.setattr(
            mod, "load_model", lambda *_a, **_k: pytest.fail("must not score")
        )
        monkeypatch.setattr(
            "sys.argv",
            [
                "prog",
                "--data-root",
                str(root),
                "--models",
                "bge_base",
                "--budget-models",
                "bge_base",
                "--negatives",
                "3",
                "--prepare-only",
            ],
        )
        mod.main()
        assert (root / mod.OUT_SUBDIR / "test_candidates.json").exists()
        assert not (root / mod.OUT_SUBDIR / "test_within_query.json").exists()
        assert "stopping before scoring" in capsys.readouterr().out
