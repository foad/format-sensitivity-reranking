from __future__ import annotations

import json

import pytest

from fsr.corpus.layout import NEGATIVES_NAME, split_dir
from fsr.training.data import (
    DEFAULT_TRAIN_NEGATIVES,
    load_corpus_index,
    load_negatives,
    prepare_dev_records,
    prepare_train_records,
    truncate_to_budget,
)
from tests.fakes import CharTokenizer, WordTokenizer

TOKENIZERS = {"word": WordTokenizer()}


def record(rid, *, question="who wrote it", body=None, pairs=None):
    return {
        "id": rid,
        "question": question,
        "pairs": pairs or [["title", rid]],
        "body": body if body is not None else f"Body of {rid}. " * 10,
        "quality_flags": [],
    }


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "nq"
    splits = split_dir(root)
    splits.mkdir(parents=True)
    for name, ids in (
        ("train", ["a", "b"]),
        ("dev", ["c"]),
        ("test", ["d"]),
        ("nq_val", ["v"]),
    ):
        (splits / f"{name}.json").write_text(
            json.dumps({"records": [record(i) for i in ids]})
        )
    (splits / NEGATIVES_NAME).write_text(
        json.dumps({"negatives": {"a": ["b", "c", "d"], "b": ["a"]}})
    )
    return root


class TestLoadNegatives:
    def test_returns_the_negatives_by_query(self, corpus):
        assert load_negatives(corpus) == {"a": ["b", "c", "d"], "b": ["a"]}

    def test_keeps_the_mining_order(self, corpus):
        assert load_negatives(corpus)["a"][0] == "b"


class TestLoadCorpusIndex:
    def test_holds_the_three_partitions(self, corpus):
        assert set(load_corpus_index(corpus, verbose=False)) == {"a", "b", "c", "d"}

    def test_excludes_the_validation_split_by_default(self, corpus):
        assert "v" not in load_corpus_index(corpus, verbose=False)

    def test_adds_the_validation_split_on_request(self, corpus):
        index = load_corpus_index(corpus, include_nq_val=True, verbose=False)
        assert set(index) == {"a", "b", "c", "d", "v"}

    def test_keys_each_record_by_its_identifier(self, corpus):
        assert load_corpus_index(corpus, verbose=False)["a"]["id"] == "a"

    def test_names_each_file_when_verbose(self, corpus, capsys):
        load_corpus_index(corpus, include_nq_val=True, verbose=True)
        assert "train.json" in capsys.readouterr().out


class TestTruncateToBudget:
    def test_cuts_at_a_sentence_boundary(self):
        body = "One two three four five. " + " ".join(["word"] * 40)
        out = truncate_to_budget(body, 20, WordTokenizer())
        assert out == "One two three four five."

    def test_returns_a_short_body_unchanged(self):
        assert truncate_to_budget("One. Two.", 20, WordTokenizer()) == "One. Two."

    def test_returns_an_empty_string_below_the_minimum_budget(self):
        assert truncate_to_budget("One. Two. Three.", 0, WordTokenizer()) == ""

    def test_returns_an_empty_string_for_an_empty_body(self):
        assert truncate_to_budget("", 20, WordTokenizer()) == ""


class TestPrepareTrainRecords:
    def test_prepares_a_record_that_has_enough_negatives(self):
        prepared, dropped = prepare_train_records(
            [record("a")],
            {"a": ["b", "c", "d"]},
            {i: record(i) for i in "bcd"},
            TOKENIZERS,
            neg_k=3,
        )
        assert dropped == 0
        assert [r.id for r in prepared] == ["a"]

    def test_keeps_exactly_neg_k_negatives(self):
        prepared, _ = prepare_train_records(
            [record("a")],
            {"a": ["b", "c", "d"]},
            {i: record(i) for i in "bcd"},
            TOKENIZERS,
            neg_k=2,
        )
        assert len(prepared[0].neg_pairs_list) == 2
        assert len(prepared[0].neg_bodies_truncated) == 2

    def test_takes_the_negatives_in_mining_order(self):
        prepared, _ = prepare_train_records(
            [record("a")],
            {"a": ["d", "b"]},
            {i: record(i) for i in "bd"},
            TOKENIZERS,
            neg_k=2,
        )
        assert [p[0][1] for p in prepared[0].neg_pairs_list] == ["d", "b"]

    def test_drops_a_record_with_too_few_negatives(self):
        prepared, dropped = prepare_train_records(
            [record("a")],
            {"a": ["b"]},
            {"b": record("b")},
            TOKENIZERS,
            neg_k=3,
        )
        assert (prepared, dropped) == ([], 1)

    def test_drops_a_record_with_no_negatives(self):
        prepared, dropped = prepare_train_records(
            [record("a")], {}, {}, TOKENIZERS, neg_k=1
        )
        assert (prepared, dropped) == ([], 1)

    def test_drops_a_record_whose_body_does_not_fit(self):
        long_question = " ".join(["word"] * 600)
        prepared, dropped = prepare_train_records(
            [record("a", question=long_question)],
            {"a": ["b"]},
            {"b": record("b")},
            TOKENIZERS,
            neg_k=1,
        )
        assert (prepared, dropped) == ([], 1)

    def test_budgets_every_negative_to_the_query_budget(self):
        prepared, _ = prepare_train_records(
            [record("a")],
            {"a": ["b"]},
            {"b": record("b", body="Long. " * 400)},
            TOKENIZERS,
            neg_k=1,
        )
        budget = prepared[0].body_budget_tokens
        assert len(prepared[0].neg_bodies_truncated[0].split()) <= budget

    def test_gives_an_empty_body_to_a_negative_with_no_body(self):
        prepared, _ = prepare_train_records(
            [record("a")],
            {"a": ["b"]},
            {"b": record("b", body="")},
            TOKENIZERS,
            neg_k=1,
        )
        assert prepared[0].neg_bodies_truncated == [""]

    def test_defaults_to_seven_negatives(self):
        ids = [f"n{i}" for i in range(7)]
        prepared, _ = prepare_train_records(
            [record("a")],
            {"a": ids},
            {i: record(i) for i in ids},
            TOKENIZERS,
        )
        assert len(prepared[0].neg_pairs_list) == DEFAULT_TRAIN_NEGATIVES == 7


class TestPrepareDevRecords:
    def test_prepares_a_record(self):
        prepared = prepare_dev_records([record("c")], TOKENIZERS)
        assert [r.id for r in prepared] == ["c"]

    def test_carries_no_negatives(self):
        prepared = prepare_dev_records([record("c")], TOKENIZERS)
        assert prepared[0].neg_pairs_list == []
        assert prepared[0].neg_bodies_truncated == []

    def test_records_the_budget(self):
        prepared = prepare_dev_records([record("c")], TOKENIZERS)
        assert prepared[0].body_budget_tokens > 0

    def test_drops_a_record_whose_body_does_not_fit(self):
        long_question = " ".join(["word"] * 600)
        assert (
            prepare_dev_records([record("c", question=long_question)], TOKENIZERS) == []
        )


LONG_BODY = "One two three four five six seven eight nine ten. " * 80


class TestRosterBudget:
    """The budget is the tightest across the roster, not the model's own."""

    def roster(self):
        return {"word": WordTokenizer(), "char": CharTokenizer()}

    def test_takes_the_tightest_tokenizer(self):
        body = LONG_BODY
        wide, _ = prepare_train_records(
            [record("a", body=body)],
            {"a": ["b"]},
            {"b": record("b")},
            {"word": WordTokenizer()},
            neg_k=1,
        )
        tight, _ = prepare_train_records(
            [record("a", body=body)],
            {"a": ["b"]},
            {"b": record("b")},
            self.roster(),
            neg_k=1,
        )
        assert tight[0].body_budget_tokens < wide[0].body_budget_tokens

    def test_the_body_is_cut_to_the_tightest_budget(self):
        body = LONG_BODY
        wide, _ = prepare_train_records(
            [record("a", body=body)],
            {"a": ["b"]},
            {"b": record("b")},
            {"word": WordTokenizer()},
            neg_k=1,
        )
        tight, _ = prepare_train_records(
            [record("a", body=body)],
            {"a": ["b"]},
            {"b": record("b")},
            self.roster(),
            neg_k=1,
        )
        assert len(tight[0].truncated_body) < len(wide[0].truncated_body)

    def test_every_model_in_the_roster_gives_the_same_text(self):
        body = LONG_BODY
        first, _ = prepare_train_records(
            [record("a", body=body)],
            {"a": ["b"]},
            {"b": record("b")},
            self.roster(),
            neg_k=1,
        )
        reordered = {"char": CharTokenizer(), "word": WordTokenizer()}
        second, _ = prepare_train_records(
            [record("a", body=body)],
            {"a": ["b"]},
            {"b": record("b")},
            reordered,
            neg_k=1,
        )
        assert first[0].truncated_body == second[0].truncated_body
        assert first[0].body_budget_tokens == second[0].body_budget_tokens

    def test_the_development_records_share_the_budget(self):
        body = LONG_BODY
        dev = prepare_dev_records([record("c", body=body)], self.roster())
        train, _ = prepare_train_records(
            [record("c", body=body)],
            {"c": ["b"]},
            {"b": record("b")},
            self.roster(),
            neg_k=1,
        )
        assert dev[0].body_budget_tokens == train[0].body_budget_tokens
        assert dev[0].truncated_body == train[0].truncated_body

    def test_the_negatives_are_cut_with_the_tightest_tokenizer(self):
        body = LONG_BODY
        prepared, _ = prepare_train_records(
            [record("a", body=body)],
            {"a": ["b"]},
            {"b": record("b", body=body)},
            self.roster(),
            neg_k=1,
        )
        budget = prepared[0].body_budget_tokens
        assert len(prepared[0].neg_bodies_truncated[0]) <= budget
