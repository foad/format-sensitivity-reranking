from __future__ import annotations

import pytest

from fsr.corpus.config import DEFAULT
from fsr.corpus.prose import (
    ANSWER_IN_INFOBOX,
    NO_LONG_ANSWER,
    NO_VERIFIED_ANSWER,
    PROSE_NEGATIVES,
    SKIP_REASONS,
    TOO_SHORT,
    ProseStats,
    answer_in_text,
    as_candidate,
    build_prose_eval,
    clean_html_to_text,
    iter_prose,
    match_example,
    prose_doc,
)

PROSE = (
    "<p>The bridge was designed by Alice Brown and opened in 1946. "
    "It carries the main road across the river and remains in daily use "
    "by the town, which has grown around it over the following decades.</p>"
)
INFOBOX = '<table class="infobox vcard"><tr><th>Born</th><td>1946</td></tr></table>'


def example(html_text, start, end, answers=("Alice Brown",), rid="1"):
    return {
        "id": rid,
        "document": {"html": html_text, "title": f"Article {rid}"},
        "question": {"text": "who designed the bridge"},
        "annotations": {
            "long_answer": [{"start_byte": start, "end_byte": end}],
            "short_answers": [{"text": list(answers)}],
        },
    }


def prose_example(**kwargs):
    html_text = f"<html>{INFOBOX}{PROSE}</html>"
    start = html_text.index("<p>")
    return example(html_text, start, start + len(PROSE), **kwargs)


def infobox_example():
    html_text = f"<html>{INFOBOX}{PROSE}</html>"
    start = html_text.index("<table")
    return example(html_text, start, start + len(INFOBOX))


class TestCleanHtmlToText:
    def test_removes_the_tags(self):
        assert "<p>" not in clean_html_to_text("<p>Hello world.</p>")

    def test_resolves_an_entity(self):
        assert clean_html_to_text("<p>Fish &amp; chips</p>") == "Fish & chips"

    def test_collapses_whitespace(self):
        assert clean_html_to_text("<p>a\n\n  b</p>") == "a b"

    def test_returns_nothing_for_empty_markup(self):
        assert clean_html_to_text("<p></p>") == ""


class TestAnswerInText:
    def test_finds_the_answer(self):
        assert answer_in_text("opened by Alice Brown in 1946", ["Alice Brown"])

    def test_ignores_case(self):
        assert answer_in_text("opened by alice brown", ["Alice Brown"])

    def test_rejects_an_absent_answer(self):
        assert not answer_in_text("opened in 1946", ["Alice Brown"])

    def test_rejects_an_example_with_no_answer(self):
        assert not answer_in_text("opened in 1946", [])

    def test_ignores_an_empty_answer_string(self):
        assert not answer_in_text("opened in 1946", [""])

    def test_accepts_any_of_several(self):
        assert answer_in_text("opened in 1946", ["Alice Brown", "1946"])


class TestProseStats:
    def test_starts_at_zero(self):
        stats = ProseStats()
        assert stats.scanned == 0
        assert set(stats.skipped) == set(SKIP_REASONS)
        assert sum(stats.skipped.values()) == 0

    def test_counts_a_rejection(self):
        stats = ProseStats()
        stats.skip(TOO_SHORT)
        assert stats.skipped[TOO_SHORT] == 1

    def test_two_instances_do_not_share_counts(self):
        first = ProseStats()
        first.skip(TOO_SHORT)
        assert ProseStats().skipped[TOO_SHORT] == 0


class TestMatchExample:
    def test_keeps_a_prose_answer(self):
        record = match_example(prose_example())
        assert record is not None
        assert "Alice Brown" in record["long_answer_text"]

    def test_records_the_identity_of_the_example(self):
        record = match_example(prose_example(rid="7"))
        assert record["id"] == "7"
        assert record["title"] == "Article 7"
        assert record["question"] == "who designed the bridge"

    def test_records_the_short_answers(self):
        assert match_example(prose_example())["short_answers"] == ["Alice Brown"]

    def test_records_the_length_before_the_cut(self):
        record = match_example(prose_example())
        assert record["long_answer_chars"] == len(clean_html_to_text(PROSE))

    def test_rejects_an_answer_inside_an_infobox(self):
        stats = ProseStats()
        assert match_example(infobox_example(), stats=stats) is None
        assert stats.skipped[ANSWER_IN_INFOBOX] == 1

    def test_keeps_an_article_with_no_infobox(self):
        html_text = f"<html>{PROSE}</html>"
        start = html_text.index("<p>")
        assert match_example(example(html_text, start, start + len(PROSE))) is not None

    def test_rejects_an_example_with_no_long_answer(self):
        ex = prose_example()
        ex["annotations"]["long_answer"] = []
        stats = ProseStats()
        assert match_example(ex, stats=stats) is None
        assert stats.skipped[NO_LONG_ANSWER] == 1

    def test_rejects_a_long_answer_with_no_byte_range(self):
        ex = prose_example()
        ex["annotations"]["long_answer"] = [{"start_byte": -1, "end_byte": -1}]
        stats = ProseStats()
        assert match_example(ex, stats=stats) is None
        assert stats.skipped[NO_LONG_ANSWER] == 1

    def test_rejects_a_malformed_long_answer(self):
        ex = prose_example()
        ex["annotations"]["long_answer"] = ["not a mapping"]
        stats = ProseStats()
        assert match_example(ex, stats=stats) is None
        assert stats.skipped[NO_LONG_ANSWER] == 1

    def test_rejects_a_passage_that_is_too_short(self):
        stats = ProseStats()
        assert (
            match_example(prose_example(), min_body_chars=10_000, stats=stats) is None
        )
        assert stats.skipped[TOO_SHORT] == 1

    def test_rejects_a_passage_whose_answer_is_absent(self):
        stats = ProseStats()
        ex = prose_example(answers=("Zebedee Quux",))
        assert match_example(ex, stats=stats) is None
        assert stats.skipped[NO_VERIFIED_ANSWER] == 1

    def test_rejects_an_example_with_no_short_answer(self):
        ex = prose_example()
        ex["annotations"]["short_answers"] = []
        stats = ProseStats()
        assert match_example(ex, stats=stats) is None
        assert stats.skipped[NO_VERIFIED_ANSWER] == 1

    def test_cuts_the_passage_at_the_limit(self):
        ex = prose_example(answers=("The bridge",))
        record = match_example(ex, min_body_chars=5, body_chars=20)
        assert len(record["long_answer_text"]) == 20

    def test_records_the_length_before_the_cut_too(self):
        ex = prose_example(answers=("The bridge",))
        record = match_example(ex, min_body_chars=5, body_chars=20)
        assert record["long_answer_chars"] > 20

    def test_verifies_the_answer_after_the_cut(self):
        stats = ProseStats()
        ex = prose_example(answers=("daily use",))
        assert match_example(ex, min_body_chars=5, body_chars=20, stats=stats) is None
        assert stats.skipped[NO_VERIFIED_ANSWER] == 1

    def test_runs_without_counters(self):
        assert match_example(prose_example(), min_body_chars=10_000) is None

    def test_uses_the_corpus_defaults(self):
        record = match_example(prose_example())
        assert DEFAULT.min_body_chars <= record["long_answer_chars"]
        assert len(record["long_answer_text"]) <= DEFAULT.body_chars


class TestIterProse:
    @pytest.fixture
    def streamed(self, monkeypatch):
        seen = {}

        def fake_load_dataset(dataset, split=None, streaming=None, revision=None):
            seen.update(
                dataset=dataset, split=split, streaming=streaming, revision=revision
            )
            return [prose_example(rid=str(i)) for i in range(3)] + [infobox_example()]

        monkeypatch.setattr("fsr.corpus.prose.load_dataset", fake_load_dataset)
        return seen

    @pytest.mark.usefixtures("streamed")
    def test_yields_only_the_prose_records(self):
        assert len(list(iter_prose(verbose=False))) == 3

    def test_streams_the_pinned_revision(self, streamed):
        list(iter_prose(verbose=False))
        assert streamed["revision"] == DEFAULT.dataset_revision
        assert streamed["dataset"] == DEFAULT.dataset
        assert streamed["streaming"] is True

    def test_reads_the_validation_split_by_default(self, streamed):
        list(iter_prose(verbose=False))
        assert streamed["split"] == "validation"

    @pytest.mark.usefixtures("streamed")
    def test_counts_what_it_scanned_and_kept(self):
        stats = ProseStats()
        list(iter_prose(verbose=False, stats=stats))
        assert stats.scanned == 4
        assert stats.matched == 3
        assert stats.skipped[ANSWER_IN_INFOBOX] == 1

    @pytest.mark.usefixtures("streamed")
    def test_stops_at_the_limit(self):
        assert len(list(iter_prose(n_limit=2, verbose=False))) == 2

    @pytest.mark.usefixtures("streamed")
    def test_reports_progress_when_asked(self, capsys):
        list(iter_prose(verbose=True))
        printed = capsys.readouterr().out
        assert "Streaming" in printed
        assert "Done: scanned 4" in printed

    def test_reports_every_thousandth_example(self, monkeypatch, capsys):
        monkeypatch.setattr(
            "fsr.corpus.prose.load_dataset",
            lambda *_a, **_k: [prose_example(rid=str(i)) for i in range(2)],
        )
        monkeypatch.setattr("fsr.corpus.prose.PROGRESS_EVERY", 1)
        list(iter_prose(verbose=True))
        assert "scanned 1," in capsys.readouterr().out


def prose_record(rid, title, text, answers=("1946",)):
    return {
        "id": str(rid),
        "title": title,
        "question": f"who built bridge {rid}",
        "long_answer_text": text,
        "long_answer_chars": len(text),
        "short_answers": list(answers),
    }


def corpus(n_articles=20, per_article=1):
    return [
        prose_record(
            f"{a}-{b}",
            f"Article {a}",
            f"The bridge {a} opened in 194{a % 10} near the town of Place{a}.",
        )
        for a in range(n_articles)
        for b in range(per_article)
    ]


class TestProseDoc:
    def test_indexes_the_title_and_the_passage(self):
        record = prose_record(1, "Bridge", "Opened in 1946.")
        assert prose_doc(record) == "Bridge Opened in 1946."

    def test_carries_no_markup(self):
        assert "<" not in prose_doc(prose_record(1, "T", "Plain text."))


class TestAsCandidate:
    def test_keeps_the_ranking_fields(self):
        record = prose_record(1, "Bridge", "Opened in 1946.")
        assert as_candidate(record) == {
            "id": "1",
            "title": "Bridge",
            "text": "Opened in 1946.",
        }

    def test_drops_the_question_and_the_answers(self):
        candidate = as_candidate(prose_record(1, "Bridge", "Text."))
        assert "question" not in candidate
        assert "short_answers" not in candidate


class TestBuildProseEval:
    def test_covers_every_record(self):
        records = corpus()
        resolved, _, _ = build_prose_eval(records, k=5, verbose=False)
        assert [r["id"] for r in resolved] == [r["id"] for r in records]

    def test_gives_each_record_the_asked_negatives(self):
        resolved, _, _ = build_prose_eval(corpus(), k=5, verbose=False)
        assert all(len(r["negatives"]) == 5 for r in resolved)

    def test_defaults_to_fifteen_negatives(self):
        resolved, _, _ = build_prose_eval(corpus(n_articles=30), verbose=False)
        assert len(resolved[0]["negatives"]) == PROSE_NEGATIVES == 15

    def test_the_positive_is_the_record_itself(self):
        records = corpus()
        resolved, _, _ = build_prose_eval(records, k=5, verbose=False)
        assert resolved[0]["positive"] == as_candidate(records[0])

    def test_never_draws_a_negative_from_the_same_article(self):
        records = corpus(n_articles=20, per_article=2)
        resolved, _, _ = build_prose_eval(records, k=5, verbose=False)
        for entry in resolved:
            assert all(n["title"] != entry["title"] for n in entry["negatives"])

    def test_draws_each_negative_from_a_distinct_article(self):
        resolved, _, _ = build_prose_eval(corpus(), k=5, verbose=False)
        for entry in resolved:
            titles = [n["title"] for n in entry["negatives"]]
            assert len(set(titles)) == len(titles)

    def test_carries_the_question_and_the_answers(self):
        resolved, _, _ = build_prose_eval(corpus(), k=5, verbose=False)
        assert resolved[0]["question"].startswith("who built")
        assert resolved[0]["short_answers"] == ["1946"]

    def test_counts_the_distinct_articles(self):
        _, _, n_articles = build_prose_eval(
            corpus(n_articles=12, per_article=2), k=5, verbose=False
        )
        assert n_articles == 12

    def test_counts_the_records_that_needed_the_fill(self):
        _, filled, _ = build_prose_eval(corpus(n_articles=20), k=5, verbose=False)
        assert filled >= 0

    def test_the_fill_runs_when_retrieval_is_title_poor(self):
        crowd = [
            prose_record(f"c{i}", "Crowded", "bridge opened 1946 town place")
            for i in range(60)
        ]
        rest = [
            prose_record(f"d{i}", f"Distinct {i}", "unrelated prose about lichen")
            for i in range(8)
        ]
        _, filled, _ = build_prose_eval(crowd + rest, k=5, verbose=False)
        assert filled > 0

    def test_refuses_a_corpus_with_too_few_articles(self):
        with pytest.raises(ValueError, match="negatives available"):
            build_prose_eval(corpus(n_articles=3), k=5, verbose=False)

    def test_the_seed_fixes_the_result(self):
        first, _, _ = build_prose_eval(corpus(), k=5, seed=3, verbose=False)
        second, _, _ = build_prose_eval(corpus(), k=5, seed=3, verbose=False)
        assert first == second

    def test_reports_progress_when_asked(self, capsys):
        build_prose_eval(corpus(), k=5, verbose=True)
        assert "Indexing" in capsys.readouterr().out
