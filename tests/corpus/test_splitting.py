"""Tests for fsr.corpus.splitting."""

from __future__ import annotations

from fsr.corpus.splitting import split_by_article, strict_gated


def record(title: str, rec_id: str = "1", flags=None) -> dict:
    """Return a parsed record."""
    return {"id": rec_id, "title": title, "quality_flags": flags or []}


def corpus(n_titles: int = 100, per_title: int = 1) -> list[dict]:
    """Return records spread across a number of article titles."""
    return [
        record(f"Article {t}", f"{t}-{i}")
        for t in range(n_titles)
        for i in range(per_title)
    ]


class TestStrictGated:
    def test_keeps_unflagged_records(self):
        assert len(strict_gated([record("A"), record("B")])) == 2

    def test_drops_flagged_records(self):
        records = [record("A"), record("B", flags=["too_few_pairs"])]
        assert [r["title"] for r in strict_gated(records)] == ["A"]

    def test_handles_no_records(self):
        assert strict_gated([]) == []


class TestSplitByArticle:
    def test_assigns_every_record(self):
        s = split_by_article(corpus(100))
        assert len(s.train) + len(s.dev) + len(s.test) == 100

    def test_no_title_spans_two_splits(self):
        s = split_by_article(corpus(100, per_title=3))
        titles = [{r["title"] for r in part} for part in (s.train, s.dev, s.test)]
        assert titles[0].isdisjoint(titles[1])
        assert titles[0].isdisjoint(titles[2])
        assert titles[1].isdisjoint(titles[2])

    def test_keeps_every_record_of_a_title_together(self):
        s = split_by_article(corpus(50, per_title=4))
        for part in (s.train, s.dev, s.test):
            for title in {r["title"] for r in part}:
                assert sum(1 for r in part if r["title"] == title) == 4

    def test_splits_by_the_configured_shares(self):
        s = split_by_article(corpus(100))
        assert s.counts["n_articles_train"] == 80
        assert s.counts["n_articles_dev"] == 10
        assert s.counts["n_articles_test"] == 10

    def test_the_test_split_takes_the_remainder(self):
        s = split_by_article(corpus(97))
        assert s.counts["n_articles_test"] == 97 - int(97 * 0.8) - int(97 * 0.1)

    def test_is_deterministic_for_a_given_seed(self):
        first = split_by_article(corpus(60), seed=7)
        second = split_by_article(corpus(60), seed=7)
        assert [r["id"] for r in first.train] == [r["id"] for r in second.train]

    def test_a_different_seed_gives_a_different_partition(self):
        first = split_by_article(corpus(60), seed=1)
        second = split_by_article(corpus(60), seed=2)
        assert [r["id"] for r in first.train] != [r["id"] for r in second.train]

    def test_the_partition_ignores_input_order(self):
        records = corpus(60)
        forward = split_by_article(records, seed=3)
        backward = split_by_article(list(reversed(records)), seed=3)
        assert {r["id"] for r in forward.train} == {r["id"] for r in backward.train}

    def test_accepts_custom_shares(self):
        s = split_by_article(corpus(100), frac_train=0.5, frac_dev=0.25)
        assert s.counts["n_articles_train"] == 50
        assert s.counts["n_articles_dev"] == 25
        assert s.counts["n_articles_test"] == 25

    def test_counts_records_as_well_as_articles(self):
        s = split_by_article(corpus(100, per_title=2))
        assert s.counts["n_records_train"] == 160
        assert s.counts["n_articles_total"] == 100

    def test_handles_no_records(self):
        s = split_by_article([])
        assert (s.train, s.dev, s.test) == ([], [], [])
        assert s.counts["n_articles_total"] == 0

    def test_a_single_article_lands_in_the_test_split(self):
        s = split_by_article(corpus(1))
        assert len(s.test) == 1
        assert s.train == [] and s.dev == []
