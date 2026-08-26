"""Tests for fsr.common.rendering.

The shape assertions compare exact strings. A change to spacing, ordering, or
delimiters fails these tests.
"""

from __future__ import annotations

import pytest
from tests.fakes import CharTokenizer, WordTokenizer

from fsr.common.rendering import (
    FORMAT_NAMES,
    FORMATS,
    MIN_BODY_TOKENS,
    compute_body_budget,
    prepare_records_with_body,
    render_inline_kv,
    render_json,
    render_markdown,
    render_toml,
    render_yaml,
    truncate_body_semantic,
)

PAIRS = [("title", "Hamlet"), ("author", "Shakespeare")]
BODY = "A tragedy."
QUESTION = "who wrote hamlet"

ORDER_PRESERVING = ["yaml", "toml", "inline_kv", "markdown"]


def _record(body=BODY, question=QUESTION, pairs=PAIRS):
    return {"question": question, "pairs": pairs, "body": body, "id": "r1"}


class TestRenderers:
    def test_render_yaml_shape(self):
        assert render_yaml(PAIRS, BODY) == (
            "---\ntitle: Hamlet\nauthor: Shakespeare\n---\nA tragedy."
        )

    def test_render_json_shape(self):
        assert render_json(PAIRS, BODY) == (
            '{"title": "Hamlet", "author": "Shakespeare"}\nA tragedy.'
        )

    def test_render_toml_shape(self):
        assert render_toml(PAIRS, BODY) == (
            'title = "Hamlet"\nauthor = "Shakespeare"\nA tragedy.'
        )

    def test_render_inline_kv_shape(self):
        assert render_inline_kv(PAIRS, BODY) == (
            "title=Hamlet author=Shakespeare\nA tragedy."
        )

    def test_render_markdown_shape(self):
        assert render_markdown(PAIRS, BODY) == (
            "**title**: Hamlet\n**author**: Shakespeare\n\nA tragedy."
        )

    @pytest.mark.parametrize("name", FORMAT_NAMES)
    def test_renderer_appends_body_verbatim(self, name):
        assert FORMATS[name](PAIRS, BODY).endswith(BODY)

    @pytest.mark.parametrize("name", FORMAT_NAMES)
    def test_renderer_accepts_empty_body(self, name):
        rendered = FORMATS[name](PAIRS, "")
        assert "Hamlet" in rendered
        assert "Shakespeare" in rendered

    @pytest.mark.parametrize("name", FORMAT_NAMES)
    def test_renderer_accepts_empty_metadata(self, name):
        assert FORMATS[name]([], BODY).endswith(BODY)

    @pytest.mark.parametrize("name", ORDER_PRESERVING)
    def test_renderer_preserves_pair_order(self, name):
        rendered = FORMATS[name](PAIRS, "")
        assert rendered.index("Hamlet") < rendered.index("Shakespeare")

    @pytest.mark.parametrize("name", ORDER_PRESERVING)
    def test_renderer_preserves_repeated_keys(self, name):
        rendered = FORMATS[name]([("k", "first"), ("k", "second")], "")
        assert "first" in rendered and "second" in rendered

    def test_render_json_collapses_repeated_keys(self):
        assert render_json([("k", "first"), ("k", "second")], "") == '{"k": "second"}\n'

    def test_format_names_matches_formats_in_order(self):
        assert tuple(FORMATS) == FORMAT_NAMES
        assert FORMAT_NAMES == ("yaml", "json", "toml", "inline_kv", "markdown")


class TestComputeBodyBudget:
    def test_budget_arithmetic_for_single_tokenizer(self):
        # 512 context - 3 question - 6 metadata (yaml/toml tie) - 3 special - 5 buffer
        budget, tightest = compute_body_budget(
            QUESTION, PAIRS, {"word": WordTokenizer()}
        )
        assert (budget, tightest) == (495, "word")

    def test_budget_picks_the_tightest_tokenizer(self):
        tokenizers = {"word": WordTokenizer(), "char": CharTokenizer()}
        budget, tightest = compute_body_budget(QUESTION, PAIRS, tokenizers)
        assert tightest == "char"
        assert budget < 495

    def test_budget_uses_the_most_verbose_format(self):
        sparse = compute_body_budget(
            QUESTION,
            PAIRS,
            {"word": WordTokenizer()},
            formats={"inline_kv": render_inline_kv},
        )
        assert sparse == (499, "word")
        verbose = compute_body_budget(
            QUESTION, PAIRS, {"word": WordTokenizer()}, formats={"yaml": render_yaml}
        )
        assert verbose == (495, "word")

    def test_budget_respects_custom_max_tokens(self):
        budget, _ = compute_body_budget(
            QUESTION, PAIRS, {"word": WordTokenizer()}, max_tokens=128
        )
        assert budget == 111

    def test_budget_goes_negative_when_context_is_exhausted(self):
        budget, _ = compute_body_budget(
            QUESTION, PAIRS, {"word": WordTokenizer()}, max_tokens=5
        )
        assert budget == -12

    def test_budget_rejects_empty_tokenizers(self):
        with pytest.raises(ValueError, match="must not be empty"):
            compute_body_budget(QUESTION, PAIRS, {})


class TestTruncateBodySemantic:
    def test_truncate_returns_none_below_min_body_tokens(self):
        assert (
            truncate_body_semantic(BODY, MIN_BODY_TOKENS - 1, CharTokenizer()) is None
        )

    def test_truncate_returns_empty_string_for_empty_body(self):
        assert truncate_body_semantic("", 100, CharTokenizer()) == ""

    def test_truncate_returns_body_unchanged_when_within_budget(self):
        body = "Short enough."
        assert truncate_body_semantic(body, 100, CharTokenizer()) == body

    def test_truncate_cuts_at_sentence_boundary(self):
        body = "Alpha beta. Gamma delta epsilon zeta."
        assert truncate_body_semantic(body, 25, CharTokenizer()) == "Alpha beta."

    def test_truncate_accepts_terminator_at_end_of_cut(self):
        body = "Alpha beta gamma delta epsilon. More text follows."
        result = truncate_body_semantic(body, 31, CharTokenizer())
        assert result == "Alpha beta gamma delta epsilon."

    def test_truncate_falls_back_to_word_boundary_without_terminator(self):
        body = "alpha beta gamma delta epsilon theta"
        assert (
            truncate_body_semantic(body, 25, CharTokenizer())
            == "alpha beta gamma delta"
        )

    def test_truncate_ignores_terminator_not_followed_by_whitespace(self):
        body = "the value is 3.5 units and more words follow"
        result = truncate_body_semantic(body, 30, CharTokenizer())
        assert result == "the value is 3.5 units and"

    def test_truncate_falls_back_to_hard_cut_without_usable_space(self):
        body = "ab " + "c" * 40
        result = truncate_body_semantic(body, 25, CharTokenizer())
        assert result == "ab " + "c" * 22

    def test_truncate_ignores_terminator_outside_the_search_window(self):
        # The only terminator is at index 5. That index is before the 500-character
        # search window, so the function cuts at a word boundary.
        body = "Start. " + ("abcde fghij " * 60)
        result = truncate_body_semantic(body, 550, CharTokenizer())
        assert not result.endswith(".")
        assert result.endswith("fghij")
        assert len(result) > 500


class TestPrepareRecordsWithBody:
    def test_prepare_adds_budget_fields_and_keeps_originals(self):
        kept, stats = prepare_records_with_body([_record()], {"word": WordTokenizer()})
        assert len(kept) == 1
        assert kept[0]["truncated_body"] == BODY
        assert kept[0]["body_budget_tokens"] == 495
        assert kept[0]["tightest_tokeniser"] == "word"
        assert kept[0]["id"] == "r1"
        assert stats == {"dropped": 0, "tightest_counts": {"word": 1}}

    def test_prepare_does_not_mutate_input_records(self):
        record = _record()
        prepare_records_with_body([record], {"word": WordTokenizer()})
        assert record == _record()

    def test_prepare_drops_records_below_the_minimum_budget(self):
        kept, stats = prepare_records_with_body(
            [_record()], {"word": WordTokenizer()}, max_tokens=25
        )
        assert kept == []
        assert stats["dropped"] == 1
        assert stats["tightest_counts"] == {"word": 1}

    def test_prepare_counts_the_tightest_tokenizer_per_record(self):
        tokenizers = {"word": WordTokenizer(), "char": CharTokenizer()}
        _, stats = prepare_records_with_body([_record(), _record()], tokenizers)
        assert stats["tightest_counts"] == {"char": 2}

    def test_prepare_handles_no_records(self):
        kept, stats = prepare_records_with_body([], {"word": WordTokenizer()})
        assert kept == []
        assert stats == {"dropped": 0, "tightest_counts": {}}

    def test_prepare_forwards_custom_formats(self):
        kept, _ = prepare_records_with_body(
            [_record()],
            {"word": WordTokenizer()},
            formats={"inline_kv": render_inline_kv},
        )
        assert kept[0]["body_budget_tokens"] == 499
