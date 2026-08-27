"""Tests for fsr.h1.wikitext."""

from __future__ import annotations

from fsr.h1.wikitext import (
    DISPLAYNONE_SPAN_OPEN_RE,
    NOPRINT_SPAN_OPEN_RE,
    clean_cell,
    clean_key,
    extract_body,
    parse_infobox,
    quality_check,
    strip_nested_tables,
    strip_spans_matching,
)

LONG_PARA = "A sufficiently long paragraph of real article prose goes here."
PARSED = {"pairs": [("a", "1"), ("b", "2"), ("c", "3")], "body": "x" * 200}
SOURCE = {"infobox_html_raw": "y" * 300}


def infobox(*rows: str) -> str:
    return '<table class="infobox">' + "".join(rows) + "</table>"


def row(key: str, value: str) -> str:
    return f"<tr><th>{key}</th><td>{value}</td></tr>"


class TestStripNestedTables:
    def test_leaves_a_table_with_no_nesting_unchanged(self):
        html = infobox(row("Born", "1946"))
        assert strip_nested_tables(html) == html

    def test_removes_a_nested_table_and_keeps_the_wrapper(self):
        html = infobox("<tr><td><table><tr><td>inner</td></tr></table></td></tr>")
        result = strip_nested_tables(html)
        assert "inner" not in result
        assert result.startswith('<table class="infobox">')
        assert result.endswith("</table>")

    def test_removes_several_nested_tables(self):
        html = infobox("<tr><td><table>FIRST</table><table>SECOND</table></td></tr>")
        result = strip_nested_tables(html)
        assert "FIRST" not in result and "SECOND" not in result
        assert result.count("<table") == 1

    def test_removes_a_deeply_nested_table(self):
        html = infobox("<tr><td><table><table>deep</table></table></td></tr>")
        assert "deep" not in strip_nested_tables(html)

    def test_ignores_a_closing_tag_with_no_opening_tag(self):
        html = "</table>" + infobox(row("Born", "1946"))
        assert strip_nested_tables(html) == html


class TestStripSpansMatching:
    def test_removes_a_matching_span(self):
        text = 'before<span class="noprint">gone</span>after'
        assert strip_spans_matching(text, NOPRINT_SPAN_OPEN_RE) == "beforeafter"

    def test_removes_a_span_that_wraps_another_span(self):
        text = 'a<span class="noprint">x<span>inner</span>y</span>b'
        assert strip_spans_matching(text, NOPRINT_SPAN_OPEN_RE) == "ab"

    def test_keeps_a_span_that_does_not_match(self):
        text = "a<span>kept</span>b"
        assert strip_spans_matching(text, NOPRINT_SPAN_OPEN_RE) == text

    def test_removes_only_the_opening_tag_when_it_is_never_closed(self):
        text = 'a<span class="noprint">dangling'
        assert strip_spans_matching(text, NOPRINT_SPAN_OPEN_RE) == "adangling"

    def test_removes_several_matching_spans(self):
        text = '<span class="noprint">a</span>keep<span class="noprint">b</span>'
        assert strip_spans_matching(text, NOPRINT_SPAN_OPEN_RE) == "keep"

    def test_skips_a_match_already_inside_a_removed_span(self):
        text = '<span class="noprint">x<span class="noprint">y</span>z</span>tail'
        assert strip_spans_matching(text, NOPRINT_SPAN_OPEN_RE) == "tail"

    def test_matches_the_display_none_pattern(self):
        text = '<span style="display: none">hidden</span>shown'
        assert strip_spans_matching(text, DISPLAYNONE_SPAN_OPEN_RE) == "shown"


class TestCleanCell:
    def test_strips_tags_and_collapses_whitespace(self):
        assert clean_cell("<b>Bold</b>   text\n\nhere") == "Bold text here"

    def test_resolves_html_entities(self):
        assert clean_cell("Tom &amp; Jerry") == "Tom & Jerry"

    def test_removes_a_reference_superscript(self):
        raw = 'Born 1946<sup class="reference"><a href="#r1">[1]</a></sup>'
        assert clean_cell(raw) == "Born 1946"

    def test_removes_a_noprint_span(self):
        assert clean_cell('Kept<span class="noprint">dropped</span>') == "Kept"

    def test_removes_a_display_none_span(self):
        assert clean_cell('Kept<span style="display:none">dropped</span>') == "Kept"

    def test_returns_an_empty_string_for_markup_only(self):
        assert clean_cell("<span></span>   ") == ""


class TestCleanKey:
    def test_strips_each_bullet_glyph(self):
        for glyph in "\u2022*\u2013\u2014-\u2192\u203a\u00bb":
            assert clean_key(f"{glyph} Born") == "Born"

    def test_strips_leading_whitespace(self):
        assert clean_key(" \t Born") == "Born"

    def test_leaves_a_clean_key_unchanged(self):
        assert clean_key("Born") == "Born"

    def test_keeps_a_glyph_that_is_not_leading(self):
        assert clean_key("Born \u2013 died") == "Born \u2013 died"


class TestParseInfobox:
    def test_extracts_pairs_in_document_order(self):
        pairs, _ = parse_infobox(infobox(row("Born", "1946"), row("Died", "2011")))
        assert pairs == [("Born", "1946"), ("Died", "2011")]

    def test_ignores_a_row_without_both_cell_types(self):
        pairs, stats = parse_infobox(infobox("<tr><td>only a data cell</td></tr>"))
        assert pairs == []
        assert stats == {}

    def test_uses_the_first_header_and_data_cell_of_a_row(self):
        html = infobox("<tr><th>A</th><th>B</th><td>1</td><td>2</td></tr>")
        pairs, _ = parse_infobox(html)
        assert pairs == [("A", "1")]

    def test_drops_and_counts_an_empty_cell(self):
        pairs, stats = parse_infobox(infobox(row("", "1946")))
        assert pairs == []
        assert stats["empty_dropped"] == 1

    def test_drops_and_counts_an_oversized_cell(self):
        pairs, stats = parse_infobox(infobox(row("K" * 81, "v"), row("k", "v" * 401)))
        assert pairs == []
        assert stats["oversized_dropped"] == 2

    def test_drops_and_counts_a_repeated_key(self):
        pairs, stats = parse_infobox(infobox(row("Born", "1946"), row("Born", "1947")))
        assert pairs == [("Born", "1946")]
        assert stats["dupes_dropped"] == 1

    def test_removes_nested_table_content_before_parsing(self):
        html = infobox(
            "<tr><td><table><tr><th>Inner</th><td>x</td></tr></table></td></tr>",
            row("Born", "1946"),
        )
        pairs, _ = parse_infobox(html)
        assert pairs == [("Born", "1946")]

    def test_accepts_custom_length_limits(self):
        pairs, stats = parse_infobox(
            infobox(row("Born", "1946")), max_key_chars=2, max_value_chars=400
        )
        assert pairs == []
        assert stats["oversized_dropped"] == 1

    def test_strips_a_bullet_from_a_key(self):
        pairs, _ = parse_infobox(infobox(row("• Born", "1946")))
        assert pairs == [("Born", "1946")]


class TestExtractBody:
    def test_joins_paragraph_text(self):
        html = f"<p>{LONG_PARA}</p><p>{LONG_PARA}</p>"
        assert extract_body(html) == f"{LONG_PARA} {LONG_PARA}"

    def test_skips_a_paragraph_below_the_minimum_length(self):
        assert extract_body(f"<p>tiny</p><p>{LONG_PARA}</p>") == LONG_PARA

    def test_accepts_a_custom_minimum_paragraph_length(self):
        result = extract_body("<p>tiny</p>", min_paragraph_chars=1)
        assert result == "tiny"

    def test_truncates_to_the_target_length(self):
        html = f"<p>{LONG_PARA}</p>" * 10
        assert len(extract_body(html, target_chars=100)) == 100

    def test_stops_collecting_once_the_target_is_reached(self):
        html = f"<p>{LONG_PARA}</p>" * 10
        assert len(extract_body(html, target_chars=50)) == 50

    def test_removes_a_comment(self):
        html = f"<!-- hidden --><p>{LONG_PARA}</p>"
        assert "hidden" not in extract_body(html)

    def test_removes_a_navigation_box(self):
        html = f'<table class="navbox">{LONG_PARA} nav</table><p>{LONG_PARA}</p>'
        assert extract_body(html) == LONG_PARA

    def test_removes_a_contents_division(self):
        html = f'<div id="toc">{LONG_PARA} toc</div><p>{LONG_PARA}</p>'
        assert extract_body(html) == LONG_PARA

    def test_removes_a_contents_table(self):
        html = f'<table id="toc">{LONG_PARA} toc</table><p>{LONG_PARA}</p>'
        assert extract_body(html) == LONG_PARA

    def test_removes_an_edit_section_link(self):
        html = f'<p><span class="mw-editsection">edit</span>{LONG_PARA}</p>'
        assert extract_body(html) == LONG_PARA

    def test_returns_an_empty_string_when_no_paragraph_qualifies(self):
        assert extract_body("<div>no paragraphs</div>") == ""


class TestQualityCheck:
    def test_a_good_record_raises_no_flags(self):
        assert quality_check(PARSED, SOURCE) == []

    def test_flags_too_few_pairs(self):
        parsed = {**PARSED, "pairs": [("a", "1")]}
        assert quality_check(parsed, SOURCE) == ["too_few_pairs"]

    def test_flags_a_short_body(self):
        parsed = {**PARSED, "body": "short"}
        assert quality_check(parsed, SOURCE) == ["body_too_short"]

    def test_flags_a_short_infobox(self):
        assert quality_check(PARSED, {"infobox_html_raw": "y"}) == ["infobox_too_short"]

    def test_flags_an_answer_absent_from_the_content(self):
        source = {**SOURCE, "short_answers": ["nowhere"]}
        assert quality_check(PARSED, source) == ["answer_not_in_content"]

    def test_accepts_an_answer_present_in_a_value(self):
        source = {**SOURCE, "short_answers": ["1"]}
        assert quality_check(PARSED, source) == []

    def test_accepts_an_answer_present_in_the_body(self):
        parsed = {**PARSED, "body": "the answer is Waterloo " + "x" * 200}
        source = {**SOURCE, "short_answers": ["waterloo"]}
        assert quality_check(parsed, source) == []

    def test_skips_the_answer_check_when_short_answers_is_empty(self):
        source = {**SOURCE, "short_answers": []}
        assert quality_check(PARSED, source) == []

    def test_skips_the_answer_check_when_short_answers_is_absent(self):
        assert quality_check(PARSED, SOURCE) == []

    def test_raises_every_applicable_flag(self):
        parsed = {"pairs": [], "body": ""}
        source = {"infobox_html_raw": "", "short_answers": ["nope"]}
        assert quality_check(parsed, source) == [
            "too_few_pairs",
            "body_too_short",
            "infobox_too_short",
            "answer_not_in_content",
        ]

    def test_accepts_custom_thresholds(self):
        parsed = {"pairs": [("a", "1")], "body": "tiny"}
        source = {"infobox_html_raw": "y"}
        assert quality_check(parsed, source, 1, 1, 1) == []
