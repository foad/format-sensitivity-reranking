"""Tests for fsr.h1.infobox."""

from __future__ import annotations

import re

import pytest

from fsr.h1.infobox import INFOBOX_OPEN, find_infobox_ranges, in_any_range

OPEN = b'<table class="infobox vcard">'
CLOSE = b"</table>"
RANGES = [(10, 20), (30, 40)]


def html(*parts: bytes) -> bytes:
    return b"<html><body>" + b"".join(parts) + b"</body></html>"


class TestFindInfoboxRanges:
    def test_finds_a_single_infobox(self):
        doc = html(OPEN, b"<tr><th>Born</th><td>1946</td></tr>", CLOSE)
        ((start, end),) = find_infobox_ranges(doc)
        assert doc[start:end].startswith(OPEN)
        assert doc[start:end].endswith(CLOSE)

    def test_returns_nothing_when_no_infobox_is_present(self):
        assert find_infobox_ranges(html(b"<table><tr><td>x</td></tr></table>")) == []

    def test_spans_a_nested_table(self):
        doc = html(OPEN, b"<tr><td><table><tr><td>x</td></tr></table></td></tr>", CLOSE)
        ((start, end),) = find_infobox_ranges(doc)
        assert doc[start:end].count(b"<table") == 2
        assert doc[start:end].endswith(CLOSE)

    def test_finds_each_infobox_in_document_order(self):
        doc = html(OPEN, b"a", CLOSE, b"<p>gap</p>", OPEN, b"b", CLOSE)
        ranges = find_infobox_ranges(doc)
        assert len(ranges) == 2
        assert ranges[0][1] <= ranges[1][0]

    def test_skips_an_infobox_that_is_never_closed(self):
        assert find_infobox_ranges(html(OPEN, b"<tr><td>x</td></tr>")) == []

    def test_skips_an_infobox_whose_nested_table_ends_the_document(self):
        assert find_infobox_ranges(OPEN + b"<table>") == []

    def test_matches_a_class_list_containing_infobox(self):
        doc = html(b'<table class="wikitable infobox biography">x', CLOSE)
        assert len(find_infobox_ranges(doc)) == 1

    def test_ignores_tag_and_attribute_case(self):
        doc = html(b'<TABLE CLASS="Infobox">x</TABLE>')
        assert len(find_infobox_ranges(doc)) == 1

    def test_accepts_a_custom_open_pattern(self):
        doc = html(b'<table class="navbox">x', CLOSE)
        pattern = re.compile(rb'<table\b[^>]*class="[^"]*navbox[^"]*"[^>]*>', re.I)
        assert len(find_infobox_ranges(doc, open_pattern=pattern)) == 1
        assert find_infobox_ranges(doc) == []

    def test_the_default_pattern_is_the_module_constant(self):
        doc = html(OPEN, b"x", CLOSE)
        assert find_infobox_ranges(doc) == find_infobox_ranges(
            doc, open_pattern=INFOBOX_OPEN
        )


class TestInAnyRange:
    def test_a_contained_span_is_inside(self):
        assert in_any_range(12, 18, RANGES)

    def test_a_span_matching_a_range_exactly_is_inside(self):
        assert in_any_range(10, 20, RANGES)

    def test_a_span_overlapping_two_ranges_is_outside(self):
        assert not in_any_range(15, 35, RANGES)

    def test_a_span_before_every_range_is_outside(self):
        assert not in_any_range(0, 5, RANGES)

    def test_a_span_crossing_a_range_boundary_is_outside(self):
        assert not in_any_range(18, 25, RANGES)

    def test_a_span_in_the_second_range_is_inside(self):
        assert in_any_range(31, 39, RANGES)

    def test_no_ranges_means_outside(self):
        assert not in_any_range(1, 2, [])

    @pytest.mark.parametrize("start,end", [(10, 10), (20, 20)])
    def test_an_empty_span_on_a_boundary_is_inside(self, start, end):
        assert in_any_range(start, end, RANGES)
