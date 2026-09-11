"""Infobox location in raw Wikipedia HTML."""

from __future__ import annotations

import re

INFOBOX_OPEN = re.compile(
    rb'<table\b[^>]*class="[^"]*infobox[^"]*"[^>]*>', re.IGNORECASE
)
TABLE_TAG = re.compile(rb"<(/?)table\b[^>]*>", re.IGNORECASE)


def find_infobox_ranges(
    html_bytes: bytes,
    open_pattern: re.Pattern[bytes] = INFOBOX_OPEN,
) -> list[tuple[int, int]]:
    """Find the byte range of every infobox table in the HTML.

    The search counts nested table tags. A table inside an infobox does not end
    the range. An infobox with no matching close tag is absent from the result.

    Args:
        html_bytes: The raw HTML.
        open_pattern: The pattern that matches an opening infobox tag.

    Returns:
        The start and end byte offset of each infobox, in document order. The
        end offset is the byte after the closing tag.
    """
    ranges = []
    for match in open_pattern.finditer(html_bytes):
        start = match.start()
        depth = 1
        pos = match.end()
        while pos < len(html_bytes) and depth > 0:
            tag = TABLE_TAG.search(html_bytes, pos)
            if not tag:
                break
            depth += -1 if tag.group(1) == b"/" else 1
            pos = tag.end()
            if depth == 0:
                ranges.append((start, pos))
                break
    return ranges


def in_any_range(start: int, end: int, ranges: list[tuple[int, int]]) -> bool:
    """Report whether a byte span falls inside any of the ranges.

    Args:
        start: The first byte of the span.
        end: The byte after the span.
        ranges: The start and end offsets to test against.

    Returns:
        True if one range contains the whole span.
    """
    return any(
        start >= range_start and end <= range_end for range_start, range_end in ranges
    )
