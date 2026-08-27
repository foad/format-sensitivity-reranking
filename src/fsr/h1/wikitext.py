"""Infobox and body-text extraction from cached Wikipedia HTML."""

from __future__ import annotations

import html
import re
from collections import Counter

TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")

REFERENCE_SUP_RE = re.compile(
    r'<sup\b[^>]*class="[^"]*reference[^"]*"[^>]*>.*?</sup>', re.DOTALL | re.IGNORECASE
)
NOPRINT_SPAN_OPEN_RE = re.compile(
    r'<span\b[^>]*class="[^"]*noprint[^"]*"[^>]*>', re.IGNORECASE
)
DISPLAYNONE_SPAN_OPEN_RE = re.compile(
    r'<span\b[^>]*style="[^"]*display:\s*none[^"]*"[^>]*>', re.IGNORECASE
)
ANY_SPAN_TAG_RE = re.compile(r"<(/?)span\b[^>]*>", re.IGNORECASE)

TABLE_TAG_RE = re.compile(r"<(/?)table\b[^>]*>", re.IGNORECASE)
ROW_RE = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.DOTALL | re.IGNORECASE)
TH_RE = re.compile(r"<th\b[^>]*>(.*?)</th>", re.DOTALL | re.IGNORECASE)
TD_RE = re.compile(r"<td\b[^>]*>(.*?)</td>", re.DOTALL | re.IGNORECASE)

# Bullet glyphs and leading whitespace stripped from metadata keys
BULLET_CHARS = "\u2022*\u2013\u2014-\u2192\u203a\u00bb \t"

COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
NAVBOX_RE = re.compile(
    r'<table\b[^>]*class="[^"]*navbox[^"]*"[^>]*>.*?</table>',
    re.DOTALL | re.IGNORECASE,
)
TOC_DIV_RE = re.compile(
    r'<div\b[^>]*id="toc"[^>]*>.*?</div>', re.DOTALL | re.IGNORECASE
)
TOC_TABLE_RE = re.compile(
    r'<table\b[^>]*id="toc"[^>]*>.*?</table>', re.DOTALL | re.IGNORECASE
)
EDITSECTION_RE = re.compile(
    r'<span\b[^>]*class="[^"]*mw-editsection[^"]*"[^>]*>.*?</span>',
    re.DOTALL | re.IGNORECASE,
)
P_RE = re.compile(r"<p\b[^>]*>(.*?)</p>", re.DOTALL | re.IGNORECASE)

MAX_KEY_CHARS = 80
MAX_VALUE_CHARS = 400
MIN_PARAGRAPH_CHARS = 30
DEFAULT_BODY_CHARS = 3000
DEFAULT_MIN_PAIRS = 3
DEFAULT_MIN_BODY_CHARS = 100
DEFAULT_MIN_INFOBOX_CHARS = 200


def strip_nested_tables(html_slice: str) -> str:
    """Remove every table nested inside the outermost table.

    The outermost table tags remain. A closing tag with no matching opening tag
    is ignored.

    Args:
        html_slice: The HTML of one table.

    Returns:
        The HTML with each nested table replaced by a space.
    """
    depth = 0
    starts: list[int] = []
    nested: list[tuple[int, int]] = []
    for tag in TABLE_TAG_RE.finditer(html_slice):
        if tag.group(1) == "":
            depth += 1
            starts.append(tag.start())
        else:
            if not starts:
                continue
            start = starts.pop()
            if depth > 1:
                nested.append((start, tag.end()))
            depth -= 1
    if not nested:
        return html_slice
    result = html_slice
    for start, end in sorted(nested, reverse=True):
        result = result[:start] + " " + result[end:]
    return result


def strip_spans_matching(text: str, opening_re: re.Pattern[str]) -> str:
    """Remove each span element whose opening tag matches the pattern.

    The search counts nested span tags, so a span that wraps another span is
    removed up to its own closing tag. An opening tag with no closing tag is
    removed on its own, and its content remains.

    Args:
        text: The HTML to filter.
        opening_re: The pattern that matches an opening span tag.

    Returns:
        The HTML without the matching span elements.
    """
    result: list[str] = []
    pos = 0
    for m in opening_re.finditer(text):
        if m.start() < pos:
            continue
        result.append(text[pos : m.start()])
        depth = 1
        end = None
        for tag in ANY_SPAN_TAG_RE.finditer(text, m.end()):
            depth += 1 if tag.group(1) == "" else -1
            if depth == 0:
                end = tag.end()
                break
        pos = end if end is not None else m.end()
    result.append(text[pos:])
    return "".join(result)


def clean_cell(raw: str) -> str:
    """Reduce one table cell to plain text.

    Reference superscripts, noprint spans, and display:none spans are removed
    before the remaining tags. HTML entities are resolved and whitespace runs
    become single spaces.

    Args:
        raw: The inner HTML of the cell.

    Returns:
        The plain text of the cell.
    """
    s = REFERENCE_SUP_RE.sub(" ", raw)
    s = strip_spans_matching(s, NOPRINT_SPAN_OPEN_RE)
    s = strip_spans_matching(s, DISPLAYNONE_SPAN_OPEN_RE)
    s = TAG_RE.sub(" ", s)
    s = html.unescape(s)
    return WS_RE.sub(" ", s).strip()


def clean_key(k: str) -> str:
    """Remove leading bullet characters and whitespace from a metadata key."""
    return k.lstrip(BULLET_CHARS).strip()


def parse_infobox(
    html_slice: str,
    *,
    max_key_chars: int = MAX_KEY_CHARS,
    max_value_chars: int = MAX_VALUE_CHARS,
) -> tuple[list[tuple[str, str]], Counter]:
    """Extract the key-value pairs from one infobox.

    A row contributes a pair when it holds both a header cell and a data cell.
    The first header cell and the first data cell of the row are used. A row is
    dropped when either cell is empty, when either cell is too long, or when the
    key already appeared.

    Args:
        html_slice: The HTML of one infobox table.
        max_key_chars: The longest key to keep.
        max_value_chars: The longest value to keep.

    Returns:
        The pairs in document order, and a count of dropped rows by reason. The
        reasons are empty_dropped, oversized_dropped, and dupes_dropped.
    """
    unnested = strip_nested_tables(html_slice)
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    stats: Counter = Counter()
    for row in ROW_RE.findall(unnested):
        th = TH_RE.search(row)
        td = TD_RE.search(row)
        if not (th and td):
            continue
        k = clean_key(clean_cell(th.group(1)))
        v = clean_cell(td.group(1))
        if not k or not v:
            stats["empty_dropped"] += 1
            continue
        if len(k) > max_key_chars or len(v) > max_value_chars:
            stats["oversized_dropped"] += 1
            continue
        if k in seen:
            stats["dupes_dropped"] += 1
            continue
        seen.add(k)
        pairs.append((k, v))
    return pairs, stats


def extract_body(
    post_infobox_html: str,
    target_chars: int = DEFAULT_BODY_CHARS,
    *,
    min_paragraph_chars: int = MIN_PARAGRAPH_CHARS,
) -> str:
    """Extract the article prose that follows the infobox.

    Comments, navigation boxes, contents tables, and edit-section links are
    removed first. The text of each remaining paragraph is joined until the
    target length is reached. A paragraph shorter than min_paragraph_chars is
    skipped.

    Args:
        post_infobox_html: The HTML after the infobox.
        target_chars: The length at which collection stops.
        min_paragraph_chars: The shortest paragraph to keep.

    Returns:
        The joined paragraph text, truncated to target_chars.
    """
    s = COMMENT_RE.sub(" ", post_infobox_html)
    s = NAVBOX_RE.sub(" ", s)
    s = TOC_DIV_RE.sub(" ", s)
    s = TOC_TABLE_RE.sub(" ", s)
    s = EDITSECTION_RE.sub(" ", s)
    parts: list[str] = []
    total = 0
    for m in P_RE.finditer(s):
        text = TAG_RE.sub(" ", m.group(1))
        text = html.unescape(text)
        text = WS_RE.sub(" ", text).strip()
        if len(text) < min_paragraph_chars:
            continue
        parts.append(text)
        total += len(text) + 1
        if total >= target_chars:
            break
    return " ".join(parts)[:target_chars]


def quality_check(
    parsed: dict,
    source: dict,
    min_pairs: int = DEFAULT_MIN_PAIRS,
    min_body: int = DEFAULT_MIN_BODY_CHARS,
    min_infobox: int = DEFAULT_MIN_INFOBOX_CHARS,
) -> list[str]:
    """Return the quality flags raised by one parsed record.

    An empty list means the record passes. The answer flag applies only when the
    source record carries a non-empty short_answers field.

    Args:
        parsed: The record with its pairs and body.
        source: The cache record with its raw infobox HTML and short answers.
        min_pairs: The fewest pairs a record may hold.
        min_body: The shortest body a record may hold.
        min_infobox: The shortest raw infobox HTML a record may hold.

    Returns:
        The flags raised, from too_few_pairs, body_too_short, infobox_too_short,
        and answer_not_in_content.
    """
    flags = []
    if len(parsed["pairs"]) < min_pairs:
        flags.append("too_few_pairs")
    if len(parsed["body"]) < min_body:
        flags.append("body_too_short")
    if len(source["infobox_html_raw"]) < min_infobox:
        flags.append("infobox_too_short")
    if "short_answers" in source:
        sa = source["short_answers"] or []
        if sa:
            haystack = (
                " ".join(v for _, v in parsed["pairs"]) + " " + parsed["body"]
            ).lower()
            if not any(ans and ans.lower() in haystack for ans in sa):
                flags.append("answer_not_in_content")
    return flags
