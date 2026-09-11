"""Tests for fsr.formats.

The shape assertions compare exact strings. A change to spacing, ordering, or
delimiters fails these tests.
"""

from __future__ import annotations

import pytest

from fsr.formats import (
    FORMAT_NAMES,
    FORMATS,
    render_inline_kv,
    render_json,
    render_markdown,
    render_toml,
    render_yaml,
)

PAIRS = [("title", "Hamlet"), ("author", "Shakespeare")]
BODY = "A tragedy."
ORDER_PRESERVING = ["yaml", "toml", "inline_kv", "markdown"]


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
