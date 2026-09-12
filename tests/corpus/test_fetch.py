"""Tests for scripts.corpus.fetch."""

from __future__ import annotations

import json

from scripts.corpus import fetch as mod

from fsr.corpus import nq

OPEN = '<table class="infobox">'
CLOSE = "</table>"
PREFIX = "<html><body>"
INFOBOX = OPEN + "<tr><th>Born</th><td>1946</td></tr>" + CLOSE
DOC = PREFIX + INFOBOX + "<p>Lead prose.</p></body></html>"
IB_START = len(PREFIX)
IB_END = IB_START + len(INFOBOX)


def example(
    html=DOC,
    long_answer=None,
    short_answers=None,
    ex_id="42",
    title="Person",
    question="when was the person born",
):
    if long_answer is None:
        long_answer = [{"start_byte": IB_START, "end_byte": IB_END}]
    annotations = {"long_answer": long_answer}
    if short_answers is not None:
        annotations["short_answers"] = short_answers
    return {
        "id": ex_id,
        "document": {"html": html, "title": title},
        "question": {"text": question},
        "annotations": annotations,
    }


class TestFetchSplit:
    def _run(self, monkeypatch, examples, tmp_path, n_limit=0):
        monkeypatch.setattr(nq, "load_dataset", lambda *_a, **_k: iter(examples))
        out = tmp_path / "matched_train.json"
        mod.fetch_split("train", out, n_limit)
        return json.loads(out.read_text())

    def test_writes_only_matching_examples(self, monkeypatch, tmp_path):
        examples = [example(ex_id="a"), example(html="<p>none</p>"), example(ex_id="b")]
        data = self._run(monkeypatch, examples, tmp_path)
        assert data["n_scanned"] == 3
        assert data["n_matched"] == 2
        assert [r["id"] for r in data["records"]] == ["a", "b"]

    def test_passes_the_dataset_and_revision_to_the_loader(self, monkeypatch, tmp_path):
        seen = {}

        def loader(name, **kwargs):
            seen["name"] = name
            seen.update(kwargs)
            return iter([example()])

        monkeypatch.setattr(nq, "load_dataset", loader)
        mod.fetch_split(
            "train", tmp_path / "m.json", 0, dataset="vendor/ds", revision="abc123"
        )
        assert seen["name"] == "vendor/ds"
        assert seen["revision"] == "abc123"
        assert seen["streaming"] is True

    def test_defaults_to_the_configured_dataset(self, monkeypatch, tmp_path):
        seen = {}

        def loader(name, **kwargs):
            seen["name"] = name
            seen.update(kwargs)
            return iter([example()])

        monkeypatch.setattr(nq, "load_dataset", loader)
        mod.fetch_split("train", tmp_path / "m.json", 0)
        assert seen["name"] == mod.DEFAULT.dataset
        assert seen["revision"] == mod.DEFAULT.dataset_revision

    def test_records_the_dataset_and_revision_in_the_output(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(nq, "load_dataset", lambda *_a, **_k: iter([example()]))
        out = tmp_path / "matched_train.json"
        mod.fetch_split("train", out, 0, dataset="vendor/ds", revision="abc123")
        data = json.loads(out.read_text())
        assert data["dataset"] == "vendor/ds"
        assert data["dataset_revision"] == "abc123"

    def test_reports_an_unpinned_revision(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(nq, "load_dataset", lambda *_a, **_k: iter([example()]))
        mod.fetch_split("train", tmp_path / "m.json", 0, revision=None)
        assert "unpinned" in capsys.readouterr().out

    def test_reports_the_pinned_revision_by_default(
        self, monkeypatch, tmp_path, capsys
    ):
        monkeypatch.setattr(nq, "load_dataset", lambda *_a, **_k: iter([example()]))
        mod.fetch_split("train", tmp_path / "m.json", 0)
        assert mod.DEFAULT.dataset_revision in capsys.readouterr().out

    def test_records_the_split_name(self, monkeypatch, tmp_path):
        assert self._run(monkeypatch, [example()], tmp_path)["split"] == "train"

    def test_stops_at_the_scan_limit(self, monkeypatch, tmp_path):
        examples = [example(ex_id=str(i)) for i in range(10)]
        data = self._run(monkeypatch, examples, tmp_path, n_limit=4)
        assert data["n_matched"] == 4

    def test_a_zero_limit_scans_everything(self, monkeypatch, tmp_path):
        examples = [example(ex_id=str(i)) for i in range(6)]
        assert self._run(monkeypatch, examples, tmp_path, n_limit=0)["n_matched"] == 6

    def test_creates_the_output_directory(self, monkeypatch, tmp_path):
        monkeypatch.setattr(nq, "load_dataset", lambda *_a, **_k: iter([example()]))
        out = tmp_path / "nested" / "dir" / "matched_train.json"
        mod.fetch_split("train", out, 0)
        assert out.exists()

    def test_writes_an_empty_record_list_when_nothing_matches(
        self, monkeypatch, tmp_path
    ):
        data = self._run(monkeypatch, [example(html="<p>none</p>")], tmp_path)
        assert data["n_matched"] == 0
        assert data["records"] == []

    def test_reports_progress_on_the_thousandth_example(
        self, monkeypatch, tmp_path, capsys
    ):
        examples = [example(html="<p>none</p>") for _ in range(1000)]
        self._run(monkeypatch, examples, tmp_path)
        assert "scanned 1,000" in capsys.readouterr().out


class TestMain:
    def test_builds_each_requested_split(self, monkeypatch, tmp_path):
        monkeypatch.setattr(nq, "load_dataset", lambda *_a, **_k: iter([example()]))
        monkeypatch.setattr(
            "sys.argv",
            ["prog", "--splits", "train", "validation", "--out-dir", str(tmp_path)],
        )
        mod.main()
        assert (tmp_path / "matched_train.json").exists()
        assert (tmp_path / "matched_validation.json").exists()

    def test_skips_a_split_whose_cache_exists(self, monkeypatch, tmp_path, capsys):
        existing = tmp_path / "matched_train.json"
        existing.write_text("{}")

        def fail(*_args, **_kwargs):
            raise AssertionError("load_dataset must not be called")

        monkeypatch.setattr(nq, "load_dataset", fail)
        monkeypatch.setattr(
            "sys.argv", ["prog", "--splits", "train", "--out-dir", str(tmp_path)]
        )
        mod.main()
        assert "Skipping train" in capsys.readouterr().out
        assert existing.read_text() == "{}"
