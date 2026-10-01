"""An offline integration run against a small synthetic corpus."""

from __future__ import annotations

import json
import random
import sys

import pytest
from scripts.corpus import build_corpus
from scripts.corpus import fetch as fetch_script
from scripts.corpus import negatives as negatives_script
from scripts.corpus import parse as parse_script
from scripts.corpus import split as split_script
from scripts.corpus import validate as validate_script

from fsr.corpus import nq
from fsr.corpus.cli import MANIFEST_NAME, SPLIT_SUBDIR
from fsr.corpus.files import digest_file
from fsr.corpus.run_record import RUN_NAME

MODULES = {
    "fetch.py": fetch_script,
    "parse.py": parse_script,
    "split.py": split_script,
    "negatives.py": negatives_script,
    "validate.py": validate_script,
}
CACHE_K = 3
N_TRAIN = 24
N_VALIDATION = 6
KEYS = ("Born", "Died", "Role", "Known for", "Spouse")
WORDS = [
    "bridge",
    "river",
    "tower",
    "city",
    "engineer",
    "architect",
    "steel",
    "stone",
    "opened",
    "span",
]


def raw_record(idx, rng):
    """Return one cache record that passes the quality gate."""
    values = [str(1800 + rng.randint(0, 200)) for _ in KEYS]
    rows = "".join(
        f"<tr><th>{key}</th><td>{value} {rng.choice(WORDS)}</td></tr>"
        for key, value in zip(KEYS, values, strict=True)
    )
    body = "".join(
        "<p>" + " ".join(rng.choice(WORDS) for _ in range(40)).capitalize() + ".</p>"
        for _ in range(3)
    )
    return {
        "id": str(idx),
        "title": f"Article {idx}",
        "question": f"who built the {rng.choice(WORDS)} of article {idx}",
        "infobox_html_raw": f'<table class="infobox">{rows}</table>',
        "post_infobox_html_raw": body,
        "short_answers": [rng.choice(values)],
    }


@pytest.fixture
def data_root(tmp_path):
    """Write a raw cache for both splits and return the data root."""
    rng = random.Random(11)
    root = tmp_path / "data" / "nq"
    root.mkdir(parents=True)
    for split, count, offset in (
        ("train", N_TRAIN, 0),
        ("validation", N_VALIDATION, 1000),
    ):
        records = [raw_record(offset + i, rng) for i in range(count)]
        (root / f"matched_{split}.json").write_text(
            json.dumps({"split": split, "records": records})
        )
    return root


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    """Stop the build from reaching Natural Questions over the network."""

    def no_network(*_args, **_kwargs):
        raise AssertionError("the build must not stream the dataset")

    monkeypatch.setattr(nq, "load_dataset", no_network)


@pytest.fixture(autouse=True)
def _clean_tree(monkeypatch):
    """Report a fixed repository state, so the record does not vary."""
    monkeypatch.setattr(
        "fsr.corpus.run_record.git_state",
        lambda _p: {"revision": "0" * 40, "dirty": False},
    )


def in_process(stage):
    """Run one stage in this process and return its exit code."""
    argv = sys.argv
    sys.argv = ["prog", *stage.args]
    try:
        MODULES[stage.script].main()
        return 0
    except SystemExit as exit_request:
        return exit_request.code if isinstance(exit_request.code, int) else 1
    finally:
        sys.argv = argv


def build(monkeypatch, data_root, *extra):
    """Run the orchestrator over the data root and return its stage names."""
    ran = []

    def run_stage(stage):
        ran.append(stage.name)
        return in_process(stage)

    monkeypatch.setattr(build_corpus, "run_stage", run_stage)
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--data-root",
            str(data_root),
            "--fetch",
            "--cache-k",
            str(CACHE_K),
            *extra,
        ],
    )
    build_corpus.main()
    return ran


def digests(data_root):
    """Return the digest of every file under the data root, keyed by name."""
    return {
        str(p.relative_to(data_root)): digest_file(p)
        for p in sorted(data_root.rglob("*"))
        if p.is_file()
    }


class TestFullBuild:
    def test_every_stage_runs(self, monkeypatch, data_root):
        assert build(monkeypatch, data_root) == [
            "fetch",
            "parse",
            "split",
            "negatives",
            "validate",
        ]

    def test_writes_the_parsed_corpus(self, monkeypatch, data_root):
        build(monkeypatch, data_root)
        parsed = json.loads((data_root / "parsed_train.json").read_text())
        assert parsed["n_records"] == N_TRAIN
        assert parsed["n_passed_quality"] == N_TRAIN
        assert parsed["strict_gate_active"] is True
        assert "short_answers" in parsed["records"][0]

    def test_writes_every_split(self, monkeypatch, data_root):
        build(monkeypatch, data_root)
        split_dir = data_root / SPLIT_SUBDIR
        counts = {
            name: json.loads((split_dir / f"{name}.json").read_text())["n_records"]
            for name in ("train", "dev", "test", "nq_val")
        }
        assert counts["train"] + counts["dev"] + counts["test"] == N_TRAIN
        assert counts["nq_val"] == N_VALIDATION

    def test_writes_the_negatives(self, monkeypatch, data_root):
        build(monkeypatch, data_root)
        payload = json.loads(
            (data_root / SPLIT_SUBDIR / "bm25_negatives.json").read_text()
        )
        assert payload["cache_k"] == CACHE_K
        assert payload["n_queries"] == N_TRAIN + N_VALIDATION
        assert all(len(picked) == CACHE_K for picked in payload["negatives"].values())

    def test_records_every_stage_in_the_manifest(self, monkeypatch, data_root):
        build(monkeypatch, data_root)
        manifest = json.loads((data_root / MANIFEST_NAME).read_text())
        assert [s["name"] for s in manifest["stages"]] == [
            "parse_train",
            "parse_validation",
            "split",
            "negatives",
        ]
        assert manifest["config"]["cache_k"] == CACHE_K

    def test_records_the_run(self, monkeypatch, data_root):
        build(monkeypatch, data_root)
        record = json.loads((data_root / RUN_NAME).read_text())
        assert [s["name"] for s in record["stages"]] == [
            "fetch",
            "parse",
            "split",
            "negatives",
            "validate",
        ]
        assert all(s["exit_code"] == 0 for s in record["stages"])
        assert record["finished"] is not None
        assert record["git"] == {"revision": "0" * 40, "dirty": False}


class TestRepeatedBuild:
    def test_a_second_build_changes_nothing(self, monkeypatch, data_root):
        build(monkeypatch, data_root)
        before = digests(data_root)
        build(monkeypatch, data_root)
        after = digests(data_root)
        assert {k: v for k, v in after.items() if k != RUN_NAME} == {
            k: v for k, v in before.items() if k != RUN_NAME
        }

    def test_a_forced_rebuild_reproduces_every_file(self, monkeypatch, data_root):
        build(monkeypatch, data_root)
        before = digests(data_root)
        build(monkeypatch, data_root, "--force")
        after = digests(data_root)
        assert {k: v for k, v in after.items() if k != RUN_NAME} == {
            k: v for k, v in before.items() if k != RUN_NAME
        }

    def test_an_interrupted_build_continues(self, monkeypatch, data_root):
        build(monkeypatch, data_root)
        (data_root / SPLIT_SUBDIR / "bm25_negatives.json").unlink()
        assert build(monkeypatch, data_root) == [
            "fetch",
            "parse",
            "split",
            "negatives",
            "validate",
        ]
        assert (data_root / SPLIT_SUBDIR / "bm25_negatives.json").exists()


class TestFailedBuild:
    def test_a_failing_stage_stops_the_build(self, monkeypatch, data_root):
        with pytest.raises(SystemExit, match="negatives failed with exit code 1"):
            build(monkeypatch, data_root, "--cache-k", "500")

    def test_the_record_holds_the_failure(self, monkeypatch, data_root):
        with pytest.raises(SystemExit):
            build(monkeypatch, data_root, "--cache-k", "500")
        record = json.loads((data_root / RUN_NAME).read_text())
        assert [(s["name"], s["exit_code"]) for s in record["stages"]] == [
            ("fetch", 0),
            ("parse", 0),
            ("split", 0),
            ("negatives", 1),
        ]
        assert record["finished"] is None

    def test_validate_catches_a_corrupt_corpus(self, monkeypatch, data_root):
        build(monkeypatch, data_root)
        meta = data_root / SPLIT_SUBDIR / "meta.json"
        meta.write_text(json.dumps({"n_records_train": 9999}))
        with pytest.raises(SystemExit, match="validate failed"):
            build(monkeypatch, data_root)
