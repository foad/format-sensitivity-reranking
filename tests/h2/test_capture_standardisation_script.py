"""Tests for scripts.h2.capture_standardisation."""

from __future__ import annotations

import json

import numpy as np
import pytest
from scripts.h2 import capture_standardisation as mod

from fsr.formats import FORMAT_NAMES
from fsr.h2_layout import adapter_dir
from fsr.reporting import parse_progress
from fsr.standardisation.layout import candidates_path, tensor_meta_path, tensor_path
from fsr.standardisation.store import load_tensor

SLUG = "minilm_l6"
Q, C = 3, 4


@pytest.fixture(autouse=True)
def stub_model(monkeypatch):
    """Keep the capture away from the network and the GPU."""
    monkeypatch.setattr(mod, "load_tokenizer", lambda *_a, **_k: object())
    monkeypatch.setattr(mod, "load_model", lambda *_a, **_k: object())
    monkeypatch.setattr(
        "fsr.standardisation.scoring.score_batch",
        lambda _m, _t, pairs, *_a, **_k: [1.0] * len(pairs),
    )


@pytest.fixture
def data_root(tmp_path):
    root = tmp_path / "nq"
    path = candidates_path(root, "test")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "prepared": [
                    {
                        "id": f"q{q}",
                        "question": f"question {q}",
                        "candidates": [
                            {"pairs": [["title", f"t{c}"]], "truncated_body": "body"}
                            for c in range(C)
                        ],
                    }
                    for q in range(Q)
                ]
            }
        )
    )
    return root


def run(monkeypatch, data_root, *extra):
    monkeypatch.setattr(
        "sys.argv",
        [
            "capture_standardisation.py",
            "--model",
            SLUG,
            "--data-root",
            str(data_root),
            *extra,
        ],
    )
    mod.main()


class TestRun:
    def test_writes_one_tensor_per_condition(self, monkeypatch, data_root):
        run(monkeypatch, data_root)
        read = load_tensor(data_root, "test", SLUG, "base")
        assert read.scores.shape == (Q, C, len(FORMAT_NAMES))

    def test_keeps_the_query_order_of_the_candidate_lists(self, monkeypatch, data_root):
        run(monkeypatch, data_root)
        assert load_tensor(data_root, "test", SLUG, "base").query_ids == [
            "q0",
            "q1",
            "q2",
        ]

    def test_describes_the_condition(self, monkeypatch, data_root):
        run(monkeypatch, data_root)
        meta = json.loads(tensor_meta_path(data_root, "test", SLUG, "base").read_text())
        assert meta["model"] == SLUG
        assert meta["arm"] == "base"
        assert meta["adapter"] is None
        assert meta["formats"] == list(FORMAT_NAMES)

    def test_counts_every_format(self, monkeypatch, data_root, tmp_path):
        path = tmp_path / "p.progress"
        run(monkeypatch, data_root, "--progress-file", str(path))
        state = parse_progress(path.read_text())
        assert state["total"] == str(len(FORMAT_NAMES))
        assert state["done"] == str(len(FORMAT_NAMES))


class TestArms:
    def test_an_arm_scores_with_its_adapter(self, monkeypatch, data_root):
        seen = {}
        monkeypatch.setattr(
            mod,
            "load_model",
            lambda *_a, **kw: seen.update(kw) or object(),
        )
        adapter = adapter_dir(data_root, SLUG, "5fmt_lam1")
        adapter.mkdir(parents=True)
        run(monkeypatch, data_root, "--arm", "5fmt_lam1")
        assert seen["lora_adapter_path"] == str(adapter)

    def test_refuses_an_arm_with_no_adapter(self, monkeypatch, data_root):
        with pytest.raises(SystemExit, match="no adapter at"):
            run(monkeypatch, data_root, "--arm", "5fmt_lam1")

    def test_the_untrained_model_loads_no_adapter(self, monkeypatch, data_root):
        seen = {}
        monkeypatch.setattr(
            mod, "load_model", lambda *_a, **kw: seen.update(kw) or object()
        )
        run(monkeypatch, data_root)
        assert seen["lora_adapter_path"] is None


class TestMissingCandidates:
    def test_refuses_when_the_lists_were_not_built(self, monkeypatch, tmp_path):
        with pytest.raises(SystemExit, match="no candidate lists"):
            run(monkeypatch, tmp_path / "nq")


class TestSkipAndForce:
    def test_skips_a_present_capture(self, monkeypatch, data_root, capsys):
        run(monkeypatch, data_root)
        run(monkeypatch, data_root)
        assert "is present" in capsys.readouterr().out

    def test_force_captures_again(self, monkeypatch, data_root, capsys):
        run(monkeypatch, data_root)
        run(monkeypatch, data_root, "--force")
        assert "is present" not in capsys.readouterr().out

    def test_a_skip_leaves_the_tensor_alone(self, monkeypatch, data_root):
        run(monkeypatch, data_root)
        before = np.load(tensor_path(data_root, "test", SLUG, "base"))
        run(monkeypatch, data_root)
        after = np.load(tensor_path(data_root, "test", SLUG, "base"))
        assert np.array_equal(after, before)
