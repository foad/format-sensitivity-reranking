"""Tests for fsr.demo.loop."""

from __future__ import annotations

import json

import numpy as np
import pytest

from fsr.demo import loop as mod
from fsr.formats import FORMAT_NAMES, FORMATS

ARMS = ["base", "5fmt_lam0"]


def candidate(marker, n_pairs=1):
    return {
        "pairs": [[f"k{i}", marker] for i in range(n_pairs)],
        "truncated_body": f"{marker} body",
    }


def pool(pid, question, markers):
    return {
        "id": pid,
        "question": question,
        "candidates": [candidate(m) for m in markers],
    }


def push(inbox, name, pools, ready=True):
    inbox.mkdir(parents=True, exist_ok=True)
    (inbox / f"{name}.json").write_text(json.dumps({"pools": pools}))
    if ready:
        (inbox / f"{name}.ready").write_text("")


class FakeScorer:
    """Scores a pair by its text length, offset per arm, and counts the calls."""

    def __init__(self):
        """Start with no calls."""
        self.calls = []

    def __call__(self, arm, pairs):
        self.calls.append((arm, len(pairs)))
        offset = ARMS.index(arm) if arm in ARMS else 0
        return [len(q) + len(t) + offset for q, t in pairs]


@pytest.fixture
def boxes(tmp_path):
    inbox, outbox = tmp_path / "inbox", tmp_path / "outbox"
    outbox.mkdir()
    return inbox, outbox


class TestReadyJobs:
    def test_needs_the_marker(self, boxes):
        inbox, _ = boxes
        push(inbox, "r1", [pool("a", "q", ["x"])])
        push(inbox, "r2", [pool("b", "q", ["x"])], ready=False)
        assert mod.ready_jobs(inbox) == ["r1"]

    def test_needs_the_job_file(self, boxes):
        inbox, _ = boxes
        inbox.mkdir()
        (inbox / "r1.ready").write_text("")
        assert mod.ready_jobs(inbox) == []

    def test_keeps_dotted_names(self, boxes):
        inbox, _ = boxes
        push(inbox, "r1.part0", [pool("a", "q", ["x"])])
        assert mod.ready_jobs(inbox) == ["r1.part0"]

    def test_absent_inbox(self, tmp_path):
        assert mod.ready_jobs(tmp_path / "none") == []


class TestClaims:
    def test_only_one_worker_claims(self, boxes):
        _, outbox = boxes
        assert mod.claim(outbox, "r1", "w0")
        assert not mod.claim(outbox, "r1", "w1")
        assert mod.claim_owner(outbox, "r1") == "w0"

    def test_workers_split_the_jobs(self, boxes):
        inbox, outbox = boxes
        push(inbox, "r1", [pool("a", "q", ["x"])])
        push(inbox, "r2", [pool("b", "q", ["x"])])
        assert mod.next_job(inbox, outbox, "w0") == "r1"
        assert mod.next_job(inbox, outbox, "w1") == "r2"

    def test_a_restarted_worker_resumes_its_claim(self, boxes):
        inbox, outbox = boxes
        push(inbox, "r1", [pool("a", "q", ["x"])])
        push(inbox, "r2", [pool("b", "q", ["x"])])
        mod.claim(outbox, "r2", "w0")
        assert mod.next_job(inbox, outbox, "w0") == "r2"

    def test_skips_finished_jobs(self, boxes):
        inbox, outbox = boxes
        push(inbox, "r1", [pool("a", "q", ["x"])])
        mod.claim(outbox, "r1", "w0")
        (outbox / "r1.done").write_text("")
        assert mod.next_job(inbox, outbox, "w0") is None

    def test_never_writes_to_the_inbox(self, boxes):
        inbox, outbox = boxes
        push(inbox, "r1", [pool("a", "q", ["x"])])
        before = sorted(p.name for p in inbox.iterdir())
        name = mod.next_job(inbox, outbox, "w0")
        mod.run_job(
            inbox, outbox, name, "w0", ARMS, FakeScorer(), mod.ScoreCache(outbox, "w0")
        )
        assert sorted(p.name for p in inbox.iterdir()) == before


class TestRenderPairs:
    def test_one_entry_per_candidate_and_format(self):
        pools = [pool("a", "q", ["x", "y"])]
        pairs, index = mod.render_pairs(pools)
        assert index.shape == (1, 2, len(FORMAT_NAMES))
        f = FORMAT_NAMES.index("json")
        assert pairs[index[0, 1, f]] == ("q", FORMATS["json"](*candidate("y").values()))

    def test_repeated_passages_render_once(self):
        pools = [pool("a", "q", ["x"]), pool("b", "q", ["x"])]
        pairs, index = mod.render_pairs(pools)
        assert len(pairs) == len(set(pairs))
        assert (index[0] == index[1]).all()

    def test_pads_smaller_pools(self):
        _, index = mod.render_pairs([pool("a", "q", ["x", "y"]), pool("b", "q", ["x"])])
        assert (index[1, 1] == -1).all()
        assert (index[1, 0] >= 0).all()


class TestScorePools:
    def test_scores_each_arm(self, boxes):
        _, outbox = boxes
        pools = [pool("a", "q", ["x", "yy"])]
        scores, scored = mod.score_pools(
            pools, ARMS, FakeScorer(), mod.ScoreCache(outbox, "w0")
        )
        assert set(scores) == set(ARMS)
        assert scores["base"].shape == (1, 2, len(FORMAT_NAMES))
        np.testing.assert_allclose(scores["5fmt_lam0"], scores["base"] + 1)
        assert scored == 2 * 2 * len(FORMAT_NAMES)

    def test_marks_missing_candidates_nan(self, boxes):
        _, outbox = boxes
        pools = [pool("a", "q", ["x", "y"]), pool("b", "q", ["x"])]
        scores, _ = mod.score_pools(
            pools, ARMS, FakeScorer(), mod.ScoreCache(outbox, "w0")
        )
        assert np.isnan(scores["base"][1, 1]).all()
        assert not np.isnan(scores["base"][1, 0]).any()

    def test_rescores_only_changed_pairs(self, boxes):
        _, outbox = boxes
        scorer = FakeScorer()
        mod.score_pools(
            [pool("a", "q", ["x", "y"])], ARMS, scorer, mod.ScoreCache(outbox, "w0")
        )
        scorer.calls.clear()
        mod.score_pools(
            [pool("a", "q", ["x", "z"])], ARMS, scorer, mod.ScoreCache(outbox, "w0")
        )
        assert scorer.calls == [(arm, len(FORMAT_NAMES)) for arm in ARMS]

    def test_reads_other_workers_scores(self, boxes):
        _, outbox = boxes
        scorer = FakeScorer()
        mod.score_pools(
            [pool("a", "q", ["x"])], ARMS, scorer, mod.ScoreCache(outbox, "w0")
        )
        scorer.calls.clear()
        _, scored = mod.score_pools(
            [pool("a", "q", ["x"])], ARMS, scorer, mod.ScoreCache(outbox, "w1")
        )
        assert scored == 0
        assert scorer.calls == []

    def test_each_worker_appends_to_its_own_file(self, boxes):
        _, outbox = boxes
        mod.score_pools(
            [pool("a", "q", ["x"])], ARMS, FakeScorer(), mod.ScoreCache(outbox, "w0")
        )
        mod.score_pools(
            [pool("a", "q", ["y"])], ARMS, FakeScorer(), mod.ScoreCache(outbox, "w1")
        )
        names = sorted(p.name for p in (outbox / mod.CACHE_SUBDIR).iterdir())
        assert names == sorted(f"{a}.{w}.jsonl" for a in ARMS for w in ("w0", "w1"))


class TestRunJob:
    def test_writes_scores_description_and_marker(self, boxes):
        inbox, outbox = boxes
        pools = [pool("a", "q", ["x"]), pool("b", "q2", ["x", "LONG"])]
        push(inbox, "r1", pools)
        summary = mod.run_job(
            inbox,
            outbox,
            "r1",
            "w0",
            ARMS,
            FakeScorer(),
            mod.ScoreCache(outbox, "w0"),
            too_long=lambda _q, t: "LONG" in t,
        )
        assert not summary["failed"]
        assert (outbox / "r1.done").exists()
        meta = json.loads((outbox / "r1.json").read_text())
        assert meta["pool_ids"] == ["a", "b"]
        assert meta["arms"] == ARMS
        assert meta["truncated_pairs"] == len(FORMAT_NAMES)
        with np.load(outbox / "r1.npz") as saved:
            assert saved["scores_base"].shape == (2, 2, len(FORMAT_NAMES))

    def test_leaves_no_temporary_files(self, boxes):
        inbox, outbox = boxes
        push(inbox, "r1", [pool("a", "q", ["x"])])
        mod.run_job(
            inbox, outbox, "r1", "w0", ARMS, FakeScorer(), mod.ScoreCache(outbox, "w0")
        )
        assert not [p for p in outbox.iterdir() if p.name.startswith(".")]

    def test_marks_a_bad_job_failed(self, boxes):
        inbox, outbox = boxes
        push(inbox, "r1", [{"id": "a", "question": "q", "candidates": []}])
        summary = mod.run_job(
            inbox, outbox, "r1", "w0", ARMS, FakeScorer(), mod.ScoreCache(outbox, "w0")
        )
        assert summary["failed"]
        assert (
            "no candidates" in json.loads((outbox / "r1.failed").read_text())["error"]
        )
        assert not (outbox / "r1.done").exists()
        assert mod.next_job(inbox, outbox, "w0") is None


class Clock:
    """A clock that moves only when slept on."""

    def __init__(self):
        """Start at zero."""
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class TestServe:
    def test_scores_every_job_then_stops_idle(self, boxes):
        inbox, outbox = boxes
        push(inbox, "r1", [pool("a", "q", ["x"])])
        push(inbox, "r2", [pool("b", "q", ["y"])])
        clock, seen = Clock(), []
        reason = mod.serve(
            inbox,
            outbox,
            "w0",
            ARMS,
            FakeScorer(),
            poll=5,
            idle=60,
            wall=3600,
            on_job=seen.append,
            clock=clock,
            sleep=clock.sleep,
        )
        assert reason == "idle"
        assert [s["job"] for s in seen] == ["r1", "r2"]
        assert 60 <= clock.now < 70

    def test_takes_a_job_that_arrives_later(self, boxes):
        inbox, outbox = boxes
        clock, seen = Clock(), []

        def sleep(seconds):
            clock.sleep(seconds)
            if clock.now == 20:
                push(inbox, "late", [pool("a", "q", ["x"])])

        mod.serve(
            inbox,
            outbox,
            "w0",
            ARMS,
            FakeScorer(),
            poll=5,
            idle=60,
            wall=3600,
            on_job=seen.append,
            clock=clock,
            sleep=sleep,
        )
        assert [s["job"] for s in seen] == ["late"]

    def test_stops_at_the_wall_clock_cap(self, boxes):
        inbox, outbox = boxes
        clock = Clock()

        def sleep(seconds):
            clock.sleep(seconds)
            push(inbox, f"r{int(clock.now)}", [pool("a", "q", ["x"])])

        reason = mod.serve(
            inbox,
            outbox,
            "w0",
            ARMS,
            FakeScorer(),
            poll=5,
            idle=60,
            wall=30,
            clock=clock,
            sleep=sleep,
        )
        assert reason == "wall"

    def test_announces_each_job(self, boxes):
        inbox, outbox = boxes
        push(inbox, "r1", [pool("a", "q", ["x"])])
        clock, started = Clock(), []
        mod.serve(
            inbox,
            outbox,
            "w0",
            ARMS,
            FakeScorer(),
            poll=5,
            idle=10,
            wall=3600,
            on_start=started.append,
            clock=clock,
            sleep=clock.sleep,
        )
        assert started == ["r1"]
