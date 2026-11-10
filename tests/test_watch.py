from __future__ import annotations

import io

import pytest
from scripts import watch as mod


def age_file(path, seconds):
    """Backdate a progress file, as if the job wrote it that long ago."""
    import os

    written = os.stat(path).st_mtime - seconds
    os.utime(path, (written, written))
    return path


def state(
    directory,
    name,
    done=1,
    total=10,
    label="with_body/yaml",
    elapsed=60.0,
    eta=30.0,
    failed=0,
    text=None,
):
    """Write one job's progress file and return its path."""
    path = directory / f"{name}.progress"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        text
        if text is not None
        else (
            f"done={done} total={total} elapsed={elapsed} "
            f"eta={eta} failed={failed} label={label}\n"
        )
    )
    return path


class TestReadState:
    def test_reports_the_state(self, tmp_path):
        s = mod.read_state(state(tmp_path, "test_cross_bge_base", done=4, total=10))
        assert (s.done, s.total) == (4, 10)

    def test_names_the_job_after_the_file(self, tmp_path):
        assert mod.read_state(state(tmp_path, "test_cross_jina_v2")).name == (
            "test_cross_jina_v2"
        )

    def test_carries_the_label_and_the_times(self, tmp_path):
        s = mod.read_state(
            state(tmp_path, "j", label="metadata_only/toml", elapsed=10.0, eta=5.0)
        )
        assert s.label == "metadata_only/toml"
        assert (s.elapsed, s.eta) == (10.0, 5.0)

    def test_reports_no_units_before_the_first(self, tmp_path):
        s = mod.read_state(state(tmp_path, "j", done=0, total=0, label=""))
        assert (s.done, s.total) == (0, 0)
        assert not s.finished

    def test_reports_no_units_for_an_empty_file(self, tmp_path):
        assert mod.read_state(state(tmp_path, "j", text="")).total == 0

    def test_survives_a_part_written_file(self, tmp_path):
        assert mod.read_state(state(tmp_path, "j", text="done=3 total=")).total == 0

    def test_survives_a_file_of_nonsense(self, tmp_path):
        assert mod.read_state(state(tmp_path, "j", text="done=x total=y")).total == 0

    def test_reports_a_failure(self, tmp_path):
        assert mod.read_state(state(tmp_path, "j", failed=1)).failed

    def test_survives_a_file_it_cannot_read(self, tmp_path):
        assert mod.read_state(tmp_path / "absent.progress").total == 0

    def test_reports_a_finished_job(self, tmp_path):
        assert mod.read_state(state(tmp_path, "j", done=10, total=10)).finished

    def test_a_job_past_its_total_is_finished(self, tmp_path):
        assert mod.read_state(state(tmp_path, "j", done=11, total=10)).finished


class TestLiveEstimates:
    def test_counts_the_time_since_the_job_wrote(self, tmp_path):
        path = age_file(state(tmp_path, "a", elapsed=60.0, eta=300.0), 120)
        s = mod.read_state(path)
        assert s.elapsed == 60.0
        assert s.eta == 300.0
        assert s.live_elapsed == pytest.approx(180.0, abs=2)
        assert s.live_eta == pytest.approx(180.0, abs=2)

    def test_a_fresh_file_reads_as_written(self, tmp_path):
        s = mod.read_state(state(tmp_path, "a", elapsed=60.0, eta=300.0))
        assert s.live_elapsed == pytest.approx(60.0, abs=2)
        assert s.live_eta == pytest.approx(300.0, abs=2)

    def test_the_estimate_stops_at_zero(self, tmp_path):
        path = age_file(state(tmp_path, "a", elapsed=60.0, eta=30.0), 600)
        assert mod.read_state(path).live_eta == 0.0

    def test_the_time_run_keeps_rising_past_the_estimate(self, tmp_path):
        path = age_file(state(tmp_path, "a", elapsed=60.0, eta=30.0), 600)
        assert mod.read_state(path).live_elapsed == pytest.approx(660.0, abs=2)

    def test_a_new_line_resets_the_estimate(self, tmp_path):
        path = age_file(state(tmp_path, "a", done=1, eta=300.0), 120)
        assert mod.read_state(path).live_eta == pytest.approx(180.0, abs=2)
        state(tmp_path, "a", done=2, eta=280.0)
        assert mod.read_state(path).live_eta == pytest.approx(280.0, abs=2)

    def test_the_line_counts_down_between_writes(self, tmp_path):
        path = age_file(state(tmp_path, "a", done=1, eta=300.0), 120)
        s = mod.read_state(path)
        line = mod.render([s])[0]
        assert f"eta {mod.format_duration(s.live_eta)}" in line
        assert mod.format_duration(s.eta) not in line

    def test_a_spent_estimate_says_overdue(self, tmp_path):
        path = age_file(state(tmp_path, "a", done=1, eta=30.0), 600)
        line = mod.render([mod.read_state(path)])[0]
        assert "eta overdue" in line
        assert "0:11:00" in line

    def test_a_finished_job_is_not_extrapolated(self, tmp_path):
        path = age_file(state(tmp_path, "a", done=10, total=10, elapsed=90.0), 600)
        assert "done in 0:01:30" in mod.render([mod.read_state(path)])[0]

    def test_the_run_line_counts_down_too(self, tmp_path):
        path = age_file(state(tmp_path, "a", done=1, eta=600.0), 120)
        s = mod.read_state(path)
        assert f"eta {mod.format_duration(s.live_eta)}" in mod.render([s])[-1]
        assert s.live_eta < s.eta


class TestByStart:
    def started(self, tmp_path, name, ago, elapsed=0.0, **over):
        """Write a job that began `ago` seconds back."""
        path = state(tmp_path, name, elapsed=elapsed, **over)
        return age_file(path, ago - elapsed)

    def test_orders_by_when_each_job_began(self, tmp_path):
        self.started(tmp_path, "late", ago=100)
        self.started(tmp_path, "early", ago=900)
        states = sorted(
            (mod.read_state(p) for p in tmp_path.glob("*.progress")), key=mod.by_start
        )
        assert [s.name for s in states] == ["early", "late"]

    def test_ignores_the_name(self, tmp_path):
        self.started(tmp_path, "aaa", ago=100)
        self.started(tmp_path, "zzz", ago=900)
        states = sorted(
            (mod.read_state(p) for p in tmp_path.glob("*.progress")), key=mod.by_start
        )
        assert [s.name for s in states] == ["zzz", "aaa"]

    def test_counts_work_already_done_before_the_last_write(self, tmp_path):
        # Both files were written a moment ago, but one job has run longer.
        self.started(tmp_path, "short", ago=60, elapsed=60.0)
        self.started(tmp_path, "long", ago=3600, elapsed=3600.0)
        states = sorted(
            (mod.read_state(p) for p in tmp_path.glob("*.progress")), key=mod.by_start
        )
        assert [s.name for s in states] == ["long", "short"]

    def test_a_finished_earlier_job_stays_above_a_running_later_one(self, tmp_path):
        self.started(tmp_path, "earlier", ago=3600, elapsed=900.0, done=10, total=10)
        self.started(tmp_path, "later", ago=600, elapsed=600.0, done=2)
        states = sorted(
            (mod.read_state(p) for p in tmp_path.glob("*.progress")), key=mod.by_start
        )
        assert [s.name for s in states] == ["earlier", "later"]

    def test_breaks_a_tie_by_name(self):
        blank = mod.JobState("b", 0, 0, "", 0.0, 0.0, False)
        other = mod.JobState("a", 0, 0, "", 0.0, 0.0, False)
        assert [s.name for s in sorted([blank, other], key=mod.by_start)] == ["a", "b"]

    def test_a_job_with_no_state_yet_sorts_last(self, tmp_path):
        self.started(tmp_path, "running", ago=600, elapsed=600.0)
        state(tmp_path, "pending", done=0, total=0, elapsed=0.0)
        states = sorted(
            (mod.read_state(p) for p in tmp_path.glob("*.progress")), key=mod.by_start
        )
        assert [s.name for s in states] == ["running", "pending"]

    def test_holds_for_names_of_another_phase(self, tmp_path):
        self.started(tmp_path, "bge_base_5fmt_lam0.1", ago=900, elapsed=900.0)
        self.started(tmp_path, "bge_base_yaml_lam0", ago=120, elapsed=120.0)
        states = sorted(
            (mod.read_state(p) for p in tmp_path.glob("*.progress")), key=mod.by_start
        )
        assert [s.name for s in states] == [
            "bge_base_5fmt_lam0.1",
            "bge_base_yaml_lam0",
        ]


class TestRenderBar:
    def test_is_empty_at_the_start(self):
        assert mod.render_bar(0, 10, width=4) == "[----]"

    def test_is_full_at_the_end(self):
        assert mod.render_bar(10, 10, width=4) == "[####]"

    def test_fills_in_proportion(self):
        assert mod.render_bar(5, 10, width=4) == "[##--]"

    def test_is_empty_with_no_total(self):
        assert mod.render_bar(0, 0, width=4) == "[----]"

    def test_keeps_the_width_past_the_total(self):
        assert mod.render_bar(99, 10, width=4) == "[####]"

    def test_uses_only_plain_characters(self):
        assert set(mod.render_bar(3, 10)) <= set("[]#-")


class TestRender:
    def test_reports_when_no_job_is_present(self):
        assert mod.render([]) == ["no job progress found"]

    def test_one_line_per_job_and_one_for_the_run(self, tmp_path):
        states = [
            mod.read_state(state(tmp_path, "a", done=1)),
            mod.read_state(state(tmp_path, "b", done=2)),
        ]
        assert len(mod.render(states)) == 3

    def test_a_running_job_shows_its_label_and_time_left(self, tmp_path):
        s = mod.read_state(state(tmp_path, "a", done=1, label="x/y", eta=300.0))
        line = mod.render([s])[0]
        assert "x/y" in line
        assert f"eta {mod.format_duration(s.live_eta)}" in line

    def test_a_finished_job_shows_how_long_it_took(self, tmp_path):
        s = mod.read_state(state(tmp_path, "a", done=10, total=10, elapsed=90))
        assert "done in 0:01:30" in mod.render([s])[0]

    def test_a_job_with_no_state_yet_is_waiting(self, tmp_path):
        s = mod.read_state(state(tmp_path, "a", done=0, total=0))
        assert "waiting" in mod.render([s])[0]

    def test_a_job_before_its_first_unit_is_starting(self, tmp_path):
        s = mod.read_state(state(tmp_path, "a", done=0, total=10, label=""))
        line = mod.render([s])[0]
        assert "starting" in line
        assert "eta" not in line

    def test_a_failed_job_says_so(self, tmp_path):
        s = mod.read_state(state(tmp_path, "a", failed=1))
        assert "FAILED" in mod.render([s])[0]

    def test_the_run_line_totals_the_units(self, tmp_path):
        states = [
            mod.read_state(state(tmp_path, "a", done=1)),
            mod.read_state(state(tmp_path, "b", done=3)),
        ]
        assert "2 jobs, 4/20 units" in mod.render(states)[-1]

    def test_the_run_line_takes_the_longest_time_left(self, tmp_path):
        states = [
            mod.read_state(state(tmp_path, "a", eta=30)),
            mod.read_state(state(tmp_path, "b", eta=600)),
        ]
        longest = max(s.live_eta for s in states)
        assert f"eta {mod.format_duration(longest)}" in mod.render(states)[-1]

    def test_a_job_that_cannot_be_read_has_no_age(self, tmp_path):
        assert mod.read_state(tmp_path / "absent.progress").age == 0.0

    def test_the_run_line_gives_no_time_left_when_all_are_done(self, tmp_path):
        s = mod.read_state(state(tmp_path, "a", done=10, total=10))
        assert "eta" not in mod.render([s])[-1]

    def test_names_are_padded_to_one_width(self, tmp_path):
        states = [
            mod.read_state(state(tmp_path, "short")),
            mod.read_state(state(tmp_path, "a_much_longer_name")),
        ]
        lines = mod.render(states)
        assert lines[0].index("[") == lines[1].index("[")


class TestDraw:
    def test_writes_each_line(self):
        out = io.StringIO()
        mod.draw(["a", "b"], out, redraw=0)
        assert out.getvalue() == "a\nb\n"

    def test_moves_the_cursor_up_to_redraw(self):
        out = io.StringIO()
        mod.draw(["a"], out, redraw=2)
        assert out.getvalue().startswith("\033[2A")

    def test_clears_each_line_on_a_redraw(self):
        out = io.StringIO()
        mod.draw(["a"], out, redraw=1)
        assert "\033[2K" in out.getvalue()


class TestWatch:
    def test_draws_once_and_stops(self, tmp_path):
        state(tmp_path, "a", done=1)
        out = io.StringIO()
        assert mod.watch(tmp_path, "*.progress", 0.0, True, out) == 0
        assert "1/10" in out.getvalue()

    def test_reports_a_failure_in_the_exit_code(self, tmp_path):
        state(tmp_path, "a", done=10, total=10, failed=1)
        out = io.StringIO()
        assert mod.watch(tmp_path, "*.progress", 0.0, True, out) == 1

    def test_keeps_watching_after_every_job_has_finished(self, tmp_path, monkeypatch):
        state(tmp_path, "a", done=10, total=10)
        rounds = []

        def later(seconds):
            rounds.append(seconds)
            if len(rounds) == 2:
                raise KeyboardInterrupt

        monkeypatch.setattr(mod.time, "sleep", later)
        with pytest.raises(KeyboardInterrupt):
            mod.watch(tmp_path, "*.progress", 5.0, False, io.StringIO())
        assert rounds == [5.0, 5.0]

    def test_picks_up_a_job_that_starts_later(self, tmp_path, monkeypatch):
        state(tmp_path, "test_cross_a", done=10, total=10)
        rounds = []

        def later(_seconds):
            rounds.append(1)
            if len(rounds) == 1:
                state(tmp_path, "test_within_a", done=1, total=10)
            else:
                raise KeyboardInterrupt

        monkeypatch.setattr(mod.time, "sleep", later)
        out = io.StringIO()
        with pytest.raises(KeyboardInterrupt):
            mod.watch(tmp_path, "*.progress", 0.0, False, out)
        assert "test_within_a" in out.getvalue()

    def test_sleeps_between_reads(self, tmp_path, monkeypatch):
        state(tmp_path, "a", done=1)
        sleeps = []

        def stop(seconds):
            sleeps.append(seconds)
            raise KeyboardInterrupt

        monkeypatch.setattr(mod.time, "sleep", stop)
        with pytest.raises(KeyboardInterrupt):
            mod.watch(tmp_path, "*.progress", 5.0, False, io.StringIO())
        assert sleeps == [5.0]

    def test_leaves_out_the_job_logs(self, tmp_path):
        state(tmp_path, "a", done=10, total=10)
        (tmp_path / "a.log").write_text("human output\n")
        (tmp_path / "a.hare.log").write_text("container output\n")
        out = io.StringIO()
        mod.watch(tmp_path, "*.progress", 0.0, True, out)
        assert "1 jobs" in out.getvalue()

    def test_lists_the_jobs_in_the_order_they_began(self, tmp_path):
        age_file(state(tmp_path, "aaa_started_late", done=1, elapsed=60.0), 0)
        age_file(state(tmp_path, "zzz_started_early", done=1, elapsed=3600.0), 0)
        out = io.StringIO()
        mod.watch(tmp_path, "*.progress", 0.0, True, out)
        lines = out.getvalue().splitlines()
        assert "zzz_started_early" in lines[0]
        assert "aaa_started_late" in lines[1]

    def test_honours_the_pattern(self, tmp_path):
        state(tmp_path, "test_cross_a", done=10, total=10)
        state(tmp_path, "test_within_a", done=10, total=10)
        out = io.StringIO()
        mod.watch(tmp_path, "test_cross_*.progress", 0.0, True, out)
        assert "test_within_a" not in out.getvalue()

    def test_reports_an_empty_directory(self, tmp_path):
        out = io.StringIO()
        assert mod.watch(tmp_path, "*.progress", 0.0, True, out) == 0
        assert "no job progress found" in out.getvalue()

    def test_waits_for_the_first_job_of_a_run(self, tmp_path, monkeypatch):
        rounds = []

        def later(_seconds):
            rounds.append(1)
            if len(rounds) == 1:
                state(tmp_path, "a", done=1, total=10)
            else:
                raise KeyboardInterrupt

        monkeypatch.setattr(mod.time, "sleep", later)
        out = io.StringIO()
        with pytest.raises(KeyboardInterrupt):
            mod.watch(tmp_path, "*.progress", 0.0, False, out)
        assert "1/10" in out.getvalue()


class TestBuildParser:
    def test_defaults_to_the_results_directory(self):
        args = mod.build_parser().parse_args([])
        assert args.results_subdir == mod.RESULTS_SUBDIR
        assert args.patterns is None
        assert mod.DEFAULT_PATTERN == "*.progress"
        assert args.max_age_hours is None
        assert args.interval == mod.DEFAULT_INTERVAL
        assert not args.once

    def test_takes_a_pattern_and_an_interval(self):
        args = mod.build_parser().parse_args(
            ["--pattern", "all_*.progress", "--interval", "2", "--once"]
        )
        assert args.patterns == ["all_*.progress"]
        assert args.interval == pytest.approx(2.0)
        assert args.once


class TestMain:
    def test_stopping_it_is_not_an_error(self, tmp_path, monkeypatch):
        state(tmp_path / "h1", "a", done=1)
        monkeypatch.setattr("sys.argv", ["prog", "--data-root", str(tmp_path)])

        def interrupt(_seconds):
            raise KeyboardInterrupt

        monkeypatch.setattr(mod.time, "sleep", interrupt)
        assert mod.main() == mod.INTERRUPTED == 130

    def test_watches_the_directory_the_arguments_name(self, tmp_path, monkeypatch):
        state(tmp_path / "h1", "a", done=10, total=10)
        monkeypatch.setattr(
            "sys.argv",
            ["prog", "--data-root", str(tmp_path), "--once"],
        )
        assert mod.main() == 0


class TestFindStates:
    def test_takes_several_patterns(self, tmp_path):
        state(tmp_path, "train_jina_v2_a")
        state(tmp_path, "train_mxbai_v1_a")
        state(tmp_path, "train_bge_base_a")
        names = {
            s.name for s in mod.find_states(tmp_path, ["*_jina_v2_*", "*_mxbai_v1_*"])
        }
        assert names == {"train_jina_v2_a", "train_mxbai_v1_a"}

    def test_lists_a_job_two_patterns_name_once(self, tmp_path):
        state(tmp_path, "train_jina_v2_a")
        states = mod.find_states(tmp_path, ["*jina*", "train_*"])
        assert [s.name for s in states] == ["train_jina_v2_a"]

    def test_takes_one_pattern_as_a_string(self, tmp_path):
        state(tmp_path, "a")
        assert len(mod.find_states(tmp_path, "*.progress")) == 1

    def test_reads_only_progress_files_whatever_the_pattern(self, tmp_path):
        """A pattern that names the job also names its logs."""
        state(tmp_path, "train_jina_v2_a")
        (tmp_path / "train_jina_v2_a.log").write_text("human output\n")
        (tmp_path / "train_jina_v2_a.hare.log").write_text("container output\n")
        states = mod.find_states(tmp_path, ["*_jina_v2_*"])
        assert [s.name for s in states] == ["train_jina_v2_a"]

    def test_leaves_out_a_job_older_than_the_limit(self, tmp_path):
        age_file(state(tmp_path, "old"), 7200)
        state(tmp_path, "new")
        states = mod.find_states(tmp_path, ["*"], max_age=3600)
        assert [s.name for s in states] == ["new"]

    def test_keeps_every_job_without_a_limit(self, tmp_path):
        age_file(state(tmp_path, "old"), 10**6)
        assert len(mod.find_states(tmp_path, ["*"])) == 1


class TestParser:
    def test_collects_each_pattern(self):
        args = mod.build_parser().parse_args(["--pattern", "a*", "--pattern", "b*"])
        assert args.patterns == ["a*", "b*"]

    def test_the_watcher_converts_hours_to_seconds(self, tmp_path, monkeypatch):
        seen = {}

        def fake_watch(_directory, patterns, _interval, _once, _out, max_age):
            seen.update(patterns=patterns, max_age=max_age)
            return 0

        monkeypatch.setattr(mod, "watch", fake_watch)
        monkeypatch.setattr(
            "sys.argv",
            ["watch.py", "--data-root", str(tmp_path), "--max-age-hours", "2"],
        )
        assert mod.main() == 0
        assert seen == {"patterns": [mod.DEFAULT_PATTERN], "max_age": 7200}


class TestWatchFilters:
    def test_hides_old_jobs_from_the_drawing(self, tmp_path):
        age_file(state(tmp_path, "finished_long_ago", done=10, total=10), 10**6)
        state(tmp_path, "running_now", done=1)
        out = io.StringIO()
        mod.watch(tmp_path, ["*"], 0.0, True, out, max_age=3600)
        assert "running_now" in out.getvalue()
        assert "finished_long_ago" not in out.getvalue()


SLUGS = ["bge_base", "bge_v2_m3", "mxbai_v1", "mxbai_v1_tanh", "minilm_l6"]


class TestSplitName:
    def test_finds_the_model_and_keeps_the_rest(self):
        assert mod.split_name("train_bge_v2_m3_only_yaml_lam0_s1", SLUGS) == (
            "bge_v2_m3",
            "train only_yaml_lam0_s1",
        )

    def test_finds_a_model_at_the_end(self):
        assert mod.split_name("test_cross_bge_base", SLUGS) == (
            "bge_base",
            "test_cross",
        )

    def test_prefers_the_longest_slug(self):
        """mxbai_v1 is a prefix of mxbai_v1_tanh."""
        assert mod.split_name("train_mxbai_v1_tanh_5fmt_lam0", SLUGS)[0] == (
            "mxbai_v1_tanh"
        )

    def test_needs_the_whole_slug(self):
        """bge_base must not match a name holding bge_base_x as one word."""
        assert mod.split_name("train_minilm_l62_a", SLUGS) == (
            None,
            "train_minilm_l62_a",
        )

    def test_leaves_a_name_with_no_model(self):
        assert mod.split_name("test_cross_a", SLUGS) == (None, "test_cross_a")

    def test_reads_the_registry_by_default(self):
        assert mod.split_name("train_jina_v2_only_json_lam0")[0] == "jina_v2"


class TestElide:
    def test_leaves_short_text(self):
        assert mod.elide("abc", 5) == "abc"

    def test_keeps_the_start_and_the_end(self):
        cut = mod.elide("train_only_inline_kv_lam0_s3", 15)
        assert len(cut) == 15
        assert cut.startswith("train_o")
        assert cut.endswith("_s3")
        assert "..." in cut

    def test_cuts_plainly_below_the_ellipsis(self):
        assert mod.elide("abcdef", 2) == "ab"


def jobs(tmp_path):
    return [
        mod.read_state(state(tmp_path, "train_bge_v2_m3_only_yaml_lam0_s1", done=3)),
        mod.read_state(state(tmp_path, "capture_bge_v2_m3_only_yaml_lam0_s1", done=0)),
        mod.read_state(state(tmp_path, "train_minilm_l6_5fmt_lam1_s2", done=10)),
    ]


class TestGroupedRender:
    def test_heads_each_model_with_its_jobs_beneath(self, tmp_path):
        lines = mod.render(jobs(tmp_path))
        assert lines[0].strip() == "bge_v2_m3  0/2 done"
        assert "train only_yaml_lam0_s1" in lines[1]
        assert "capture only_yaml_lam0_s1" in lines[2]
        assert lines[3].strip() == "minilm_l6  1/1 done"

    def test_keeps_one_line_for_the_run(self, tmp_path):
        lines = mod.render(jobs(tmp_path))
        assert len(lines) == 2 + 3 + 1
        assert "3 jobs" in lines[-1]

    def test_lists_jobs_naming_no_model_flat_and_first(self, tmp_path):
        states = [*jobs(tmp_path), mod.read_state(state(tmp_path, "select_lambda"))]
        lines = mod.render(states)
        assert lines[0].startswith("  select_lambda")

    def test_bars_line_up_across_groups(self, tmp_path):
        lines = [line for line in mod.render(jobs(tmp_path)) if "[" in line]
        assert len({line.index("[") for line in lines}) == 1


class TestFitting:
    def test_draws_in_full_without_a_width(self, tmp_path):
        lines = mod.render(jobs(tmp_path))
        assert any("#" * 2 in line for line in lines)
        assert any("with_body/yaml" in line for line in lines)

    @pytest.mark.parametrize("width", [100, 70, 60, 50, 40, 30, 20, 12])
    def test_no_line_passes_the_width(self, tmp_path, width):
        assert all(len(line) <= width for line in mod.render(jobs(tmp_path), width))

    def test_narrows_the_bar_before_anything_else(self, tmp_path):
        full = mod.render(jobs(tmp_path))
        widest = max(len(line) for line in full)
        squeezed = mod.render(jobs(tmp_path), widest - 8)
        assert any("with_body/yaml" in line for line in squeezed)
        assert "[" + "-" * mod.BAR_WIDTH not in "".join(squeezed)

    def test_drops_the_label_when_the_bar_is_not_enough(self, tmp_path):
        lines = mod.render(jobs(tmp_path), 60)
        assert not any("with_body/yaml" in line for line in lines)
        assert any("eta" in line for line in lines)

    def test_drops_the_bar_before_shortening_names(self, tmp_path):
        lines = mod.render(jobs(tmp_path), 56)
        assert not any("[" in line for line in lines)
        assert any("train only_yaml_lam0_s1" in line for line in lines)

    def test_shortens_the_names_before_cutting_lines(self, tmp_path):
        lines = mod.render(jobs(tmp_path), 40)
        rows = [line for line in lines if "/10" in line]
        assert any("..." in row for row in rows)
        assert all(
            row.rstrip().endswith(("eta 0:00:29", "eta 0:00:30", "starting", "done"))
            for row in rows
        )

    def test_a_shortened_name_keeps_more_of_its_start(self):
        cut = mod.elide("capture only_markdown_lam0_s2", 18)
        assert cut.startswith("capture on")
        assert cut.endswith("_s2")

    def test_keeps_a_wide_enough_block_whole(self, tmp_path):
        assert mod.render(jobs(tmp_path), 500) == mod.render(jobs(tmp_path))


class TestDrawClears:
    def test_clears_below_a_redrawn_block(self):
        out = io.StringIO()
        mod.draw(["a"], out, redraw=3)
        assert out.getvalue().endswith(mod.CLEAR_BELOW)

    def test_a_first_block_clears_nothing(self):
        out = io.StringIO()
        mod.draw(["a"], out, redraw=0)
        assert mod.CLEAR_BELOW not in out.getvalue()


class TestShortNotes:
    def test_keep_only_the_time_left(self, tmp_path):
        running = mod.read_state(state(tmp_path, "a", done=1))
        assert mod.notes(running, short=True).startswith("eta ")

    def test_drop_the_times_of_finished_and_starting_jobs(self, tmp_path):
        finished = mod.read_state(state(tmp_path, "a", done=10, total=10))
        starting = mod.read_state(state(tmp_path, "b", done=0))
        assert mod.notes(finished, short=True) == "done"
        assert mod.notes(starting, short=True) == "starting"

    def test_name_a_failure_in_either_form(self, tmp_path):
        failed = mod.read_state(state(tmp_path, "a", failed=1))
        assert mod.notes(failed, short=True) == mod.notes(failed) == "FAILED"
