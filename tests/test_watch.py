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
        assert args.pattern == mod.DEFAULT_PATTERN == "*.progress"
        assert args.interval == mod.DEFAULT_INTERVAL
        assert not args.once

    def test_takes_a_pattern_and_an_interval(self):
        args = mod.build_parser().parse_args(
            ["--pattern", "all_*.progress", "--interval", "2", "--once"]
        )
        assert args.pattern == "all_*.progress"
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
