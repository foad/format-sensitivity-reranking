from __future__ import annotations

import io

import pytest
from scripts import watch as mod

from fsr.reporting import PROGRESS_TAG


def log(directory, name, lines):
    path = directory / f"{name}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def progress_line(done, total, label="with_body/yaml", elapsed=60.0, eta=30.0):
    return (
        f"{PROGRESS_TAG} done={done} total={total} "
        f"elapsed={elapsed} eta={eta} label={label}"
    )


class TestReadState:
    def test_reports_the_last_progress_line(self, tmp_path):
        path = log(
            tmp_path,
            "test_cross_bge_base",
            [progress_line(1, 10), progress_line(4, 10)],
        )
        state = mod.read_state(path)
        assert (state.done, state.total) == (4, 10)

    def test_names_the_job_after_the_log(self, tmp_path):
        path = log(tmp_path, "test_cross_jina_v2", [progress_line(1, 10)])
        assert mod.read_state(path).name == "test_cross_jina_v2"

    def test_carries_the_label_and_the_times(self, tmp_path):
        path = log(
            tmp_path, "j", [progress_line(2, 4, "metadata_only/toml", 10.0, 5.0)]
        )
        state = mod.read_state(path)
        assert state.label == "metadata_only/toml"
        assert (state.elapsed, state.eta) == (10.0, 5.0)

    def test_reports_no_units_for_a_log_with_no_progress(self, tmp_path):
        path = log(tmp_path, "j", ["Loading tokenizers...", "  ok  model/a"])
        state = mod.read_state(path)
        assert (state.done, state.total) == (0, 0)
        assert not state.finished

    def test_reports_no_units_for_an_empty_log(self, tmp_path):
        path = log(tmp_path, "j", [])
        assert mod.read_state(path).total == 0

    def test_reports_a_failure(self, tmp_path):
        path = log(tmp_path, "j", [progress_line(1, 10), "FAILURES (1)"])
        assert mod.read_state(path).failed

    def test_reports_a_failure_before_any_progress(self, tmp_path):
        assert mod.read_state(log(tmp_path, "j", ["FAILURES (1)"])).failed

    def test_survives_a_log_it_cannot_read(self, tmp_path):
        assert mod.read_state(tmp_path / "absent.log").total == 0

    def test_reports_a_finished_job(self, tmp_path):
        assert mod.read_state(log(tmp_path, "j", [progress_line(10, 10)])).finished

    def test_a_job_past_its_total_is_finished(self, tmp_path):
        assert mod.read_state(log(tmp_path, "j", [progress_line(11, 10)])).finished


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
    def test_reports_when_no_log_is_present(self):
        assert mod.render([]) == ["no job logs found"]

    def test_one_line_per_job_and_one_for_the_run(self, tmp_path):
        states = [
            mod.read_state(log(tmp_path, "a", [progress_line(1, 10)])),
            mod.read_state(log(tmp_path, "b", [progress_line(2, 10)])),
        ]
        assert len(mod.render(states)) == 3

    def test_a_running_job_shows_its_label_and_time_left(self, tmp_path):
        state = mod.read_state(log(tmp_path, "a", [progress_line(1, 10, "x/y")]))
        assert "x/y" in mod.render([state])[0]
        assert "eta 0:00:30" in mod.render([state])[0]

    def test_a_finished_job_shows_how_long_it_took(self, tmp_path):
        state = mod.read_state(log(tmp_path, "a", [progress_line(10, 10, elapsed=90)]))
        assert "done in 0:01:30" in mod.render([state])[0]

    def test_a_waiting_job_says_so(self, tmp_path):
        state = mod.read_state(log(tmp_path, "a", ["starting"]))
        assert "waiting" in mod.render([state])[0]

    def test_a_failed_job_says_so(self, tmp_path):
        state = mod.read_state(log(tmp_path, "a", ["FAILURES (1)"]))
        assert "FAILED" in mod.render([state])[0]

    def test_the_run_line_totals_the_units(self, tmp_path):
        states = [
            mod.read_state(log(tmp_path, "a", [progress_line(1, 10)])),
            mod.read_state(log(tmp_path, "b", [progress_line(3, 10)])),
        ]
        assert "2 jobs, 4/20 units" in mod.render(states)[-1]

    def test_the_run_line_takes_the_longest_time_left(self, tmp_path):
        states = [
            mod.read_state(log(tmp_path, "a", [progress_line(1, 10, eta=30)])),
            mod.read_state(log(tmp_path, "b", [progress_line(3, 10, eta=600)])),
        ]
        assert "eta 0:10:00" in mod.render(states)[-1]

    def test_the_run_line_gives_no_time_left_when_all_are_done(self, tmp_path):
        state = mod.read_state(log(tmp_path, "a", [progress_line(10, 10)]))
        assert "eta" not in mod.render([state])[-1]

    def test_names_are_padded_to_one_width(self, tmp_path):
        states = [
            mod.read_state(log(tmp_path, "short", [progress_line(1, 10)])),
            mod.read_state(log(tmp_path, "a_much_longer_name", [progress_line(1, 10)])),
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
        log(tmp_path, "a", [progress_line(1, 10)])
        out = io.StringIO()
        assert mod.watch(tmp_path, "*.log", 0.0, True, out) == 0
        assert "1/10" in out.getvalue()

    def test_stops_when_every_job_is_finished(self, tmp_path):
        log(tmp_path, "a", [progress_line(10, 10)])
        out = io.StringIO()
        assert mod.watch(tmp_path, "*.log", 0.0, False, out) == 0

    def test_reports_a_failure_in_the_exit_code(self, tmp_path):
        log(tmp_path, "a", [progress_line(10, 10), "FAILURES (1)"])
        out = io.StringIO()
        assert mod.watch(tmp_path, "*.log", 0.0, False, out) == 1

    def test_keeps_reading_until_the_jobs_settle(self, tmp_path, monkeypatch):
        path = log(tmp_path, "a", [progress_line(1, 10)])
        sleeps = []

        def finish(_seconds):
            sleeps.append(_seconds)
            path.write_text(progress_line(10, 10) + "\n")

        monkeypatch.setattr(mod.time, "sleep", finish)
        out = io.StringIO()
        assert mod.watch(tmp_path, "*.log", 5.0, False, out) == 0
        assert sleeps == [5.0]

    def test_leaves_out_the_runner_logs(self, tmp_path):
        log(tmp_path, "a", [progress_line(10, 10)])
        log(tmp_path, "a.hare", ["container output"])
        out = io.StringIO()
        mod.watch(tmp_path, "*.log", 0.0, True, out)
        assert "a.hare" not in out.getvalue()

    def test_honours_the_pattern(self, tmp_path):
        log(tmp_path, "test_cross_a", [progress_line(10, 10)])
        log(tmp_path, "test_within_a", [progress_line(10, 10)])
        out = io.StringIO()
        mod.watch(tmp_path, "test_cross_*.log", 0.0, True, out)
        assert "test_within_a" not in out.getvalue()

    def test_reports_an_empty_directory(self, tmp_path):
        out = io.StringIO()
        assert mod.watch(tmp_path, "*.log", 0.0, True, out) == 0
        assert "no job logs found" in out.getvalue()


class TestBuildParser:
    def test_defaults_to_the_results_directory(self):
        args = mod.build_parser().parse_args([])
        assert args.results_subdir == mod.RESULTS_SUBDIR
        assert args.pattern == mod.DEFAULT_PATTERN
        assert args.interval == mod.DEFAULT_INTERVAL
        assert not args.once

    def test_takes_a_pattern_and_an_interval(self):
        args = mod.build_parser().parse_args(
            ["--pattern", "all_*.log", "--interval", "2", "--once"]
        )
        assert args.pattern == "all_*.log"
        assert args.interval == pytest.approx(2.0)
        assert args.once


class TestMain:
    def test_stopping_it_is_not_an_error(self, tmp_path, monkeypatch):
        log(tmp_path / "h1", "a", [progress_line(1, 10)])
        monkeypatch.setattr("sys.argv", ["prog", "--data-root", str(tmp_path)])

        def interrupt(_seconds):
            raise KeyboardInterrupt

        monkeypatch.setattr(mod.time, "sleep", interrupt)
        assert mod.main() == mod.INTERRUPTED == 130

    def test_watches_the_directory_the_arguments_name(self, tmp_path, monkeypatch):
        directory = tmp_path / "h1"
        log(directory, "a", [progress_line(10, 10)])
        monkeypatch.setattr(
            "sys.argv",
            ["prog", "--data-root", str(tmp_path), "--once"],
        )
        assert mod.main() == 0
