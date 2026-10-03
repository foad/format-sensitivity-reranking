from __future__ import annotations

import json

import pytest

from fsr.training.log import (
    END,
    EVAL,
    LOG_NAME,
    RUN,
    STEP,
    TrainLog,
    eta_seconds,
    events,
    format_duration,
    last_step,
    read_log,
    rewind,
    step_line,
)


@pytest.fixture
def log_path(tmp_path):
    return tmp_path / "run" / LOG_NAME


def write_run(path, steps):
    with TrainLog(path) as log:
        log.write(RUN, model="bge-base", max_steps=max(steps, default=0))
        for s in steps:
            log.write(STEP, step=s, loss=1.0 / s)
        log.write(END, step=max(steps, default=0))


class TestTrainLog:
    def test_creates_the_directory(self, log_path):
        TrainLog(log_path).close()
        assert log_path.exists()

    def test_writes_one_object_per_line(self, log_path):
        write_run(log_path, [1, 2])
        lines = log_path.read_text().splitlines()
        assert len(lines) == 4
        assert all(json.loads(line) for line in lines)

    def test_tags_each_record_with_its_event(self, log_path):
        write_run(log_path, [1])
        assert [json.loads(x)["event"] for x in log_path.read_text().splitlines()] == [
            RUN,
            STEP,
            END,
        ]

    def test_returns_the_record_it_wrote(self, log_path):
        with TrainLog(log_path) as log:
            assert log.write(STEP, step=3) == {"event": STEP, "step": 3}

    def test_flushes_every_line(self, log_path):
        log = TrainLog(log_path)
        log.write(STEP, step=1)
        assert read_log(log_path) == [{"event": STEP, "step": 1}]
        log.close()

    def test_appends_to_an_existing_log(self, log_path):
        write_run(log_path, [1])
        with TrainLog(log_path) as log:
            log.write(STEP, step=2)
        assert last_step(read_log(log_path)) == 2

    def test_serialises_a_value_json_cannot_hold(self, log_path, tmp_path):
        with TrainLog(log_path) as log:
            log.write(RUN, out_dir=tmp_path)
        assert read_log(log_path)[0]["out_dir"] == str(tmp_path)


class TestReadLog:
    def test_returns_an_empty_list_for_a_missing_file(self, log_path):
        assert read_log(log_path) == []

    def test_returns_the_records_in_order(self, log_path):
        write_run(log_path, [1, 2, 3])
        assert [r.get("step") for r in read_log(log_path)] == [None, 1, 2, 3, 3]

    def test_ignores_a_blank_line(self, log_path):
        log_path.parent.mkdir(parents=True)
        log_path.write_text('{"event": "step", "step": 1}\n\n')
        assert read_log(log_path) == [{"event": STEP, "step": 1}]

    def test_stops_at_a_partial_trailing_line(self, log_path):
        log_path.parent.mkdir(parents=True)
        log_path.write_text('{"event": "step", "step": 1}\n{"event": "ste')
        assert read_log(log_path) == [{"event": STEP, "step": 1}]


class TestEvents:
    def test_keeps_only_the_named_kind(self, log_path):
        write_run(log_path, [1, 2])
        assert len(events(read_log(log_path), STEP)) == 2

    def test_returns_an_empty_list_when_none_match(self, log_path):
        write_run(log_path, [1])
        assert events(read_log(log_path), EVAL) == []

    def test_keeps_the_order(self, log_path):
        write_run(log_path, [1, 2, 3])
        assert [r["step"] for r in events(read_log(log_path), STEP)] == [1, 2, 3]


class TestLastStep:
    def test_returns_the_highest_step(self, log_path):
        write_run(log_path, [1, 5, 3])
        assert last_step(read_log(log_path)) == 5

    def test_returns_zero_for_an_empty_log(self):
        assert last_step([]) == 0

    def test_ignores_a_record_with_no_step(self):
        assert last_step([{"event": RUN, "model": "bge-base"}]) == 0


class TestRewind:
    def test_drops_the_records_past_the_step(self, log_path):
        write_run(log_path, [1, 2, 3, 4])
        assert rewind(log_path, 2) == 3
        assert [r.get("step") for r in read_log(log_path)] == [None, 1, 2]

    def test_keeps_the_run_header(self, log_path):
        write_run(log_path, [1, 2])
        rewind(log_path, 0)
        assert [r["event"] for r in read_log(log_path)] == [RUN]

    def test_drops_nothing_when_the_log_is_short(self, log_path):
        write_run(log_path, [1, 2])
        assert rewind(log_path, 9) == 0

    def test_leaves_a_missing_file_alone(self, log_path):
        assert rewind(log_path, 5) == 0
        assert not log_path.exists()

    def test_a_resumed_run_has_no_duplicate_step(self, log_path):
        write_run(log_path, [1, 2, 3])
        rewind(log_path, 2)
        with TrainLog(log_path) as log:
            log.write(STEP, step=3)
        steps = [r["step"] for r in events(read_log(log_path), STEP)]
        assert steps == [1, 2, 3]


class TestEtaSeconds:
    def test_estimates_at_the_rate_so_far(self):
        assert eta_seconds(10, 110, 20.0) == 200.0

    def test_is_zero_before_the_first_step(self):
        assert eta_seconds(0, 100, 0.0) == 0.0

    def test_is_zero_at_the_last_step(self):
        assert eta_seconds(100, 100, 50.0) == 0.0

    def test_is_zero_past_the_last_step(self):
        assert eta_seconds(120, 100, 60.0) == 0.0


class TestFormatDuration:
    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (0, "0:00:00"),
            (9, "0:00:09"),
            (90, "0:01:30"),
            (3600, "1:00:00"),
            (3661, "1:01:01"),
            (86400, "24:00:00"),
            (59.9, "0:00:59"),
        ],
    )
    def test_formats_the_duration(self, seconds, expected):
        assert format_duration(seconds) == expected


class TestStepLine:
    def test_reports_every_quantity(self):
        line = step_line(10, 500, 1.5, 1.2, 0.3, 2e-4, 20.0)
        assert line == (
            "step    10/500  loss=1.5000  rank=1.2000  inv=0.3000  "
            "lr=2.00e-04  2.00s/step  eta 0:16:20"
        )

    def test_reports_no_rate_before_the_first_step(self):
        assert "0.00s/step" in step_line(0, 500, 1.5, 1.2, 0.3, 2e-4, 0.0)

    def test_pads_the_step_to_keep_the_lines_aligned(self):
        short = step_line(1, 500, 1.0, 1.0, 0.0, 1e-4, 1.0)
        long = step_line(100, 500, 1.0, 1.0, 0.0, 1e-4, 1.0)
        assert short.index("loss=") == long.index("loss=")
