"""Tests for scripts.hold_gpu."""

from __future__ import annotations

from scripts import hold_gpu as mod


class TestShouldStop:
    def test_holds_with_no_condition_met(self, tmp_path):
        flag = tmp_path / "hold"
        flag.touch()
        assert not mod.should_stop(10.0, 60.0, flag, False)

    def test_stops_when_the_file_is_gone(self, tmp_path):
        assert mod.should_stop(10.0, 60.0, tmp_path / "hold", False)

    def test_stops_at_the_time_limit(self, tmp_path):
        flag = tmp_path / "hold"
        flag.touch()
        assert mod.should_stop(3600.0, 60.0, flag, False)

    def test_no_limit_holds_past_any_time(self):
        assert not mod.should_stop(10**7, 0.0, None, False)

    def test_stops_on_a_signal(self):
        assert mod.should_stop(0.0, 0.0, None, True)


class TestParser:
    def test_holds_until_stopped_by_default(self):
        args = mod.build_parser().parse_args([])
        assert args.minutes == 0.0
        assert args.until_gone is None
