from __future__ import annotations

import numpy as np
import pytest
import torch

from fsr.reporting import PROGRESS_SUFFIX, ProgressCounter, parse_progress
from fsr.training.batch import TrainRecord
from fsr.training.checkpoint import CHECKPOINT_NAME, load_checkpoint
from fsr.training.log import EVAL, LOG_NAME, STEP, TrainLog, events, read_log, rewind
from fsr.training.loop import (
    DEFAULT_LOG_EVERY,
    LoopConfig,
    TrainResult,
    autocast_for,
    dev_sensitivity,
    train,
)
from tests.fakes import HashingPairTokenizer, ScoringModel

FORMATS_USED = ("yaml", "json", "toml")


def record(rec_id: str) -> TrainRecord:
    return TrainRecord(
        id=rec_id,
        question=f"who is {rec_id}",
        pairs=[("Name", rec_id), ("Role", "Engineer")],
        truncated_body=f"Body of {rec_id}.",
        body_budget_tokens=64,
        neg_pairs_list=[[("Name", f"{rec_id}-neg")]],
        neg_bodies_truncated=[f"Negative body of {rec_id}."],
    )


@pytest.fixture
def corpus():
    return [record(f"r{i}") for i in range(12)]


@pytest.fixture
def dev():
    return [record(f"d{i}") for i in range(6)]


def build(seed=0, width=4):
    torch.manual_seed(seed)
    model = ScoringModel(width)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    return model, HashingPairTokenizer(width), optimizer, scheduler


def config(**overrides):
    base = {
        "max_steps": 4,
        "physical_batch": 2,
        "grad_accum": 2,
        "eval_every": 2,
        "lambda_inv": 1.0,
        "log_every": 1,
    }
    base.update(overrides)
    return LoopConfig(**base)


def run(
    tmp_path,
    corpus,
    dev,
    *,
    seed=0,
    cfg=None,
    start_step=0,
    checkpoint=False,
    counter=None,
):
    model, tokenizer, optimizer, scheduler = build(seed)
    sampler = np.random.default_rng(seed)
    with TrainLog(tmp_path / LOG_NAME) as log:
        result = train(
            model,
            tokenizer,
            records=corpus,
            dev_records=dev,
            formats=FORMATS_USED,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=sampler,
            config=cfg or config(),
            device="cpu",
            log=log,
            checkpoint_path=(tmp_path / CHECKPOINT_NAME) if checkpoint else None,
            start_step=start_step,
            counter=counter,
        )
    return model, optimizer, scheduler, sampler, result


class TestAutocastFor:
    # torch reports the missing device when the suite runs off a GPU.
    @pytest.mark.filterwarnings("ignore:User provided device_type")
    def test_uses_bfloat16_on_a_cuda_device(self):
        assert isinstance(autocast_for("cuda"), torch.autocast)

    @pytest.mark.filterwarnings("ignore:User provided device_type")
    def test_names_an_indexed_cuda_device(self):
        assert isinstance(autocast_for("cuda:1"), torch.autocast)

    def test_does_nothing_on_the_host(self):
        assert not isinstance(autocast_for("cpu"), torch.autocast)


class TestLoopConfig:
    def test_reports_the_effective_batch(self):
        assert config(physical_batch=2, grad_accum=8).effective_batch == 16

    def test_defaults_to_twenty_steps_between_lines(self):
        assert LoopConfig(1, 1, 1, 1, 1.0).log_every == DEFAULT_LOG_EVERY == 20

    def test_defaults_to_no_checkpointing(self):
        assert LoopConfig(1, 1, 1, 1, 1.0).checkpoint_every == 0


class TestDevSensitivity:
    def test_reports_an_effect_size_and_its_pair(self, dev):
        model, tokenizer, _, _ = build()
        max_abs_d, summary = dev_sensitivity(
            model, tokenizer, dev, FORMATS_USED, "cpu", 4
        )
        assert max_abs_d >= 0.0
        assert " - " in summary["max_d_pair"]

    def test_covers_every_training_format(self, dev):
        model, tokenizer, _, _ = build()
        _, summary = dev_sensitivity(model, tokenizer, dev, FORMATS_USED, "cpu", 4)
        assert set(summary["per_format"]) == set(FORMATS_USED)

    def test_pairs_only_the_training_formats(self, dev):
        model, tokenizer, _, _ = build()
        _, summary = dev_sensitivity(model, tokenizer, dev, ("yaml", "json"), "cpu", 4)
        assert [p["pair"] for p in summary["pairwise"]] == ["yaml - json"]

    def test_reports_the_mean_difference_of_each_pair(self, dev):
        model, tokenizer, _, _ = build()
        _, summary = dev_sensitivity(model, tokenizer, dev, FORMATS_USED, "cpu", 4)
        assert all("mean_diff" in p for p in summary["pairwise"])


class TestTrain:
    def test_runs_the_step_budget(self, tmp_path, corpus, dev):
        _, _, _, _, result = run(tmp_path, corpus, dev)
        assert isinstance(result, TrainResult)
        assert result.step == 4

    def test_updates_the_parameters(self, tmp_path, corpus, dev):
        before = build()[0].classifier.weight.detach().clone()
        model, _, _, _, _ = run(tmp_path, corpus, dev)
        assert not torch.equal(model.classifier.weight, before)

    def test_advances_the_scheduler_once_per_step(self, tmp_path, corpus, dev):
        _, _, scheduler, _, _ = run(tmp_path, corpus, dev)
        assert scheduler.last_epoch == 4

    def test_leaves_no_gradient_behind(self, tmp_path, corpus, dev):
        model, _, _, _, _ = run(tmp_path, corpus, dev)
        assert all(p.grad is None or torch.all(p.grad == 0) for p in model.parameters())

    def test_leaves_the_model_in_training_mode(self, tmp_path, corpus, dev):
        model, _, _, _, _ = run(tmp_path, corpus, dev)
        assert model.training

    def test_reports_the_last_development_score(self, tmp_path, corpus, dev):
        _, _, _, _, result = run(tmp_path, corpus, dev)
        assert result.dev_max_abs_d is not None

    def test_draws_the_micro_batches_through_the_sampler(self, tmp_path, corpus, dev):
        _, _, _, sampler, _ = run(tmp_path, corpus, dev)
        fresh = np.random.default_rng(0)
        for _ in range(4 * 2):
            fresh.choice(len(corpus), size=2, replace=False)
        assert sampler.bit_generator.state == fresh.bit_generator.state

    def test_does_nothing_when_the_budget_is_already_spent(self, tmp_path, corpus, dev):
        _, _, _, _, result = run(tmp_path, corpus, dev, start_step=4)
        assert result.step == 4
        assert events(read_log(tmp_path / LOG_NAME), STEP) == []


class TestLogging:
    def test_writes_a_step_record(self, tmp_path, corpus, dev):
        run(tmp_path, corpus, dev)
        steps = events(read_log(tmp_path / LOG_NAME), STEP)
        assert [r["step"] for r in steps] == [1, 2, 3, 4]

    def test_a_step_record_holds_every_loss_term(self, tmp_path, corpus, dev):
        run(tmp_path, corpus, dev)
        first = events(read_log(tmp_path / LOG_NAME), STEP)[0]
        assert {"loss", "l_rank", "l_inv", "lr", "elapsed_s"} <= set(first)

    def test_writes_an_eval_record_on_each_eval_step(self, tmp_path, corpus, dev):
        run(tmp_path, corpus, dev)
        evals = events(read_log(tmp_path / LOG_NAME), EVAL)
        assert [r["step"] for r in evals] == [2, 4]

    def test_an_eval_record_holds_what_the_original_log_held(
        self, tmp_path, corpus, dev
    ):
        run(tmp_path, corpus, dev)
        first = events(read_log(tmp_path / LOG_NAME), EVAL)[0]
        assert {
            "step",
            "loss",
            "l_rank",
            "l_inv",
            "dev_max_abs_d",
            "dev_max_d_pair",
        } <= set(first)

    def test_evaluates_on_the_last_step_even_off_the_interval(
        self, tmp_path, corpus, dev
    ):
        run(tmp_path, corpus, dev, cfg=config(max_steps=3, eval_every=2))
        evals = events(read_log(tmp_path / LOG_NAME), EVAL)
        assert [r["step"] for r in evals] == [2, 3]

    def test_writes_an_end_record(self, tmp_path, corpus, dev):
        run(tmp_path, corpus, dev)
        assert read_log(tmp_path / LOG_NAME)[-1]["event"] == "end"

    def test_prints_the_first_step_and_then_the_interval(
        self, tmp_path, corpus, dev, capsys
    ):
        run(tmp_path, corpus, dev, cfg=config(max_steps=4, log_every=3))
        lines = capsys.readouterr().out.splitlines()
        printed = [x for x in lines if x.startswith("step")]
        assert [x.split()[1] for x in printed] == ["1/4", "3/4"]

    def test_the_step_line_carries_the_time_left(self, tmp_path, corpus, dev, capsys):
        run(tmp_path, corpus, dev)
        assert "eta " in capsys.readouterr().out

    def test_prints_the_development_score(self, tmp_path, corpus, dev, capsys):
        run(tmp_path, corpus, dev)
        assert "dev max|d|=" in capsys.readouterr().out


class TestCheckpointing:
    def test_writes_no_checkpoint_without_a_path(self, tmp_path, corpus, dev):
        run(tmp_path, corpus, dev, cfg=config(checkpoint_every=1))
        assert not (tmp_path / CHECKPOINT_NAME).exists()

    def test_writes_no_checkpoint_when_the_interval_is_zero(
        self, tmp_path, corpus, dev
    ):
        run(tmp_path, corpus, dev, checkpoint=True, cfg=config(checkpoint_every=0))
        assert not (tmp_path / CHECKPOINT_NAME).exists()

    def test_writes_a_checkpoint_on_the_interval(self, tmp_path, corpus, dev):
        run(tmp_path, corpus, dev, checkpoint=True, cfg=config(checkpoint_every=2))
        state = torch.load(tmp_path / CHECKPOINT_NAME, weights_only=False)
        assert state["step"] == 4

    def test_the_checkpoint_holds_the_step_it_was_taken_at(self, tmp_path, corpus, dev):
        run(
            tmp_path,
            corpus,
            dev,
            checkpoint=True,
            cfg=config(max_steps=3, checkpoint_every=3),
        )
        assert torch.load(tmp_path / CHECKPOINT_NAME, weights_only=False)["step"] == 3


class TestResumeEquality:
    def test_a_resumed_run_matches_an_uninterrupted_one(self, tmp_path, corpus, dev):
        straight = tmp_path / "straight"
        straight.mkdir()
        whole, _, _, _, whole_result = run(
            straight, corpus, dev, cfg=config(max_steps=6, eval_every=3)
        )

        broken = tmp_path / "broken"
        broken.mkdir()
        run(
            broken,
            corpus,
            dev,
            cfg=config(max_steps=3, eval_every=3, checkpoint_every=3),
            checkpoint=True,
        )

        model, tokenizer, optimizer, scheduler = build(seed=99)
        sampler = np.random.default_rng(99)
        step = load_checkpoint(
            broken / CHECKPOINT_NAME,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=sampler,
        )
        rewind(broken / LOG_NAME, step)
        with TrainLog(broken / LOG_NAME) as log:
            resumed = train(
                model,
                tokenizer,
                records=corpus,
                dev_records=dev,
                formats=FORMATS_USED,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=sampler,
                config=config(max_steps=6, eval_every=3),
                device="cpu",
                log=log,
                start_step=step,
            )

        assert resumed.step == whole_result.step == 6
        assert resumed.dev_max_abs_d == whole_result.dev_max_abs_d
        assert torch.equal(model.classifier.weight, whole.classifier.weight)
        assert torch.equal(model.classifier.bias, whole.classifier.bias)

    def test_the_resumed_log_has_each_step_once(self, tmp_path, corpus, dev):
        broken = tmp_path / "broken"
        broken.mkdir()
        run(
            broken,
            corpus,
            dev,
            cfg=config(max_steps=3, eval_every=3, checkpoint_every=2),
            checkpoint=True,
        )
        model, tokenizer, optimizer, scheduler = build(seed=99)
        sampler = np.random.default_rng(99)
        step = load_checkpoint(
            broken / CHECKPOINT_NAME,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=sampler,
        )
        rewind(broken / LOG_NAME, step)
        with TrainLog(broken / LOG_NAME) as log:
            train(
                model,
                tokenizer,
                records=corpus,
                dev_records=dev,
                formats=FORMATS_USED,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=sampler,
                config=config(max_steps=4, eval_every=4),
                device="cpu",
                log=log,
                start_step=step,
            )
        steps = [r["step"] for r in events(read_log(broken / LOG_NAME), STEP)]
        assert steps == sorted(set(steps))


class TestProgress:
    def test_runs_without_a_counter(self, tmp_path, corpus, dev):
        _, _, _, _, result = run(tmp_path, corpus, dev)
        assert result.step == 4

    def test_counts_one_unit_per_step(self, tmp_path, corpus, dev):
        counter = ProgressCounter(4, tmp_path / ("job" + PROGRESS_SUFFIX))
        run(tmp_path, corpus, dev, counter=counter)
        assert counter.done == 4

    def test_writes_the_state_the_watcher_reads(self, tmp_path, corpus, dev):
        path = tmp_path / ("job" + PROGRESS_SUFFIX)
        run(tmp_path, corpus, dev, counter=ProgressCounter(4, path))
        fields = parse_progress(path.read_text())
        assert fields["done"] == "4"
        assert fields["total"] == "4"
        assert fields["failed"] == "0"

    def test_labels_a_plain_step_with_the_loss(self, tmp_path, corpus, dev):
        counter = ProgressCounter(4, tmp_path / ("job" + PROGRESS_SUFFIX))
        run(tmp_path, corpus, dev, counter=counter, cfg=config(eval_every=4))
        run_label = counter.label
        assert run_label.startswith("loss=")

    def test_labels_an_evaluation_step_with_the_development_score(
        self, tmp_path, corpus, dev
    ):
        counter = ProgressCounter(4, tmp_path / ("job" + PROGRESS_SUFFIX))
        run(tmp_path, corpus, dev, counter=counter, cfg=config(eval_every=2))
        assert "dev|d|=" in counter.label

    def test_a_resumed_job_continues_the_count(self, tmp_path, corpus, dev):
        counter = ProgressCounter(6, tmp_path / ("job" + PROGRESS_SUFFIX), done=3)
        run(
            tmp_path,
            corpus,
            dev,
            counter=counter,
            start_step=3,
            cfg=config(max_steps=6, eval_every=3),
        )
        assert counter.done == 6

    def test_a_resumed_job_reports_a_full_bar_at_the_end(self, tmp_path, corpus, dev):
        path = tmp_path / ("job" + PROGRESS_SUFFIX)
        counter = ProgressCounter(6, path, done=3)
        run(
            tmp_path,
            corpus,
            dev,
            counter=counter,
            start_step=3,
            cfg=config(max_steps=6, eval_every=3),
        )
        fields = parse_progress(path.read_text())
        assert fields["done"] == fields["total"] == "6"
