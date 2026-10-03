from __future__ import annotations

import numpy as np
import pytest
import torch

from fsr.training.checkpoint import (
    CHECKPOINT_NAME,
    capture_rng,
    load_checkpoint,
    restore_rng,
    save_checkpoint,
    should_checkpoint,
    trainable_state,
)


class TinyModel(torch.nn.Module):
    """A frozen layer and a trainable one, as an adapted model has."""

    def __init__(self):
        """Build the frozen base and the trainable adapter."""
        super().__init__()
        self.frozen = torch.nn.Linear(4, 4)
        self.adapter = torch.nn.Linear(4, 2)
        base = torch.Generator().manual_seed(1234)
        with torch.no_grad():
            self.frozen.weight.copy_(torch.rand(4, 4, generator=base))
            self.frozen.bias.copy_(torch.rand(4, generator=base))
        for p in self.frozen.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        return self.adapter(self.frozen(x))


def build_run(seed=0):
    torch.manual_seed(seed)
    model = TinyModel()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=0.1
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    sampler = np.random.default_rng(seed)
    return model, optimizer, scheduler, sampler


def train_steps(model, optimizer, scheduler, sampler, data, n_steps):
    """Run n_steps, drawing each batch through the sampler."""
    for _ in range(n_steps):
        idx = sampler.choice(len(data), size=2, replace=False)
        batch = data[idx]
        loss = model(batch).pow(2).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return loss.item()


@pytest.fixture
def data():
    return torch.arange(40, dtype=torch.float32).reshape(10, 4)


class TestShouldCheckpoint:
    def test_saves_on_a_multiple_of_the_interval(self):
        assert should_checkpoint(100, 50)

    def test_does_not_save_between_intervals(self):
        assert not should_checkpoint(99, 50)

    def test_zero_disables_checkpointing(self):
        assert not should_checkpoint(100, 0)

    def test_a_negative_interval_disables_checkpointing(self):
        assert not should_checkpoint(100, -1)


class TestTrainableState:
    def test_holds_only_the_trainable_parameters(self):
        names = set(trainable_state(TinyModel()))
        assert names == {"adapter.weight", "adapter.bias"}

    def test_detaches_from_the_graph(self):
        assert all(not t.requires_grad for t in trainable_state(TinyModel()).values())

    def test_copies_rather_than_aliases(self):
        model = TinyModel()
        saved = trainable_state(model)
        with torch.no_grad():
            model.adapter.bias.add_(1.0)
        assert not torch.equal(saved["adapter.bias"], model.adapter.bias)


class TestCaptureRng:
    def test_holds_every_source(self):
        state = capture_rng(np.random.default_rng(0))
        assert set(state) == {"torch", "torch_cuda", "numpy", "sampler"}

    def test_reports_no_cuda_state_without_a_device(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert capture_rng(np.random.default_rng(0))["torch_cuda"] is None

    def test_captures_the_cuda_state_when_a_device_is_present(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: ["device0"])
        assert capture_rng(np.random.default_rng(0))["torch_cuda"] == ["device0"]


class TestRestoreRng:
    def test_restores_the_sampler_sequence(self):
        sampler = np.random.default_rng(0)
        sampler.random(5)
        state = capture_rng(sampler)
        expected = sampler.random(5)
        sampler.random(5)
        restore_rng(state, sampler)
        assert np.array_equal(sampler.random(5), expected)

    def test_restores_the_torch_sequence(self):
        state = capture_rng(np.random.default_rng(0))
        expected = torch.randn(3)
        torch.randn(3)
        restore_rng(state, np.random.default_rng(0))
        assert torch.equal(torch.randn(3), expected)

    def test_restores_the_legacy_numpy_sequence(self):
        state = capture_rng(np.random.default_rng(0))
        expected = np.random.rand(3)
        np.random.rand(3)
        restore_rng(state, np.random.default_rng(0))
        assert np.array_equal(np.random.rand(3), expected)

    def test_skips_the_cuda_state_without_a_device(self, monkeypatch):
        state = capture_rng(np.random.default_rng(0))
        state["torch_cuda"] = ["device0"]
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        restore_rng(state, np.random.default_rng(0))

    def test_restores_the_cuda_state_when_the_device_count_matches(self, monkeypatch):
        seen = []
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
        monkeypatch.setattr(torch.cuda, "set_rng_state_all", seen.append)
        state = capture_rng(np.random.default_rng(0))
        state["torch_cuda"] = ["device0"]
        restore_rng(state, np.random.default_rng(0))
        assert seen == [["device0"]]

    def test_skips_the_cuda_state_when_the_device_count_differs(self, monkeypatch):
        seen = []
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
        monkeypatch.setattr(torch.cuda, "set_rng_state_all", seen.append)
        state = capture_rng(np.random.default_rng(0))
        state["torch_cuda"] = ["device0"]
        restore_rng(state, np.random.default_rng(0))
        assert seen == []


class TestSaveCheckpoint:
    def test_writes_the_file(self, tmp_path):
        model, optimizer, scheduler, sampler = build_run()
        path = save_checkpoint(
            tmp_path / "run" / CHECKPOINT_NAME,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=7,
            sampler=sampler,
        )
        assert path.exists()

    def test_holds_every_part_of_the_state(self, tmp_path):
        model, optimizer, scheduler, sampler = build_run()
        path = save_checkpoint(
            tmp_path / CHECKPOINT_NAME,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=7,
            sampler=sampler,
        )
        state = torch.load(path, weights_only=False)
        assert set(state) == {"step", "model", "optimizer", "scheduler", "rng"}
        assert state["step"] == 7

    def test_leaves_no_partial_file_behind(self, tmp_path):
        model, optimizer, scheduler, sampler = build_run()
        path = tmp_path / CHECKPOINT_NAME
        model.named_parameters = None
        with pytest.raises(TypeError):
            save_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=1,
                sampler=sampler,
            )
        assert list(tmp_path.iterdir()) == []


class TestLoadCheckpoint:
    def test_returns_the_saved_step(self, tmp_path):
        model, optimizer, scheduler, sampler = build_run()
        path = save_checkpoint(
            tmp_path / CHECKPOINT_NAME,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=42,
            sampler=sampler,
        )
        fresh = build_run()
        assert (
            load_checkpoint(
                path,
                model=fresh[0],
                optimizer=fresh[1],
                scheduler=fresh[2],
                sampler=fresh[3],
            )
            == 42
        )

    def test_restores_the_trainable_weights(self, tmp_path, data):
        model, optimizer, scheduler, sampler = build_run()
        train_steps(model, optimizer, scheduler, sampler, data, 3)
        path = save_checkpoint(
            tmp_path / CHECKPOINT_NAME,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=3,
            sampler=sampler,
        )
        fresh = build_run()
        load_checkpoint(
            path,
            model=fresh[0],
            optimizer=fresh[1],
            scheduler=fresh[2],
            sampler=fresh[3],
        )
        assert torch.equal(fresh[0].adapter.weight, model.adapter.weight)

    def test_restores_the_scheduler_position(self, tmp_path, data):
        model, optimizer, scheduler, sampler = build_run()
        train_steps(model, optimizer, scheduler, sampler, data, 5)
        path = save_checkpoint(
            tmp_path / CHECKPOINT_NAME,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=5,
            sampler=sampler,
        )
        fresh = build_run()
        load_checkpoint(
            path,
            model=fresh[0],
            optimizer=fresh[1],
            scheduler=fresh[2],
            sampler=fresh[3],
        )
        assert fresh[2].get_last_lr() == scheduler.get_last_lr()

    def test_rejects_a_checkpoint_from_another_model(self, tmp_path):
        model, optimizer, scheduler, sampler = build_run()
        path = save_checkpoint(
            tmp_path / CHECKPOINT_NAME,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=1,
            sampler=sampler,
        )
        state = torch.load(path, weights_only=False)
        state["model"]["other.weight"] = torch.zeros(2)
        torch.save(state, path)
        fresh = build_run()
        with pytest.raises(ValueError, match=r"other\.weight"):
            load_checkpoint(
                path,
                model=fresh[0],
                optimizer=fresh[1],
                scheduler=fresh[2],
                sampler=fresh[3],
            )


class TestResumeEquality:
    def test_a_resumed_run_matches_an_uninterrupted_one(self, tmp_path, data):
        uninterrupted = build_run()
        final_loss = train_steps(*uninterrupted, data, 10)

        first = build_run()
        train_steps(*first, data, 4)
        path = save_checkpoint(
            tmp_path / CHECKPOINT_NAME,
            model=first[0],
            optimizer=first[1],
            scheduler=first[2],
            step=4,
            sampler=first[3],
        )

        resumed = build_run(seed=99)
        step = load_checkpoint(
            path,
            model=resumed[0],
            optimizer=resumed[1],
            scheduler=resumed[2],
            sampler=resumed[3],
        )
        resumed_loss = train_steps(*resumed, data, 10 - step)

        assert resumed_loss == pytest.approx(final_loss, rel=0, abs=0)
        assert torch.equal(resumed[0].adapter.weight, uninterrupted[0].adapter.weight)
        assert torch.equal(resumed[0].adapter.bias, uninterrupted[0].adapter.bias)

    def test_the_batch_order_continues_where_it_stopped(self, tmp_path, data):
        uninterrupted = build_run()
        drawn = [
            uninterrupted[3].choice(len(data), size=2, replace=False) for _ in range(10)
        ]

        first = build_run()
        for _ in range(4):
            first[3].choice(len(data), size=2, replace=False)
        path = save_checkpoint(
            tmp_path / CHECKPOINT_NAME,
            model=first[0],
            optimizer=first[1],
            scheduler=first[2],
            step=4,
            sampler=first[3],
        )

        resumed = build_run(seed=99)
        load_checkpoint(
            path,
            model=resumed[0],
            optimizer=resumed[1],
            scheduler=resumed[2],
            sampler=resumed[3],
        )
        rest = [resumed[3].choice(len(data), size=2, replace=False) for _ in range(6)]
        assert all(np.array_equal(a, b) for a, b in zip(rest, drawn[4:], strict=True))

    def test_a_run_resumed_without_the_rng_state_diverges(self, tmp_path, data):
        uninterrupted = build_run()
        train_steps(*uninterrupted, data, 10)

        first = build_run()
        train_steps(*first, data, 4)
        path = save_checkpoint(
            tmp_path / CHECKPOINT_NAME,
            model=first[0],
            optimizer=first[1],
            scheduler=first[2],
            step=4,
            sampler=first[3],
        )
        resumed = build_run(seed=99)
        load_checkpoint(
            path,
            model=resumed[0],
            optimizer=resumed[1],
            scheduler=resumed[2],
            sampler=resumed[3],
        )
        resumed[3].bit_generator.state = np.random.default_rng(1).bit_generator.state
        train_steps(*resumed, data, 6)
        assert not torch.equal(
            resumed[0].adapter.weight, uninterrupted[0].adapter.weight
        )
