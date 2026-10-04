"""Tests for scripts.h2.train."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from scripts.h2 import train as mod
from tests.fakes import HashingPairTokenizer, ScoringModel
from transformers import get_scheduler

from fsr.corpus.layout import NEGATIVES_NAME, split_dir
from fsr.formats import FORMAT_NAMES
from fsr.h2_layout import ADAPTER_NAME, train_dir
from fsr.models.adapters import AdaptedModel
from fsr.models.registry import by_slug
from fsr.reporting import parse_progress
from fsr.training.checkpoint import CHECKPOINT_NAME, save_checkpoint
from fsr.training.log import LOG_NAME, STEP, TrainLog, events, read_log

SLUG = "minilm_l6"
DEFAULT_ARM = "5fmt_lam1"


def record(rec_id):
    return {
        "id": str(rec_id),
        "title": f"Article {rec_id}",
        "question": f"who built bridge {rec_id}",
        "pairs": [["Born", "1946"], ["Role", "Engineer"]],
        "body": f"The bridge {rec_id} opened in 1946 and carries the road. " * 4,
        "quality_flags": [],
    }


@pytest.fixture
def data_root(tmp_path):
    root = tmp_path / "nq"
    splits = split_dir(root)
    splits.mkdir(parents=True)
    ids = {
        "train": [f"t{i}" for i in range(8)],
        "dev": [f"d{i}" for i in range(4)],
        "test": [f"s{i}" for i in range(2)],
        "nq_val": [f"v{i}" for i in range(2)],
    }
    for name, names in ids.items():
        (splits / f"{name}.json").write_text(
            json.dumps({"records": [record(i) for i in names]})
        )
    pool = ids["train"] + ids["dev"] + ids["test"]
    (splits / NEGATIVES_NAME).write_text(
        json.dumps(
            {
                "negatives": {
                    rid: [other for other in pool if other != rid][:4] for rid in pool
                }
            }
        )
    )
    return root


@pytest.fixture
def fake_model(monkeypatch):
    """Replace the loaders, so no download and no device are needed."""
    seen: dict = {"saved": []}

    def fake_build(model_id, device, **kwargs):
        seen["model_id"] = model_id
        seen["device"] = device
        seen.update(kwargs)
        torch.manual_seed(0)
        model = ScoringModel(4)
        model.save_pretrained = seen["saved"].append
        seen["model"] = model
        return AdaptedModel(
            model=model,
            jina_patched=0,
            trainable=5,
            total=100,
            gradient_checkpointing=kwargs.get("gradient_checkpointing", True),
        )

    monkeypatch.setattr(mod, "build_adapted_model", fake_build)
    monkeypatch.setattr(mod, "load_tokenizer", lambda _id: HashingPairTokenizer(4))
    return seen


def run(monkeypatch, data_root, *extra, model=SLUG):
    argv = [
        "train.py",
        "--model",
        model,
        "--data-root",
        str(data_root),
        "--max-steps",
        "4",
        "--eval-every",
        "2",
        "--warmup-steps",
        "1",
        "--physical-batch",
        "2",
        "--grad-accum",
        "1",
        "--neg-k",
        "2",
        *extra,
    ]
    monkeypatch.setattr("sys.argv", argv)
    mod.main()


def log_of(data_root, arm_name=DEFAULT_ARM, slug=SLUG):
    return read_log(train_dir(data_root, slug, arm_name) / LOG_NAME)


class TestResolveModel:
    def test_accepts_a_slug(self):
        assert mod.resolve_model("bge_base").slug == "bge_base"

    def test_accepts_an_identifier(self):
        assert mod.resolve_model("BAAI/bge-reranker-base").slug == "bge_base"

    def test_rejects_an_unknown_name(self):
        with pytest.raises(SystemExit, match="unknown model 'nope'"):
            mod.resolve_model("nope")

    def test_rejects_an_unregistered_identifier(self):
        with pytest.raises(SystemExit, match="unknown model"):
            mod.resolve_model("some/other-model")

    def test_names_the_known_slugs_when_it_refuses(self):
        with pytest.raises(SystemExit, match="minilm_l6"):
            mod.resolve_model("nope")


class TestTrainingFormats:
    def test_keeps_every_format_when_none_is_held_out(self):
        assert mod.training_formats(None) == list(FORMAT_NAMES)

    def test_drops_the_held_out_format(self):
        assert "yaml" not in mod.training_formats("yaml")

    def test_keeps_the_published_order(self):
        assert mod.training_formats("toml") == [f for f in FORMAT_NAMES if f != "toml"]

    def test_leaves_four_formats_for_a_fold(self):
        assert len(mod.training_formats("markdown")) == len(FORMAT_NAMES) - 1


class TestDefaults:
    def test_uses_the_published_step_budget(self):
        args = mod.build_parser().parse_args(["--model", SLUG])
        assert args.max_steps == mod.DEFAULT_MAX_STEPS == 500
        assert args.warmup_steps == mod.DEFAULT_WARMUP_STEPS == 30
        assert args.eval_every == mod.DEFAULT_EVAL_EVERY == 50

    def test_uses_the_published_learning_rate(self):
        args = mod.build_parser().parse_args(["--model", SLUG])
        assert args.lr == 2e-4
        assert args.weight_decay == 0.01

    def test_writes_no_checkpoint_by_default(self):
        assert mod.build_parser().parse_args(["--model", SLUG]).checkpoint_every == 0

    def test_trains_on_every_format_by_default(self):
        args = mod.build_parser().parse_args(["--model", SLUG])
        assert args.held_out_format == mod.HELD_OUT_NONE

    @pytest.mark.usefixtures("fake_model")
    def test_takes_the_geometry_from_the_registry(self, monkeypatch, data_root):
        monkeypatch.setattr(
            "sys.argv",
            [
                "train.py",
                "--model",
                SLUG,
                "--data-root",
                str(data_root),
                "--max-steps",
                "2",
                "--eval-every",
                "2",
                "--warmup-steps",
                "1",
                "--neg-k",
                "2",
            ],
        )
        mod.main()
        entry = log_of(data_root)[0]
        assert entry["physical_batch"] == by_slug(SLUG).physical_batch
        assert entry["grad_accum"] == by_slug(SLUG).grad_accum


class TestRun:
    def test_saves_the_adapter_in_the_arm_directory(
        self, monkeypatch, data_root, fake_model
    ):
        run(monkeypatch, data_root)
        assert fake_model["saved"] == [
            train_dir(data_root, SLUG, DEFAULT_ARM) / ADAPTER_NAME
        ]

    @pytest.mark.usefixtures("fake_model")
    def test_names_the_arm_from_the_fold_and_the_weight(self, monkeypatch, data_root):
        run(monkeypatch, data_root, "--held-out-format", "yaml", "--lambda-inv", "0.1")
        assert (train_dir(data_root, SLUG, "yaml_lam0.1") / LOG_NAME).exists()

    @pytest.mark.usefixtures("fake_model")
    def test_writes_a_run_record_first(self, monkeypatch, data_root):
        run(monkeypatch, data_root, "--lambda-inv", "0.01")
        records = log_of(data_root, "5fmt_lam0.01")
        assert records[0]["event"] == "run"
        assert records[0]["model"] == SLUG
        assert records[0]["lambda_inv"] == 0.01

    @pytest.mark.usefixtures("fake_model")
    def test_the_run_record_names_the_training_formats(self, monkeypatch, data_root):
        run(monkeypatch, data_root, "--held-out-format", "json")
        assert "json" not in log_of(data_root, "json_lam1")[0]["formats"]

    @pytest.mark.usefixtures("fake_model")
    def test_runs_the_step_budget(self, monkeypatch, data_root):
        run(monkeypatch, data_root)
        assert log_of(data_root)[-1]["step"] == 4

    def test_passes_the_registry_targets_to_the_loader(
        self, monkeypatch, data_root, fake_model
    ):
        run(monkeypatch, data_root)
        assert fake_model["lora_targets"] == by_slug(SLUG).lora_targets

    def test_an_explicit_target_list_wins(self, monkeypatch, data_root, fake_model):
        run(monkeypatch, data_root, "--lora-target-modules", "Wqkv")
        assert fake_model["lora_targets"] == ["Wqkv"]

    def test_passes_the_registry_head_choice_to_the_loader(
        self, monkeypatch, data_root, fake_model
    ):
        run(monkeypatch, data_root)
        assert fake_model["tanh_head"] is False

    def test_an_explicit_head_flag_wins(self, monkeypatch, data_root, fake_model):
        run(monkeypatch, data_root, "--tanh-head")
        assert fake_model["tanh_head"] is True

    def test_the_tanh_variant_needs_no_flag(self, monkeypatch, data_root, fake_model):
        run(monkeypatch, data_root, model="mxbai_v1_tanh")
        assert fake_model["tanh_head"] is True

    def test_keeps_activations_on_request(self, monkeypatch, data_root, fake_model):
        run(monkeypatch, data_root, "--no-grad-checkpoint")
        assert fake_model["gradient_checkpointing"] is False

    def test_requests_the_eager_kernel_on_request(
        self, monkeypatch, data_root, fake_model
    ):
        run(monkeypatch, data_root, "--eager-attn")
        assert fake_model["eager_attn"] is True

    def test_applies_an_explicit_rank(self, monkeypatch, data_root, fake_model):
        run(monkeypatch, data_root, "--lora-rank", "8", "--lora-alpha", "16")
        assert fake_model["rank"] == 8
        assert fake_model["alpha"] == 16

    def test_omits_the_rank_when_none_is_given(
        self, monkeypatch, data_root, fake_model
    ):
        run(monkeypatch, data_root)
        assert "rank" not in fake_model

    @pytest.mark.usefixtures("fake_model")
    def test_records_the_rank_in_the_arm_when_asked(self, monkeypatch, data_root):
        run(monkeypatch, data_root, "--lora-rank", "8", "--rank-tag", "8")
        assert (train_dir(data_root, SLUG, "5fmt_lam1_r8") / LOG_NAME).exists()

    @pytest.mark.usefixtures("fake_model")
    def test_reports_a_patched_fused_projection(self, monkeypatch, data_root, capsys):
        from fsr.models.adapters import AdaptedModel

        plain = mod.build_adapted_model

        def patched(*a, **k):
            built = plain(*a, **k)
            return AdaptedModel(
                model=built.model,
                jina_patched=3,
                trainable=built.trainable,
                total=built.total,
                gradient_checkpointing=built.gradient_checkpointing,
            )

        monkeypatch.setattr(mod, "build_adapted_model", patched)
        run(monkeypatch, data_root)
        assert "fused projections patched: 3" in capsys.readouterr().out

    @pytest.mark.usefixtures("fake_model")
    def test_caps_the_records_on_request(self, monkeypatch, data_root):
        run(monkeypatch, data_root, "--limit-train", "4", "--limit-dev", "2")
        assert log_of(data_root)[0]["train_records"] <= 4


class TestSkipAndForce:
    @pytest.mark.usefixtures("fake_model")
    def test_skips_a_finished_arm(self, monkeypatch, data_root, capsys):
        (train_dir(data_root, SLUG, DEFAULT_ARM) / ADAPTER_NAME).mkdir(parents=True)
        run(monkeypatch, data_root)
        assert "Skipping" in capsys.readouterr().out

    @pytest.mark.usefixtures("fake_model")
    def test_writes_nothing_when_it_skips(self, monkeypatch, data_root):
        (train_dir(data_root, SLUG, DEFAULT_ARM) / ADAPTER_NAME).mkdir(parents=True)
        run(monkeypatch, data_root)
        assert not (train_dir(data_root, SLUG, DEFAULT_ARM) / LOG_NAME).exists()

    @pytest.mark.usefixtures("fake_model")
    def test_force_trains_over_a_finished_arm(self, monkeypatch, data_root):
        (train_dir(data_root, SLUG, DEFAULT_ARM) / ADAPTER_NAME).mkdir(parents=True)
        run(monkeypatch, data_root, "--force")
        assert (train_dir(data_root, SLUG, DEFAULT_ARM) / LOG_NAME).exists()


class TestFreshStart:
    @pytest.mark.usefixtures("fake_model")
    def test_discards_a_log_from_an_earlier_run(self, monkeypatch, data_root):
        run(monkeypatch, data_root)
        first = len(log_of(data_root))
        run(monkeypatch, data_root, "--force")
        assert len(log_of(data_root)) == first

    @pytest.mark.usefixtures("fake_model")
    def test_the_second_run_has_each_step_once(self, monkeypatch, data_root):
        run(monkeypatch, data_root)
        run(monkeypatch, data_root, "--force")
        steps = [r["step"] for r in events(log_of(data_root), "step")]
        assert steps == sorted(set(steps))


class TestCheckpointing:
    @pytest.mark.usefixtures("fake_model")
    def test_writes_no_checkpoint_by_default(self, monkeypatch, data_root):
        run(monkeypatch, data_root)
        assert not (train_dir(data_root, SLUG, DEFAULT_ARM) / CHECKPOINT_NAME).exists()

    @pytest.mark.usefixtures("fake_model")
    def test_removes_the_checkpoint_once_the_run_finishes(self, monkeypatch, data_root):
        run(monkeypatch, data_root, "--checkpoint-every", "2")
        assert not (train_dir(data_root, SLUG, DEFAULT_ARM) / CHECKPOINT_NAME).exists()

    @pytest.mark.usefixtures("fake_model")
    def test_resume_without_a_checkpoint_starts_from_the_beginning(
        self, monkeypatch, data_root
    ):
        run(monkeypatch, data_root, "--resume")
        assert log_of(data_root)[0]["event"] == "run"


class TestResume:
    def seed_checkpoint(self, data_root, step):
        """Leave the state a run killed at `step` would have left behind."""
        run_dir = train_dir(data_root, SLUG, DEFAULT_ARM)
        model = ScoringModel(4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
        scheduler = get_scheduler(
            "cosine", optimizer=optimizer, num_warmup_steps=1, num_training_steps=4
        )
        save_checkpoint(
            run_dir / CHECKPOINT_NAME,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=step,
            sampler=np.random.default_rng(0),
        )
        with TrainLog(run_dir / LOG_NAME) as log:
            log.write("run", model=SLUG)
            for n in range(1, step + 3):
                log.write(STEP, step=n, loss=1.0)
        return run_dir

    @pytest.mark.usefixtures("fake_model")
    def test_reports_the_step_it_resumes_at(self, monkeypatch, data_root, capsys):
        self.seed_checkpoint(data_root, 2)
        run(monkeypatch, data_root, "--resume", "--checkpoint-every", "2")
        assert "Resuming at step 2" in capsys.readouterr().out

    @pytest.mark.usefixtures("fake_model")
    def test_drops_the_log_records_past_the_checkpoint(
        self, monkeypatch, data_root, capsys
    ):
        self.seed_checkpoint(data_root, 2)
        run(monkeypatch, data_root, "--resume", "--checkpoint-every", "2")
        assert "dropped 2 log records" in capsys.readouterr().out

    @pytest.mark.usefixtures("fake_model")
    def test_the_resumed_log_has_each_step_once(self, monkeypatch, data_root):
        self.seed_checkpoint(data_root, 2)
        run(monkeypatch, data_root, "--resume", "--checkpoint-every", "2")
        steps = [r["step"] for r in events(log_of(data_root), "step")]
        assert steps == sorted(set(steps))

    @pytest.mark.usefixtures("fake_model")
    def test_writes_no_second_run_record(self, monkeypatch, data_root):
        self.seed_checkpoint(data_root, 2)
        run(monkeypatch, data_root, "--resume", "--checkpoint-every", "2")
        assert len(events(log_of(data_root), "run")) == 1

    @pytest.mark.usefixtures("fake_model")
    def test_finishes_the_remaining_steps(self, monkeypatch, data_root):
        self.seed_checkpoint(data_root, 2)
        run(monkeypatch, data_root, "--resume", "--checkpoint-every", "2")
        assert log_of(data_root)[-1]["step"] == 4

    def test_saves_the_adapter_after_resuming(self, monkeypatch, data_root, fake_model):
        self.seed_checkpoint(data_root, 2)
        run(monkeypatch, data_root, "--resume", "--checkpoint-every", "2")
        assert fake_model["saved"] == [
            train_dir(data_root, SLUG, DEFAULT_ARM) / ADAPTER_NAME
        ]

    @pytest.mark.usefixtures("fake_model")
    def test_ignores_a_checkpoint_without_the_flag(
        self, monkeypatch, data_root, capsys
    ):
        self.seed_checkpoint(data_root, 2)
        run(monkeypatch, data_root)
        assert "Resuming" not in capsys.readouterr().out


class TestProgress:
    @pytest.mark.usefixtures("fake_model")
    def test_writes_no_progress_file_by_default(self, monkeypatch, data_root, tmp_path):
        run(monkeypatch, data_root)
        assert not list(tmp_path.glob("*.progress"))

    @pytest.mark.usefixtures("fake_model")
    def test_writes_the_progress_file_it_is_given(
        self, monkeypatch, data_root, tmp_path
    ):
        path = tmp_path / "job.progress"
        run(monkeypatch, data_root, "--progress-file", str(path))
        fields = parse_progress(path.read_text())
        assert fields["done"] == fields["total"] == "4"
        assert fields["failed"] == "0"

    @pytest.mark.usefixtures("fake_model")
    def test_marks_the_job_failed_when_the_loop_raises(
        self, monkeypatch, data_root, tmp_path
    ):
        path = tmp_path / "job.progress"

        def boom(*_a, **_k):
            raise RuntimeError("out of memory")

        monkeypatch.setattr(mod, "train", boom)
        with pytest.raises(RuntimeError, match="out of memory"):
            run(monkeypatch, data_root, "--progress-file", str(path))
        assert parse_progress(path.read_text())["failed"] == "1"

    def test_saves_no_adapter_when_the_loop_raises(
        self, monkeypatch, data_root, fake_model
    ):
        def boom(*_a, **_k):
            raise RuntimeError("out of memory")

        monkeypatch.setattr(mod, "train", boom)
        with pytest.raises(RuntimeError):
            run(monkeypatch, data_root)
        assert fake_model["saved"] == []


class TestEmptyCorpus:
    @pytest.mark.usefixtures("fake_model")
    def test_stops_when_no_record_survives(self, monkeypatch, data_root):
        monkeypatch.setattr(mod, "prepare_train_records", lambda *_a, **_k: ([], 8))
        with pytest.raises(SystemExit, match="nothing to train on"):
            run(monkeypatch, data_root)

    @pytest.mark.usefixtures("fake_model")
    def test_stops_when_no_development_record_survives(self, monkeypatch, data_root):
        monkeypatch.setattr(mod, "prepare_dev_records", lambda *_a, **_k: [])
        with pytest.raises(SystemExit, match="nothing to train on"):
            run(monkeypatch, data_root)


class TestSeeding:
    @pytest.mark.usefixtures("fake_model")
    def test_two_runs_of_one_arm_agree(self, monkeypatch, data_root):
        run(monkeypatch, data_root)
        first = [r["loss"] for r in events(log_of(data_root), "step")]
        run(monkeypatch, data_root, "--force")
        second = [r["loss"] for r in events(log_of(data_root), "step")]
        assert first == second

    @pytest.mark.usefixtures("fake_model")
    def test_the_sampler_follows_the_seed(self, monkeypatch, data_root):
        run(monkeypatch, data_root, "--seed", "0")
        with_zero = [r["loss"] for r in events(log_of(data_root), "step")]
        run(monkeypatch, data_root, "--force", "--seed", "7")
        with_seven = [r["loss"] for r in events(log_of(data_root), "step")]
        assert with_zero != with_seven
