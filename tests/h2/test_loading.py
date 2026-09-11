"""Tests for fsr.h2.loading."""

from __future__ import annotations

import pytest
import torch

from fsr.h2 import loading as mod


class Loaded:
    """A model stub that records the device and the eval call."""

    def __init__(self) -> None:
        """Start with no recorded device and no eval call."""
        self.device = None
        self.evaluated = False

    def to(self, device):
        """Record the device and return the model."""
        self.device = device
        return self

    def eval(self):
        """Record the eval call and return the model."""
        self.evaluated = True
        return self


@pytest.fixture
def capture(monkeypatch):
    """Capture the keyword arguments passed to the model loader."""
    seen: dict = {}

    def loader(name, **kwargs):
        seen["name"] = name
        seen.update(kwargs)
        return Loaded()

    monkeypatch.setattr(
        mod.AutoModelForSequenceClassification, "from_pretrained", loader
    )
    return seen


class TestLoadTokenizer:
    def test_returns_the_tokenizer(self, monkeypatch):
        monkeypatch.setattr(
            mod.AutoTokenizer, "from_pretrained", lambda *_a, **_k: "TOKENIZER"
        )
        assert mod.load_tokenizer("some/model") == "TOKENIZER"

    @pytest.mark.parametrize("trust", [True, False])
    def test_forwards_the_trust_flag(self, monkeypatch, trust):
        seen = {}

        def loader(_name, **kwargs):
            seen.update(kwargs)
            return "TOKENIZER"

        monkeypatch.setattr(mod.AutoTokenizer, "from_pretrained", loader)
        mod.load_tokenizer("some/model", trust_remote_code=trust)
        assert seen["trust_remote_code"] is trust

    def test_trusts_remote_code_by_default(self, monkeypatch):
        seen = {}

        def loader(_name, **kwargs):
            seen.update(kwargs)
            return "TOKENIZER"

        monkeypatch.setattr(mod.AutoTokenizer, "from_pretrained", loader)
        mod.load_tokenizer("some/model")
        assert seen["trust_remote_code"] is True


class TestLoadModel:
    @pytest.mark.usefixtures("capture")
    def test_moves_the_model_to_the_device_and_evaluates(self):
        model = mod.load_model("some/model", "cpu")
        assert model.device == "cpu"
        assert model.evaluated is True

    def test_loads_in_float32_by_default(self, capture):
        mod.load_model("some/model", "cpu")
        assert capture["torch_dtype"] == "float32"

    @pytest.mark.parametrize(
        "dtype,expected",
        [
            (torch.float32, "float32"),
            (torch.bfloat16, "bfloat16"),
            (torch.float16, "float16"),
        ],
    )
    def test_converts_a_torch_dtype_to_a_string(self, capture, dtype, expected):
        mod.load_model("some/model", "cpu", dtype=dtype)
        assert capture["torch_dtype"] == expected

    def test_passes_a_string_dtype_through(self, capture):
        mod.load_model("some/model", "cpu", dtype="bfloat16")
        assert capture["torch_dtype"] == "bfloat16"

    def test_omits_the_attention_argument_by_default(self, capture):
        mod.load_model("some/model", "cpu")
        assert "attn_implementation" not in capture

    def test_requests_eager_attention_when_asked(self, capture):
        mod.load_model("some/model", "cpu", eager_attn=True)
        assert capture["attn_implementation"] == "eager"

    def test_forwards_the_trust_flag(self, capture):
        mod.load_model("some/model", "cpu", trust_remote_code=False)
        assert capture["trust_remote_code"] is False

    def test_retries_without_eager_attention_on_a_type_error(self, monkeypatch):
        calls = []

        def loader(_name, **kwargs):
            calls.append(dict(kwargs))
            if "attn_implementation" in kwargs:
                raise TypeError("unexpected keyword")
            return Loaded()

        monkeypatch.setattr(
            mod.AutoModelForSequenceClassification, "from_pretrained", loader
        )
        model = mod.load_model("some/model", "cpu", eager_attn=True)
        assert model.evaluated is True
        assert len(calls) == 2
        assert "attn_implementation" not in calls[1]

    @pytest.mark.usefixtures("capture")
    def test_does_not_load_an_adapter_by_default(self, monkeypatch):
        def fail(*_args, **_kwargs):
            raise AssertionError("PeftModel.from_pretrained must not be called")

        monkeypatch.setattr(mod.PeftModel, "from_pretrained", fail)
        mod.load_model("some/model", "cpu")

    @pytest.mark.usefixtures("capture")
    def test_loads_and_patches_an_adapter(self, monkeypatch):
        seen = {}

        def from_pretrained(model, path):
            seen["path"] = path
            return model

        monkeypatch.setattr(mod.PeftModel, "from_pretrained", from_pretrained)
        monkeypatch.setattr(
            mod, "patch_jina_lora", lambda _m: seen.setdefault("patched", True)
        )
        model = mod.load_model("some/model", "cpu", lora_adapter_path="adapters/mxbai")
        assert seen["path"] == "adapters/mxbai"
        assert seen["patched"] is True
        assert model.evaluated is True
