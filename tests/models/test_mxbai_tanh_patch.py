"""Tests for fsr.models.mxbai_tanh_patch."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from fsr.models.mxbai_tanh_patch import apply_mxbai_tanh_patch

IN_FEATURES = 16


class Model(nn.Module):
    """A model stub carrying a single-Linear reranker head."""

    def __init__(self, in_features=IN_FEATURES, out_features=1):
        """Build the classifier head."""
        super().__init__()
        self.classifier = nn.Linear(in_features, out_features)


def patched_model(seed: int = 0) -> Model:
    """Return a patched model."""
    torch.manual_seed(seed)
    model = Model()
    apply_mxbai_tanh_patch(model)
    return model


class TestStructure:
    def test_builds_a_dense_tanh_projection_head(self):
        head = patched_model().classifier
        assert isinstance(head, nn.Sequential)
        assert [type(layer) for layer in head] == [nn.Linear, nn.Tanh, nn.Linear]

    def test_the_dense_layer_is_square(self):
        dense = patched_model().classifier[0]
        assert (dense.in_features, dense.out_features) == (IN_FEATURES, IN_FEATURES)

    def test_the_output_projection_keeps_the_original_shape(self):
        out_proj = patched_model().classifier[2]
        assert (out_proj.in_features, out_proj.out_features) == (IN_FEATURES, 1)

    def test_both_added_layers_have_a_bias(self):
        head = patched_model().classifier
        assert head[0].bias is not None
        assert head[2].bias is not None


class TestWeightTransfer:
    def test_the_output_projection_copies_the_original_weight_and_bias(self):
        torch.manual_seed(0)
        model = Model()
        original_weight = model.classifier.weight.detach().clone()
        original_bias = model.classifier.bias.detach().clone()
        apply_mxbai_tanh_patch(model)
        assert torch.equal(model.classifier[2].weight, original_weight)
        assert torch.equal(model.classifier[2].bias, original_bias)

    def test_the_copy_is_not_the_original_tensor(self):
        torch.manual_seed(0)
        model = Model()
        original = model.classifier.weight
        apply_mxbai_tanh_patch(model)
        assert model.classifier[2].weight is not original

    def test_the_dense_layer_is_not_an_identity(self):
        dense = patched_model().classifier[0]
        assert not torch.allclose(dense.weight, torch.eye(IN_FEATURES), atol=1e-3)


class TestDtypeAndDevice:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_both_layers_match_the_original_dtype(self, dtype):
        torch.manual_seed(0)
        model = Model()
        model.classifier = model.classifier.to(dtype=dtype)
        apply_mxbai_tanh_patch(model)
        assert model.classifier[0].weight.dtype == dtype
        assert model.classifier[2].weight.dtype == dtype

    def test_both_layers_match_the_original_device(self):
        model = patched_model()
        assert model.classifier[0].weight.device == torch.device("cpu")
        assert model.classifier[2].weight.device == torch.device("cpu")


class TestForward:
    def test_the_head_maps_hidden_states_to_one_score(self):
        head = patched_model().classifier
        scores = head(torch.randn(4, IN_FEATURES))
        assert scores.shape == (4, 1)

    def test_the_head_output_is_not_the_unpatched_output(self):
        torch.manual_seed(0)
        model = Model()
        x = torch.randn(4, IN_FEATURES)
        before = model.classifier(x)
        apply_mxbai_tanh_patch(model)
        assert not torch.allclose(before, model.classifier(x), atol=1e-6)


class TestRejections:
    def test_rejects_a_model_with_no_classifier(self):
        with pytest.raises(RuntimeError, match="no `classifier`"):
            apply_mxbai_tanh_patch(nn.Module())

    def test_rejects_a_classifier_that_is_not_linear(self):
        model = Model()
        model.classifier = nn.Sequential(nn.Linear(IN_FEATURES, 1))
        with pytest.raises(RuntimeError, match=r"expected nn\.Linear"):
            apply_mxbai_tanh_patch(model)

    def test_rejects_a_multi_output_classifier(self):
        with pytest.raises(RuntimeError, match="out_features=1"):
            apply_mxbai_tanh_patch(Model(out_features=2))

    def test_rejects_a_second_patch(self):
        model = patched_model()
        with pytest.raises(RuntimeError, match=r"expected nn\.Linear"):
            apply_mxbai_tanh_patch(model)
