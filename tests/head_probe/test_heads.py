"""Tests for fsr.head_probe.heads."""

from __future__ import annotations

import pytest
import torch

from fsr.head_probe.heads import (
    DEEP,
    HEAD_NAMES,
    LINEAR,
    TANH,
    WIDE_LINEAR,
    build_head,
    is_affine,
    parameter_count,
)

DIM = 8


class TestBuildHead:
    @pytest.mark.parametrize("name", HEAD_NAMES)
    def test_every_head_maps_a_representation_to_one_score(self, name):
        head = build_head(name, DIM)
        assert head(torch.zeros(3, DIM)).shape == (3, 1)

    @pytest.mark.parametrize("name", HEAD_NAMES)
    def test_the_same_seed_gives_the_same_parameters(self, name):
        first = build_head(name, DIM, seed=7)
        second = build_head(name, DIM, seed=7)
        for a, b in zip(first.parameters(), second.parameters(), strict=True):
            assert torch.equal(a, b)

    @pytest.mark.parametrize("name", HEAD_NAMES)
    def test_a_different_seed_gives_different_parameters(self, name):
        first = build_head(name, DIM, seed=1)
        second = build_head(name, DIM, seed=2)
        assert any(
            not torch.equal(a, b)
            for a, b in zip(first.parameters(), second.parameters(), strict=True)
        )

    def test_leaves_the_global_generator_where_it_found_it(self):
        torch.manual_seed(123)
        before = torch.rand(4)
        torch.manual_seed(123)
        build_head(TANH, DIM, seed=99)
        assert torch.equal(torch.rand(4), before)

    def test_refuses_an_unknown_head(self):
        with pytest.raises(ValueError, match="unknown head"):
            build_head("mystery", DIM)


class TestCapacityControl:
    def test_the_wide_linear_head_matches_the_tanh_head_in_size(self):
        """The control only works when the only difference is the activation."""
        wide = parameter_count(build_head(WIDE_LINEAR, DIM))
        tanh = parameter_count(build_head(TANH, DIM))
        assert wide == tanh

    def test_the_plain_linear_head_is_much_smaller(self):
        assert parameter_count(build_head(LINEAR, DIM)) < parameter_count(
            build_head(TANH, DIM)
        )

    def test_the_deep_head_is_the_largest(self):
        counts = {n: parameter_count(build_head(n, DIM)) for n in HEAD_NAMES}
        assert counts[DEEP] == max(counts.values())

    def test_the_wide_linear_head_is_affine_like_the_linear_one(self):
        """Two stacked linear layers compose to one, so expressiveness is equal."""
        assert is_affine(LINEAR)
        assert is_affine(WIDE_LINEAR)

    def test_the_nonlinear_heads_are_not_affine(self):
        assert not is_affine(TANH)
        assert not is_affine(DEEP)

    def test_a_wide_linear_head_is_reproducible_by_one_matrix(self):
        head = build_head(WIDE_LINEAR, DIM, seed=3)
        x = torch.randn(5, DIM)
        first, second = head[0], head[1]
        folded = x @ (second.weight @ first.weight).T + (
            second.weight @ first.bias + second.bias
        )
        assert torch.allclose(head(x), folded, atol=1e-5)

    def test_refuses_an_unknown_head(self):
        with pytest.raises(ValueError, match="unknown head"):
            is_affine("mystery")


class TestParameterCount:
    def test_counts_only_trainable_parameters(self):
        head = build_head(TANH, DIM)
        head[0].weight.requires_grad_(False)
        assert parameter_count(head) < sum(p.numel() for p in head.parameters())
