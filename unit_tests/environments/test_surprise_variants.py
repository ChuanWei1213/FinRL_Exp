from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from finrl.meta.rewards import IntrinsicRewardConfig
from finrl.meta.rewards import PaperFaithfulIntrinsicRewardController
from finrl.meta.rewards import PaperFaithfulSurpriseModel
from finrl.meta.rewards import RobustIntrinsicRewardController


def config(**overrides) -> IntrinsicRewardConfig:
    values = {
        "total_timesteps": 32,
        "alpha": 1.0,
        "beta": 0.0,
        "warmup_steps": 0,
        "batch_size": 2,
        "replay_capacity": 32,
        "latent_dim": 2,
        "volatility_window": 3,
        "seed": 7,
        "device": "cpu",
    }
    values.update(overrides)
    return IntrinsicRewardConfig(**values)


def set_unit_gaussian(model: PaperFaithfulSurpriseModel) -> None:
    with torch.no_grad():
        for parameter in model.predictor.parameters():
            parameter.zero_()
        model.mean_head.weight.zero_()
        model.mean_head.bias.zero_()
        model.std_head[0].weight.zero_()
        desired_softplus = 1.0 - model.std_min
        inverse_softplus = math.log(math.expm1(desired_softplus))
        model.std_head[0].bias.fill_(inverse_softplus)


def test_paper_faithful_surprise_sums_gaussian_nll_dimensions():
    model = PaperFaithfulSurpriseModel(
        observation_dim=2,
        action_dim=1,
        device=torch.device("cpu"),
    )
    set_unit_gaussian(model)
    bonus = model.compute_bonus(
        np.zeros(2, dtype=np.float32),
        np.zeros(1, dtype=np.float32),
        np.array([1.0, -1.0], dtype=np.float32),
    )
    per_dimension = 0.5 * (math.log(2.0 * math.pi) + 1.0)
    assert bonus == pytest.approx(2.0 * per_dimension, rel=1e-5)


def test_paper_faithful_controller_does_not_rectify_negative_nll():
    controller = PaperFaithfulIntrinsicRewardController(2, 1, config())
    controller.observe_initial(np.zeros(2, dtype=np.float32))
    controller.surprise_model.compute_bonus = lambda *_: -2.0

    reward, info = controller.process_transition(
        np.zeros(2, dtype=np.float32),
        np.zeros(1, dtype=np.float32),
        np.ones(2, dtype=np.float32),
        0.25,
        [100.0, 102.0, 99.0, 104.0],
    )

    assert info["reward_surprise_raw"] == -2.0
    assert info["reward_surprise"] == -2.0
    assert info["reward_intrinsic"] < 0.0
    assert reward < 0.25


def test_robust_surprise_uses_positive_running_z_and_clipping():
    controller = RobustIntrinsicRewardController(
        2,
        1,
        config(bonus_clip=3.0),
    )
    first_bonus, first_info = controller._transform_surprise(2.0)
    assert first_bonus == pytest.approx(2.0)
    assert first_info["intrinsic_surprise_center"] == 0.0

    controller._update_surprise_statistics(2.0)
    repeated_bonus, repeated_info = controller._transform_surprise(2.0)
    lower_bonus, lower_info = controller._transform_surprise(1.0)
    upper_bonus, upper_info = controller._transform_surprise(10.0)

    assert repeated_bonus < 0.05
    assert lower_bonus == 0.0
    assert lower_info["intrinsic_surprise_z"] < 0.0
    assert upper_bonus == 3.0
    assert upper_info["intrinsic_surprise_z"] > 3.0
    assert repeated_info["intrinsic_surprise_std"] > 0.0


def test_robust_surprise_reference_checkpoint_round_trip(tmp_path):
    source = RobustIntrinsicRewardController(2, 1, config())
    source.observe_initial(np.zeros(2, dtype=np.float32))
    source.surprise_model.compute_bonus = lambda *_: 1.5
    source.process_transition(
        np.zeros(2, dtype=np.float32),
        np.zeros(1, dtype=np.float32),
        np.ones(2, dtype=np.float32),
        0.25,
        [100.0, 102.0, 99.0, 104.0],
    )
    checkpoint = tmp_path / "robust.pt"
    source.save(checkpoint)

    restored = RobustIntrinsicRewardController(2, 1, config())
    restored.load(checkpoint)
    assert restored.global_step == source.global_step
    assert restored.surprise_reference.count == pytest.approx(
        source.surprise_reference.count
    )
    np.testing.assert_allclose(
        restored.surprise_reference.mean,
        source.surprise_reference.mean,
    )
