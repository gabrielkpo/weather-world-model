import pytest
import torch

from weather_world_model.seeding import seed_everything


def test_same_seed_gives_same_draws() -> None:
    a = seed_everything(3).normal(size=4)
    ta = torch.randn(4)
    b = seed_everything(3).normal(size=4)
    tb = torch.randn(4)
    assert (a == b).all()
    assert torch.equal(ta, tb)


def test_negative_seed_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        seed_everything(-1)
