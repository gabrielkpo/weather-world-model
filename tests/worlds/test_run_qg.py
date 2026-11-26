import json
from pathlib import Path

import numpy as np
import pytest

from weather_world_model.worlds.qg import DIAGNOSTIC_NAMES
from weather_world_model.worlds.run_qg import (
    _ridge,
    clips,
    main,
    r2_score,
    split_sizes,
)


def test_split_sizes_are_70_15_15_and_non_empty() -> None:
    assert split_sizes(80) == (56, 12, 12)
    assert split_sizes(3) == (1, 1, 1)
    with pytest.raises(ValueError, match="three runs"):
        split_sizes(2)


def test_clips_use_stride() -> None:
    seqs = np.arange(2 * 10 * 1, dtype=np.float64).reshape(2, 10, 1)
    out = clips(seqs, 4, 3)
    assert out.shape == (2 * 3, 4, 1)
    assert np.array_equal(out[0, :, 0], [0, 1, 2, 3])
    assert np.array_equal(out[2, :, 0], [3, 4, 5, 6])


def test_ridge_recovers_linear_map() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((200, 5))
    y = x @ np.arange(1.0, 6.0) + 2.0
    w = _ridge(x, y, 1e-6)
    pred = np.hstack([x, np.ones((200, 1))]) @ w
    assert r2_score(y, pred) > 0.999
    assert np.isnan(r2_score(np.ones(3), np.ones(3)))


def test_smoke_profile_writes_record(tmp_path: Path) -> None:
    main(["--profile", "smoke", "--out", str(tmp_path)])
    record = json.loads((tmp_path / "qg-smoke.json").read_text())
    assert record["profile"] == "smoke"
    assert record["split_runs"] == {"train": 5, "val": 1, "test": 1}
    seed = record["seeds"][0]
    for model in ("k1", "k8", "random"):
        assert set(seed[model]["readouts"]) == set(DIAGNOSTIC_NAMES)
    for model in ("k1", "k8"):
        assert 0.0 <= seed[model]["arrow_accuracy"] <= 1.0
        assert "eps0.001" in seed[model]["predictability"]
        assert isinstance(seed[model]["collapsed"], bool)
    verdicts = record["verdicts"]
    assert set(verdicts) == {"W-A", "W-C", "W-P", "any_collapse"}
    assert isinstance(verdicts["W-A"]["supported"], bool)
    assert 0.0 <= verdicts["W-C"]["oracle_accuracy"] <= 1.0


def test_smoke_profile_resumes_from_partial(tmp_path: Path) -> None:
    from weather_world_model.worlds.run_qg import run

    first = run("smoke")
    partial = tmp_path / "qg-smoke.partial.json"
    partial.write_text(json.dumps({k: v for k, v in first.items() if k != "verdicts"}))
    resumed = run("smoke", partial=partial)
    assert resumed["resumed"] is True
    assert resumed["seeds"] == first["seeds"]
