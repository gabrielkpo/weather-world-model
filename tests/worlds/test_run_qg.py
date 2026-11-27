import json
from pathlib import Path

import numpy as np
import pytest

from weather_world_model.worlds.qg import DIAGNOSTIC_NAMES
from weather_world_model.worlds.run_qg import (
    HORIZONS,
    THRESHOLDED,
    clips,
    main,
    split_sizes,
    verdicts_amendment3,
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


def test_smoke_profile_writes_record(tmp_path: Path) -> None:
    main(["--profile", "smoke", "--out", str(tmp_path)])
    record = json.loads((tmp_path / "qg-stationary-base-smoke.json").read_text())
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
    assert 0.0 <= verdicts["W-C"]["oracle_accuracy_mirrored_control"] <= 1.0
    assert 0.0 <= seed["k1"]["arrow_accuracy_mirrored"] <= 1.0
    assert "accuracy_mirrored_control" in verdicts["W-C"]["k8"]
    assert record["stationarity"]["passes"] is None  # the smoke profile only reports it
    assert record["stationarity"]["relative_difference"] >= 0.0


def test_smoke_profile_resumes_from_partial(tmp_path: Path) -> None:
    from weather_world_model.worlds.run_qg import run

    first = run("smoke")
    partial = tmp_path / "qg-stationary-base-smoke.partial.json"
    partial.write_text(json.dumps({k: v for k, v in first.items() if k != "verdicts"}))
    resumed = run("smoke", partial=partial)
    assert resumed["resumed"] is True
    assert resumed["seeds"] == first["seeds"]


def test_amendment3_smoke_writes_record_and_reuses_checkpoints(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = ["--profile", "smoke", "--amendment3", "--out", str(tmp_path / "out")]
    args += ["--checkpoints", str(tmp_path / "ckpt")]
    main(args)
    record = json.loads((tmp_path / "out" / "qg-stationary-smoke.json").read_text())
    assert record["amendment3"]["noise_dim"] == 8
    seed = record["seeds"][0]
    for model in ("k1", "k8", "stoch"):
        readouts = seed[model]["rollout_readouts"]
        assert readouts["n_sampled_rollouts"] == (8 if model == "stoch" else 1)
        for h in HORIZONS:
            assert set(readouts[f"h{h}"]) == set(DIAGNOSTIC_NAMES)
            assert set(readouts[f"h{h}"]["enstrophy"]) == {"rollout_r2", "persistence_r2"}
        assert (tmp_path / "ckpt" / "qg-stationary-smoke" / f"seed0-{model}.pt").exists()
    assert "spread" in seed["stoch"]["predictability"]
    assert "spread" not in seed["k1"]["predictability"]
    assert "rollout_readouts" not in seed["random"]
    verdicts = record["verdicts"]
    assert set(verdicts) == {
        "W-R",
        "W-S",
        "stochastic_reported",
        "retrained_k1_k8",
        "any_collapse",
    }
    assert isinstance(verdicts["W-R"]["supported"], bool)
    assert isinstance(verdicts["W-S"]["supported"], bool)
    assert len(verdicts["W-R"]["models"]["k8"]["mean_gain_over_persistence"]["h10"]) == 1

    # A second run loads the saved weights and reproduces every measured value.
    capsys.readouterr()
    main(args)
    assert capsys.readouterr().err.count("loaded from") == 3
    again = json.loads((tmp_path / "out" / "qg-stationary-smoke.json").read_text())
    for model in ("k1", "k8", "stoch"):
        assert again["seeds"][0][model]["from_checkpoint"] is True
        for key in ("readouts", "rollout_readouts", "predictability", "arrow_accuracy"):
            assert again["seeds"][0][model][key] == seed[model][key]


def test_amendment3_verdict_rules() -> None:
    def model(gain: float, rate: float | None) -> dict[str, object]:
        quantity = {"rollout_r2": 0.5 + gain, "persistence_r2": 0.5}
        other = {"rollout_r2": 0.0, "persistence_r2": 0.9}  # unthresholded: must not count
        at_h = {q: (quantity if q in THRESHOLDED else other) for q in DIAGNOSTIC_NAMES}
        growth = {"growth_rate": rate}
        return {
            "rollout_readouts": {f"h{h}": at_h for h in HORIZONS},
            "predictability": {"eps0.001": growth, "spread": {"growth_rate": 0.4}},
            "arrow_accuracy": 0.75,
            "arrow_accuracy_mirrored": 0.5,
            "readouts": {q: {"test_r2": 0.6} for q in DIAGNOSTIC_NAMES},
            "collapsed": False,
        }

    truth = {"eps0.001": {"growth_rate": 1.0}}

    def seed(g1: float, g8: float, det: float | None, sens: float | None) -> dict[str, object]:
        return {
            "k1": model(g1, det),
            "k8": model(g8, 0.0),
            "stoch": model(0.0, sens),
            "random": {"readouts": {q: {"test_r2": 0.3} for q in DIAGNOSTIC_NAMES}},
        }

    good = verdicts_amendment3(truth, [seed(0.06, 0.2, 0.2, 0.9), seed(0.05, 0.1, None, 1.1)])
    assert good["W-R"]["supported"] and good["W-S"]["supported"]
    assert good["W-R"]["models"]["k1"]["mean_gain_over_persistence"]["h10"] == pytest.approx(
        [0.06, 0.05]
    )
    bad = verdicts_amendment3(truth, [seed(0.06, 0.04, 0.9, 0.8), seed(0.06, 0.2, 0.2, 0.6)])
    assert not bad["W-R"]["supported"]  # k8 below 0.05 on one seed
    assert not bad["W-S"]["supported"]  # not closer than k1 on seed 0, beyond 30 % on seed 1
    assert not verdicts_amendment3(truth, [seed(0.1, 0.1, 0.2, None)])["W-S"]["supported"]


def test_stationarity_check() -> None:
    from weather_world_model.worlds.run_qg import stationarity_check

    rng = np.random.default_rng(0)
    flat = 0.9 + 0.1 * rng.standard_normal((56, 100))
    ok = stationarity_check(flat)
    assert ok["passes"] is True
    assert ok["relative_difference"] < 0.05
    rising = np.linspace(0.2, 0.8, 100)[None, :] + 0.02 * rng.standard_normal((56, 100))
    bad = stationarity_check(rising)
    assert bad["passes"] is False
    assert bad["first_third_mean"] == pytest.approx(0.3, abs=0.02)
    assert bad["last_third_mean"] == pytest.approx(0.7, abs=0.02)
    assert bad["relative_difference"] > 1.0
    # A falling series fails as well; without a tolerance the value is only reported.
    assert stationarity_check(rising[:, ::-1])["passes"] is False
    assert stationarity_check(rising, None)["passes"] is None
    with pytest.raises(ValueError, match="kinetic energies"):
        stationarity_check(np.ones(10))


def test_run_stops_before_training_when_not_stationary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from dataclasses import replace

    from weather_world_model.worlds import run_qg

    strict = replace(run_qg.PROFILES["smoke"], stationarity_tolerance=1e-6)
    monkeypatch.setitem(run_qg.PROFILES, "smoke", strict)

    def no_training(*args: object, **kwargs: object) -> None:
        raise AssertionError("training must not start")

    monkeypatch.setattr(run_qg, "train_jepa", no_training)
    with pytest.raises(RuntimeError, match="not stationary"):
        run_qg.run("smoke", partial=tmp_path / "p.json")
    assert not (tmp_path / "p.json").exists()


def test_mirror_x_is_an_involution_that_flips_physical_v() -> None:
    from weather_world_model.worlds.run_qg import PROFILES, mirror_x, simulate

    data = simulate(PROFILES["smoke"])
    frames = data.frames["test"]
    mirrored = mirror_x(frames, data)
    np.testing.assert_allclose(mirror_x(mirrored, data), frames, atol=1e-5)
    physical = frames * data.std + data.mean
    physical_mirrored = mirrored * data.std + data.mean
    np.testing.assert_allclose(
        physical_mirrored[..., 0, :, :], physical[..., 0, :, ::-1], atol=1e-5
    )
    np.testing.assert_allclose(
        physical_mirrored[..., 1, :, :], -physical[..., 1, :, ::-1], atol=1e-5
    )


def test_conv_norm_option_reaches_every_encoder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import torch

    from weather_world_model import jepa
    from weather_world_model.jepa import JEPAConfig, SequenceJEPA
    from weather_world_model.worlds import run_qg

    built: list[str] = []
    original = jepa._build_encoder

    def spy(config: JEPAConfig) -> torch.nn.Module:
        built.append(config.conv_norm)
        return original(config)

    monkeypatch.setattr(jepa, "_build_encoder", spy)
    args = ["--profile", "smoke", "--conv-norm", "none", "--out", str(tmp_path / "out")]
    run_qg.main([*args, "--checkpoints", str(tmp_path / "ckpt")])
    # k1, k8 and the random-weight control: every encoder is built without normalisation.
    assert built == ["none", "none", "none"]
    record = json.loads((tmp_path / "out" / "qg-stationary-base-nonorm-smoke.json").read_text())
    assert record["conv_norm"] == "none"
    path = tmp_path / "ckpt" / "qg-stationary-base-nonorm-smoke" / "seed0-k1.pt"
    saved = torch.load(path, weights_only=True)
    assert saved["config"]["conv_norm"] == "none"
    model = SequenceJEPA(JEPAConfig(**{**saved["config"], "input_shape": (2, 64, 64)}))
    assert not any(isinstance(m, torch.nn.GroupNorm) for m in model.modules())
    model.load_state_dict(saved["state_dict"])

    # The default keeps GroupNorm and the earlier record name.
    built.clear()
    run_qg.main(["--profile", "smoke", "--out", str(tmp_path / "out")])
    assert built == ["group", "group", "group"]
    default = json.loads((tmp_path / "out" / "qg-stationary-base-smoke.json").read_text())
    assert default["conv_norm"] == "group"
    assert default["seeds"][0]["random"]["readouts"] != record["seeds"][0]["random"]["readouts"]


def test_checkpoint_saved_before_a_config_field_existed_still_matches() -> None:
    from dataclasses import asdict

    from weather_world_model.jepa import JEPAConfig
    from weather_world_model.worlds.run_qg import PROFILES, _plain, checkpoint_matches

    profile = PROFILES["smoke"]
    cfg = JEPAConfig(encoder="conv", input_shape=(2, 8, 8), conv_channels=(4, 8))
    saved = {"config": _plain(asdict(cfg)), "profile": _plain(asdict(profile)), "seed": 0}
    assert checkpoint_matches(saved, cfg, profile, 0)
    old = {**saved, "config": {k: v for k, v in saved["config"].items() if k != "conv_norm"}}
    assert checkpoint_matches(old, cfg, profile, 0)  # read with the default, "group"
    no_norm = JEPAConfig(
        encoder="conv", input_shape=(2, 8, 8), conv_channels=(4, 8), conv_norm="none"
    )
    assert not checkpoint_matches(old, no_norm, profile, 0)
    assert not checkpoint_matches(saved, cfg, profile, 1)
    assert not checkpoint_matches({}, cfg, profile, 0)
