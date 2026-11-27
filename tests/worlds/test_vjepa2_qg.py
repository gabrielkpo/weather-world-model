"""CPU tests of the V-JEPA 2 adapter with a tiny stand-in encoder (no weights downloaded)."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from weather_world_model.worlds.qg import DIAGNOSTIC_NAMES
from weather_world_model.worlds.vjepa2_qg import (
    VideoSpec,
    clip_latents,
    clip_videos,
    encode_clips,
    evaluate_encoder,
    fit_render_scales,
    frame_latents,
    mirror_fields,
    physical,
    pool_positions,
    record_norm,
    render,
    small_jepa_reference,
    wv_comparison,
)

SPEC = VideoSpec(image_size=32, patch_size=16, tubelet=2, clip_len=4, clip_stride=2)


class TinyVJEPA(nn.Module):
    """Same input and token layout as V-JEPA 2: (B, F, 3, S, S) -> (B, F/2 * grid², D)."""

    def __init__(self, dim: int = 6) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.proj = nn.Conv3d(3, dim, kernel_size=(2, 16, 16), stride=(2, 16, 16))

    def forward(self, video: Tensor) -> Tensor:
        out: Tensor = self.proj(video.permute(0, 2, 1, 3, 4)).flatten(2).transpose(1, 2)
        return out


def _runs(n: int, t: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, t, 2, 8, 8))


def test_render_is_normalised_rgb_of_fixed_size() -> None:
    runs = _runs(2, 3)
    scales = fit_render_scales(runs)
    img = render(torch.as_tensor(runs[0], dtype=torch.float32), scales, SPEC)
    assert img.shape == (3, 3, 32, 32)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    raw = img * std + mean
    assert raw.min() >= -1e-6 and raw.max() <= 1 + 1e-6
    assert scales.speed_max > 0


def test_mirror_flips_x_and_negates_v() -> None:
    f = torch.as_tensor(_runs(1, 1)[0, 0])
    m = mirror_fields(f)
    assert torch.allclose(m[0], torch.flip(f[0], dims=(-1,)))
    assert torch.allclose(m[1], -torch.flip(f[1], dims=(-1,)))
    assert torch.allclose(mirror_fields(m), f)


def test_physical_undoes_standardisation() -> None:
    mean, std = np.array([1.0, -2.0]).reshape(2, 1, 1), np.array([2.0, 3.0]).reshape(2, 1, 1)
    x = np.zeros((2, 8, 8))
    assert np.allclose(physical(x, mean, std)[1], -2.0)


def test_snapshots_repeat_over_the_tubelet() -> None:
    clips = torch.as_tensor(_runs(1, 4), dtype=torch.float32)
    scales = fit_render_scales(clips.numpy())
    video = clip_videos(clips, scales, SPEC, reverse=False, mirror=False)
    assert video.shape == (1, 8, 3, 32, 32)
    assert torch.equal(video[0, 0], video[0, 1])
    rev = clip_videos(clips, scales, SPEC, reverse=True, mirror=False)
    assert torch.equal(rev[0, 0], video[0, -1])


def test_pool_positions_checks_token_count() -> None:
    with pytest.raises(ValueError, match="tokens"):
        pool_positions(torch.zeros(1, 5, 3), 2, SPEC)


def test_reversed_clip_latents_are_forward_latents_reversed() -> None:
    # The stand-in encoder has no temporal mixing, so reversal only permutes positions.
    runs = _runs(2, 8)
    scales = fit_render_scales(runs)
    enc = TinyVJEPA()
    fwd = clip_latents(enc, runs, scales, SPEC, reverse=False, mirror=False, batch_size=3)
    rev = clip_latents(enc, runs, scales, SPEC, reverse=True, mirror=False, batch_size=3)
    assert fwd.shape == (2 * 3, 4, 6)  # starts 0, 2, 4 for each of 2 runs, start-major
    assert np.allclose(rev[:, ::-1], fwd, atol=1e-5)
    first = encode_clips(enc, torch.as_tensor(runs[1:2, 2:6], dtype=torch.float32), scales, SPEC)
    assert np.allclose(fwd[3], first[0].numpy(), atol=1e-5)  # start 2, run 1


def test_frame_latents_cover_every_snapshot_once() -> None:
    runs = _runs(2, 10)
    scales = fit_render_scales(runs)
    enc = TinyVJEPA()
    lat = frame_latents(enc, runs, scales, SPEC, batch_size=4)
    assert lat.shape == (2, 10, 6)
    tail = encode_clips(enc, torch.as_tensor(runs[:, 6:10], dtype=torch.float32), scales, SPEC)
    assert np.allclose(lat[:, 8:], tail[:, 2:].numpy(), atol=1e-5)
    head = encode_clips(enc, torch.as_tensor(runs[:, 4:8], dtype=torch.float32), scales, SPEC)
    assert np.allclose(lat[:, 4:8], head.numpy(), atol=1e-5)
    with pytest.raises(ValueError, match="shorter"):
        frame_latents(enc, runs[:, :3], scales, SPEC, batch_size=4)


def test_evaluate_encoder_reports_wp_and_wc() -> None:
    rng = np.random.default_rng(1)
    fields = {k: _runs(n, 8, i) for i, (k, n) in enumerate((("train", 4), ("val", 2), ("test", 2)))}
    diags = {k: rng.normal(size=(*v.shape[:2], len(DIAGNOSTIC_NAMES))) for k, v in fields.items()}
    out = evaluate_encoder(TinyVJEPA(), fields, diags, SPEC, batch_size=4, probe_seeds=(0,))
    assert set(out["readouts"]) == set(DIAGNOSTIC_NAMES)
    for key in ("arrow_clip_plain", "arrow_clip_mirrored", "arrow_asymmetry_plain"):
        assert len(out[key]) == 1 and 0.0 <= out[key][0] <= 1.0
    assert out["n_clips"] == {"train": 12, "test": 6}
    no_arrow = evaluate_encoder(TinyVJEPA(), fields, diags, SPEC, batch_size=4, arrow=False)
    assert "arrow_clip_plain" not in no_arrow


def _record(acc: float, r2: float, **extra: Any) -> dict[str, Any]:
    model = {
        "arrow_accuracy": acc,
        "arrow_accuracy_mirrored": 0.5,
        "readouts": {q: {"test_r2": r2} for q in DIAGNOSTIC_NAMES},
    }
    return {"code_commit": "abc", "seeds": [{"k1": model, "k8": model}], **extra}


def test_record_norm_reads_a_runner_record_under_a_neutral_name(tmp_path: Path) -> None:
    import json

    from weather_world_model.worlds import run_qg

    run_qg.main(["--profile", "smoke", "--conv-norm", "none", "--out", str(tmp_path)])
    record = json.loads((tmp_path / "qg-stationary-base-nonorm-smoke.json").read_text())
    # The name carries no hint: the value must come from the record itself.
    assert record_norm(record, Path("reference.json")) == "none"
    assert record_norm({**record, "conv_norm": "group"}, Path("x-nonorm.json")) == "group"


def test_record_norm_falls_back_to_config_then_file_name() -> None:
    assert record_norm(_record(0.5, 0.5, config={"conv_norm": "none"}), Path("x.json")) == "none"
    assert record_norm(_record(0.5, 0.5, config=None), Path("qg-stationary-nonorm.json")) == "none"
    assert record_norm(_record(0.5, 0.5), Path("qg-stationary.json")) == "group"


def test_wv_skips_a_record_without_small_jepa_models() -> None:
    vjepa = {
        "arrow_clip_plain": [0.9],
        "arrow_asymmetry_plain": [0.6],
        "readouts": {q: {"test_r2": 0.8} for q in DIAGNOSTIC_NAMES},
    }
    empty = small_jepa_reference({"code_commit": "abc", "seeds": [{"random": {}}]})
    out = wv_comparison(vjepa, {"empty": empty}, {"empty": "none"})
    assert out["against"]["empty"]["supported"] is None
    assert out["supported"] is None and "note" in out


def test_wv_verdict_only_against_no_norm_records() -> None:
    vjepa = {
        "arrow_clip_plain": [0.9, 0.9],
        "arrow_asymmetry_plain": [0.6],
        "readouts": {q: {"test_r2": 0.8} for q in DIAGNOSTIC_NAMES},
    }
    refs = {"gn": small_jepa_reference(_record(0.7, 0.5))}
    open_ = wv_comparison(vjepa, refs, {"gn": "group"})
    assert open_["supported"] is None and "note" in open_
    assert open_["against"]["gn"]["accuracy_margin"] == pytest.approx(0.2)
    refs["nn"] = small_jepa_reference(_record(0.88, 0.5))
    closed = wv_comparison(vjepa, refs, {"gn": "group", "nn": "none"})
    assert closed["supported"] is False  # accuracy margin 0.02 < 0.05
    refs["nn"] = small_jepa_reference(_record(0.8, 0.7))
    assert wv_comparison(vjepa, refs, {"gn": "group", "nn": "none"})["supported"] is True
