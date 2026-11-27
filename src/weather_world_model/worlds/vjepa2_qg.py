"""Hypothesis W-V (protocol amendment 2): frozen V-JEPA 2 on the QG world rendered as video.

This module turns the QG data of :mod:`weather_world_model.worlds.run_qg` into V-JEPA 2
inputs, extracts frozen features and scores them with the same probes as the small JEPA
(W-P ridge readouts, W-C logistic arrow-of-time probe with the mirrored control of
amendment 4). It does not import ``transformers``: the encoder is any callable that maps a
normalised video ``(B, F, 3, S, S)`` to patch tokens ``(B, F / tubelet * (S / patch)², D)``
in V-JEPA 2's token order (time, then rows, then columns). The Kaggle notebook
``notebooks/vjepa2_qg_kaggle.ipynb`` wraps ``transformers.VJEPA2Model.get_vision_features``;
the tests use a tiny stand-in with the same token layout.

Fixed choices (stated before any measurement):

- **Rendering.** Each snapshot of the upper-layer eddy winds, in physical units, becomes one
  RGB image: red = ``0.5 + (u - mean_u) / (6 std_u)``, green = the same for ``v``, blue =
  ``speed / (mean_speed + 3 std_speed)``, all clipped to [0, 1]. Means and standard
  deviations are computed once on the training runs (:func:`fit_render_scales`). The image
  is resized from 64 x 64 to 256 x 256 (bilinear, no crop, so that the whole periodic
  domain is seen), then normalised with the ImageNet mean and standard deviation used by
  the V-JEPA 2 video processor.
- **Time.** V-JEPA 2 merges ``tubelet = 2`` consecutive video frames into one token. Every
  snapshot is repeated ``tubelet`` times, so a clip of 16 snapshots is a 32-frame video
  whose 16 temporal token positions map one-to-one to the 16 snapshots. Clips of 16
  snapshots match the small JEPA's W-C clips (``clip_len = 16``, stride 8).
- **Per-frame latents (W-P).** The patch tokens of one temporal position are averaged over
  the 16 x 16 spatial grid: one 1024-dimensional vector per snapshot. Each run is cut into
  consecutive clips of 16 snapshots (the last clip is aligned to the end of the run and
  only its new snapshots are used), so every snapshot is encoded once, with up to 15
  snapshots of context.
- **Clip latents (W-C).** Clip latent = mean of all patch tokens of the clip (the mean of
  its per-frame latents). Amendment 1 keeps the protocol as written for clip-level
  encoders: a reversed clip is the encoding of the time-reversed video, and the logistic
  probe reads the clip latent (:func:`clip_arrow_probe`, the W-V decider). The asymmetry
  features of amendment 1, computed on the per-position latents after a PCA to 8
  dimensions, are reported alongside for comparison with the small JEPA
  (:func:`asymmetry_arrow_probe`). For the mirrored control the FIELDS are mirrored in x
  with ``v`` negated before rendering, then reversed and encoded.

Only the frozen variant is measured. The "adapted" variant of the protocol (continued
self-supervised pretraining) needs a V-JEPA 2 training loop and is not part of this module.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.nn import functional

from weather_world_model.calibration.run import _pca
from weather_world_model.diagnostics.arrow import (
    arrow_of_time_dataset,
    probe_accuracy,
    train_probe,
)
from weather_world_model.diagnostics.readouts import fit_readout_probes
from weather_world_model.worlds.qg import DIAGNOSTIC_NAMES

FloatArray = NDArray[np.float64]
Encoder = Callable[[Tensor], Tensor]

MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"
MODEL_REVISION = "b3c1679b7c34d3255ef3547f27c7b226aefab26f"
IMAGE_SIZE = 256
PATCH_SIZE = 16
TUBELET = 2
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
THRESHOLDED = ("kinetic_energy", "available_potential_energy", "enstrophy", "jet_speed_max")
WV_MARGIN = 0.05
PROBE_SEEDS = (0, 1, 2)


@dataclass(frozen=True)
class RenderScales:
    """Fixed colour scales, from the training runs (physical units)."""

    u_mean: float
    u_std: float
    v_mean: float
    v_std: float
    speed_max: float


@dataclass(frozen=True)
class VideoSpec:
    """How snapshots become a V-JEPA 2 video."""

    image_size: int = IMAGE_SIZE
    patch_size: int = PATCH_SIZE
    tubelet: int = TUBELET
    clip_len: int = 16  # snapshots per clip (W-C and W-P)
    clip_stride: int = 8  # W-C clip stride, as for the small JEPA

    @property
    def grid(self) -> int:
        return self.image_size // self.patch_size


def fit_render_scales(train_fields: NDArray[np.floating]) -> RenderScales:
    """Colour scales from training fields ``(..., 2, H, W)`` in physical units."""
    f = np.asarray(train_fields, dtype=np.float64)
    u, v = f[..., 0, :, :], f[..., 1, :, :]
    speed = np.hypot(u, v)
    return RenderScales(
        u_mean=float(u.mean()),
        u_std=float(u.std()),
        v_mean=float(v.mean()),
        v_std=float(v.std()),
        speed_max=float(speed.mean() + 3.0 * speed.std()),
    )


def physical(frames: NDArray[np.floating], mean: NDArray[Any], std: NDArray[Any]) -> FloatArray:
    """Undo the runner's per-channel standardisation; ``mean``/``std`` have shape (2, 1, 1)."""
    return np.asarray(np.asarray(frames) * std + mean, dtype=np.float64)


def mirror_fields(fields: Tensor) -> Tensor:
    """Physical ``(u, v)`` fields ``(..., 2, H, W)`` mirrored in x with ``v`` negated."""
    out = torch.flip(fields, dims=(-1,))
    return torch.stack([out[..., 0, :, :], -out[..., 1, :, :]], dim=-3)


def render(fields: Tensor, scales: RenderScales, spec: VideoSpec) -> Tensor:
    """Physical fields ``(M, 2, H, W)`` -> normalised RGB images ``(M, 3, S, S)``."""
    u, v = fields[:, 0], fields[:, 1]
    rgb = torch.stack(
        [
            0.5 + (u - scales.u_mean) / (6.0 * scales.u_std),
            0.5 + (v - scales.v_mean) / (6.0 * scales.v_std),
            torch.sqrt(u**2 + v**2) / scales.speed_max,
        ],
        dim=1,
    ).clamp(0.0, 1.0)
    size = (spec.image_size, spec.image_size)
    rgb = functional.interpolate(rgb, size=size, mode="bilinear", align_corners=False)
    mean = torch.tensor(IMAGENET_MEAN, dtype=rgb.dtype, device=rgb.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=rgb.dtype, device=rgb.device).view(1, 3, 1, 1)
    return (rgb - mean) / std


def clip_videos(
    clips: Tensor, scales: RenderScales, spec: VideoSpec, *, reverse: bool, mirror: bool
) -> Tensor:
    """Field clips ``(B, L, 2, H, W)`` -> videos ``(B, L * tubelet, 3, S, S)``.

    Mirroring acts on the fields before rendering; reversal flips the snapshot order; each
    snapshot is then repeated ``tubelet`` times.
    """
    if mirror:
        clips = mirror_fields(clips)
    if reverse:
        clips = torch.flip(clips, dims=(1,))
    b, length = clips.shape[:2]
    images = render(clips.reshape(b * length, *clips.shape[2:]), scales, spec)
    video = images.reshape(b, length, *images.shape[1:])
    return video.repeat_interleave(spec.tubelet, dim=1)


def pool_positions(tokens: Tensor, n_positions: int, spec: VideoSpec) -> Tensor:
    """Patch tokens ``(B, N, D)`` -> per-position latents ``(B, n_positions, D)`` (spatial mean)."""
    b, n, d = tokens.shape
    if n != n_positions * spec.grid**2:
        raise ValueError(f"expected {n_positions * spec.grid**2} tokens, got {n}")
    return tokens.reshape(b, n_positions, spec.grid**2, d).float().mean(dim=2)


def encode_clips(
    encoder: Encoder,
    clips: Tensor,
    scales: RenderScales,
    spec: VideoSpec,
    *,
    reverse: bool = False,
    mirror: bool = False,
) -> Tensor:
    """Per-snapshot latents ``(B, L, D)`` of field clips ``(B, L, 2, H, W)`` (video order)."""
    video = clip_videos(clips, scales, spec, reverse=reverse, mirror=mirror)
    with torch.no_grad():
        tokens = encoder(video)
    return pool_positions(tokens, clips.shape[1], spec)


def _batches(items: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def frame_latents(
    encoder: Encoder,
    runs: NDArray[np.floating],
    scales: RenderScales,
    spec: VideoSpec,
    *,
    batch_size: int,
    device: torch.device | str = "cpu",
) -> FloatArray:
    """W-P latents ``(N, T, D)`` of physical runs ``(N, T, 2, H, W)``, every snapshot once."""
    n, t_len = runs.shape[:2]
    length = spec.clip_len
    if t_len < length:
        raise ValueError(f"runs of {t_len} snapshots are shorter than a clip ({length})")
    starts = list(range(0, t_len - length + 1, length))
    if starts[-1] + length < t_len:
        starts.append(t_len - length)
    jobs = [(i, s) for i in range(n) for s in starts]
    out: FloatArray | None = None
    for batch in _batches(jobs, batch_size):
        x = torch.as_tensor(
            np.stack([runs[i, s : s + length] for i, s in batch]),
            dtype=torch.float32,
            device=device,
        )
        lat = encode_clips(encoder, x, scales, spec).cpu().numpy().astype(np.float64)
        if out is None:
            out = np.empty((n, t_len, lat.shape[-1]))
        for (i, s), z in zip(batch, lat, strict=True):
            first_new = 0 if s % length == 0 else (starts[-2] + length) - s
            out[i, s + first_new : s + length] = z[first_new:]
    assert out is not None  # noqa: S101 (narrowing; jobs is never empty)
    return out


def clip_latents(
    encoder: Encoder,
    runs: NDArray[np.floating],
    scales: RenderScales,
    spec: VideoSpec,
    *,
    reverse: bool,
    mirror: bool,
    batch_size: int,
    device: torch.device | str = "cpu",
) -> FloatArray:
    """W-C per-position latents ``(C, L, D)`` of the clips cut from runs ``(N, T, 2, H, W)``.

    Clips start every ``clip_stride`` snapshots and are ordered start-major, as
    :func:`weather_world_model.worlds.run_qg.clips` orders them. With ``reverse`` the clip
    is encoded as the time-reversed video, so its positions are in reversed time order.
    """
    n, t_len = runs.shape[:2]
    jobs = [(i, s) for s in range(0, t_len - spec.clip_len + 1, spec.clip_stride) for i in range(n)]
    parts = []
    for batch in _batches(jobs, batch_size):
        x = torch.as_tensor(
            np.stack([runs[i, s : s + spec.clip_len] for i, s in batch]),
            dtype=torch.float32,
            device=device,
        )
        lat = encode_clips(encoder, x, scales, spec, reverse=reverse, mirror=mirror)
        parts.append(lat.cpu().numpy().astype(np.float64))
    return np.concatenate(parts, axis=0)


# --------------------------------------------------------------------- probes
def clip_arrow_probe(
    train_forward: FloatArray,
    train_reversed: FloatArray,
    test_forward: FloatArray,
    test_reversed: FloatArray,
    seed: int,
) -> float:
    """Protocol diagnostic C for a clip-level encoder: logistic probe on clip latents.

    Inputs are per-position latents ``(C, L, D)``; the clip latent is their mean over
    positions (the mean of all patch tokens). Each clip is shown forward or as its reversed
    encoding with probability 1/2 (label 1 = forward), probe as for the small JEPA
    (``l2 = 1e-3``).
    """
    rng = np.random.default_rng(seed)

    def dataset(fwd: FloatArray, rev: FloatArray) -> tuple[FloatArray, NDArray[np.int64]]:
        labels = np.asarray(rng.integers(0, 2, size=fwd.shape[0]), dtype=np.int64)
        x: FloatArray = np.where(labels[:, None] == 1, fwd.mean(axis=1), rev.mean(axis=1))
        return x, labels

    x_tr, y_tr = dataset(train_forward, train_reversed)
    x_te, y_te = dataset(test_forward, test_reversed)
    probe = train_probe(x_tr, y_tr, l2=1e-3, seed=seed)
    return probe_accuracy(probe, x_te, y_te)


def asymmetry_arrow_probe(
    train_forward: FloatArray,
    train_reversed: FloatArray,
    test_forward: FloatArray,
    test_reversed: FloatArray,
    seed: int,
) -> float:
    """Amendment-1 asymmetry features on per-position latents (comparison with the small JEPA).

    PCA to 8 dimensions fitted on the forward training clips, then the asymmetry features
    of :func:`arrow_of_time_dataset`; the reversed class is the encoding of the reversed
    video, put back in its own time order before the dataset reverses it.
    """
    rng = np.random.default_rng(seed)
    mean, basis = _pca(train_forward, min(8, train_forward.shape[-1]))

    def proj(x: FloatArray) -> FloatArray:
        return np.asarray((x - mean) @ basis, dtype=np.float64)

    x_tr, y_tr = arrow_of_time_dataset(
        proj(train_forward), rng, features="asymmetry", reversed_from=proj(train_reversed)[:, ::-1]
    )
    x_te, y_te = arrow_of_time_dataset(
        proj(test_forward), rng, features="asymmetry", reversed_from=proj(test_reversed)[:, ::-1]
    )
    probe = train_probe(x_tr, y_tr, l2=1e-3, seed=seed)
    return probe_accuracy(probe, x_te, y_te)


def evaluate_encoder(
    encoder: Encoder,
    fields: dict[str, FloatArray],
    diags: dict[str, FloatArray],
    spec: VideoSpec,
    *,
    batch_size: int,
    device: torch.device | str = "cpu",
    arrow: bool = True,
    probe_seeds: Sequence[int] = PROBE_SEEDS,
    log: Callable[[str], None] = lambda _m: None,
) -> dict[str, Any]:
    """W-P (all splits) and W-C (train and test) for one frozen encoder.

    ``fields`` holds physical runs ``(N, T, 2, H, W)`` per split, ``diags`` the aligned
    simulator diagnostics ``(N, T, Q)`` in :data:`DIAGNOSTIC_NAMES` order.
    """
    scales = fit_render_scales(fields["train"])
    kw: dict[str, Any] = {"batch_size": batch_size, "device": device}
    lat = {}
    for k, v in fields.items():
        lat[k] = frame_latents(encoder, v, scales, spec, **kw)
        log(f"W-P latents {k}: {lat[k].shape}")
    probes = fit_readout_probes(
        {k: v.reshape(-1, v.shape[-1]) for k, v in lat.items()},
        {k: v.reshape(-1, v.shape[-1]) for k, v in diags.items()},
        DIAGNOSTIC_NAMES,
    )
    result: dict[str, Any] = {
        "render_scales": asdict(scales),
        "latent_dim": int(lat["train"].shape[-1]),
        "readouts": probes.scores,
        "test_latent_std": float(lat["test"].reshape(-1, lat["test"].shape[-1]).std(axis=0).mean()),
    }
    if not arrow:
        return result
    cl: dict[str, dict[str, FloatArray]] = {}
    for split in ("train", "test"):
        cl[split] = {}
        for name, rev, mir in (("forward", False, False), ("reversed", True, False),
                               ("mirrored_reversed", True, True)):  # fmt: skip
            cl[split][name] = clip_latents(
                encoder, fields[split], scales, spec, reverse=rev, mirror=mir, **kw
            )
            log(f"W-C clips {split} {name}: {cl[split][name].shape}")
    for key, rev_name in (("plain", "reversed"), ("mirrored", "mirrored_reversed")):
        args = (cl["train"]["forward"], cl["train"][rev_name], cl["test"]["forward"],
                cl["test"][rev_name])  # fmt: skip
        result[f"arrow_clip_{key}"] = [clip_arrow_probe(*args, s) for s in probe_seeds]
        result[f"arrow_asymmetry_{key}"] = [asymmetry_arrow_probe(*args, s) for s in probe_seeds]
    result["arrow_probe_seeds"] = list(probe_seeds)
    result["n_clips"] = {s: int(cl[s]["forward"].shape[0]) for s in cl}
    return result


# --------------------------------------------------------------------- W-V
def mean_r2(readouts: dict[str, Any], names: Sequence[str] = THRESHOLDED) -> float:
    """Mean test R² over ``names`` from one ``readouts`` dict."""
    return float(np.mean([readouts[q]["test_r2"] for q in names]))


def small_jepa_reference(record: dict[str, Any]) -> dict[str, Any]:
    """W-C accuracy and mean W-P R² of the small JEPA models (k1, k8) in a step-2 record.

    Values are means over the record's training seeds.
    """
    out: dict[str, Any] = {"code_commit": record.get("code_commit")}
    for m in ("k1", "k8"):
        seeds = [s[m] for s in record["seeds"] if m in s]
        if not seeds:
            continue
        out[m] = {
            "arrow_accuracy": float(np.mean([s["arrow_accuracy"] for s in seeds])),
            "arrow_accuracy_mirrored": float(
                np.mean([s.get("arrow_accuracy_mirrored", np.nan) for s in seeds])
            ),
            "mean_r2_thresholded": float(np.mean([mean_r2(s["readouts"]) for s in seeds])),
            "mean_r2_all": float(
                np.mean([mean_r2(s["readouts"], DIAGNOSTIC_NAMES) for s in seeds])
            ),
        }
    return out


def record_norm(record: dict[str, Any], path: Path) -> str:
    """Normalisation of the small JEPA's conv encoder in a record: ``"group"`` or ``"none"``.

    Read from a ``conv_norm`` entry of the record's configuration when present; otherwise a
    file name containing ``nonorm`` or ``no-norm`` means ``"none"``; otherwise ``"group"``
    (the default of every record written before that option existed).
    """
    for section in ("config", "jepa_config"):
        value = record.get(section, {}).get("conv_norm")
        if value is not None:
            return str(value)
    stem = path.stem.lower()
    return "none" if ("nonorm" in stem or "no-norm" in stem) else "group"


def wv_comparison(
    vjepa: dict[str, Any], references: dict[str, dict[str, Any]], norms: dict[str, str]
) -> dict[str, Any]:
    """Apply W-V: V-JEPA 2 beats the small JEPA by >= 0.05 on W-C accuracy AND mean W-P R².

    "Beats the small JEPA" is read as beating the better of its one-step and multi-step
    models (seed means). W-C uses the clip-level probe against plain reversal (protocol
    section 4.C as kept by amendment 1); mean W-P R² is over the four quantities W-P
    thresholds (the mean over all seven is reported). A verdict is given only against a
    record of the small JEPA without normalisation (amendment 6); records with GroupNorm
    are compared without a verdict.
    """
    acc = float(np.mean(vjepa["arrow_clip_plain"]))
    r2 = mean_r2(vjepa["readouts"])
    out: dict[str, Any] = {
        "criterion": "V-JEPA 2 W-C accuracy and mean W-P R2 (4 thresholded quantities) "
        f"both >= best small JEPA (k1, k8; seed means) + {WV_MARGIN}",
        "vjepa2": {
            "arrow_accuracy_clip": acc,
            "arrow_accuracy_asymmetry": float(np.mean(vjepa["arrow_asymmetry_plain"])),
            "mean_r2_thresholded": r2,
            "mean_r2_all": mean_r2(vjepa["readouts"], DIAGNOSTIC_NAMES),
        },
        "against": {},
    }
    verdicts = []
    for name, ref in references.items():
        models = [ref[m] for m in ("k1", "k8") if m in ref]
        best_acc = max(m["arrow_accuracy"] for m in models)
        best_r2 = max(m["mean_r2_thresholded"] for m in models)
        entry: dict[str, Any] = {
            "conv_norm": norms[name],
            "small_jepa": ref,
            "accuracy_margin": acc - best_acc,
            "r2_margin": r2 - best_r2,
        }
        if norms[name] == "none":
            entry["supported"] = bool(acc - best_acc >= WV_MARGIN and r2 - best_r2 >= WV_MARGIN)
            verdicts.append(entry["supported"])
        else:
            entry["supported"] = None
        out["against"][name] = entry
    out["supported"] = verdicts[0] if len(verdicts) == 1 else None
    if not verdicts:
        out["note"] = "no record of the small JEPA without normalisation: verdict left open"
    elif len(verdicts) > 1:
        out["note"] = "several no-normalisation records: see each entry"
    return out
