"""Exploratory analysis: does V-JEPA 2's arrow-of-time result survive spatial symmetry controls?

Not a protocol result. The frozen V-JEPA 2 record gives about 100 % on the mirrored control of
diagnostic C, where every small JEPA is at chance. In that control only the reversed class is
mirrored, so an encoder that sees whether an image is mirrored separates the classes without
any notion of time. This script measures, for the pretrained model and for the same
architecture with random weights (seed 0), the probes of
:func:`weather_world_model.worlds.vjepa2_qg.symmetry_controls`:

- ``plain`` and ``mirrored``: the W-C probes of the record;
- ``mirror_only``: plain against mirrored clips, time unchanged;
- ``d4``: forward against reversed clips, each clip of both classes turned by an independent
  random element of the square's symmetry group (rotations and reflections, applied to the
  grid and to the wind vector);
- ``d4_static``: the same with each clip frozen on one snapshot, first against last.

It needs a GPU and ``transformers``; it runs from a clone of the repository on Kaggle
(``notebooks/arrow_symmetry_kaggle.ipynb``). ``--dry-run`` uses the smoke profile and a tiny
randomly initialised V-JEPA 2 on CPU, to test the script.

Run: ``python scripts/explore_arrow_symmetry_vjepa2.py --out results/exploratory``.
"""

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from weather_world_model.records import provenance
from weather_world_model.worlds import run_qg
from weather_world_model.worlds import vjepa2_qg as vq
from weather_world_model.worlds.waves import wave_runs

WORLDS = ("qg", "waves", "damped_waves")
WAVE_DAMPING = 0.2  # amendment 8: the wave energy falls by 1 - exp(-0.4 * 1.5), 45 %, over a clip


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=Path("results/exploratory"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--worlds",
        nargs="+",
        choices=WORLDS,
        default=["qg"],
        help="qg: the QG world; waves: reversible Rossby waves; damped_waves: the same, damped",
    )
    args = parser.parse_args(argv)

    from transformers import (  # type: ignore[import-not-found]  # not a dependency: GPU only
        VJEPA2Config,
        VJEPA2Model,
    )

    t0 = time.perf_counter()

    def log(msg: str) -> None:
        print(f"[{time.perf_counter() - t0:7.0f}s] {msg}", flush=True)

    device = "cuda" if torch.cuda.is_available() and not args.dry_run else "cpu"
    if not args.dry_run and device != "cuda":
        sys.exit("a GPU is required (use --dry-run to test on CPU)")
    dtype = torch.float16 if device == "cuda" else torch.float32

    profile_name = "smoke" if args.dry_run else "full"
    profile = run_qg.PROFILES[profile_name]
    worlds: dict[str, dict[str, np.ndarray]] = {}
    for world in args.worlds:
        if world == "qg":
            data = run_qg.simulate(profile)
            worlds[world] = {
                k: vq.physical(data.frames[k], data.mean, data.std).astype(np.float32)
                for k in ("train", "test")
            }
        else:
            n_train, _, n_test = run_qg.split_sizes(profile.n_runs)
            rng = np.random.default_rng([profile.data_seed, WORLDS.index(world)])
            damping = WAVE_DAMPING if world == "damped_waves" else 0.0
            worlds[world] = {
                split: wave_runs(n, profile.snapshots, rng=rng, damping=damping)
                for split, n in (("train", n_train), ("test", n_test))
            }
        log(f"{world}: { ({k: v.shape for k, v in worlds[world].items()}) }")

    if args.dry_run:
        torch.manual_seed(0)
        tiny = {
            "hidden_size": 64,
            "num_attention_heads": 4,
            "num_hidden_layers": 2,
            "pred_hidden_size": 32,
            "pred_num_attention_heads": 4,
            "pred_num_hidden_layers": 1,
        }
        model = VJEPA2Model(VJEPA2Config(**tiny))
    else:
        model = VJEPA2Model.from_pretrained(vq.MODEL_ID, revision=vq.MODEL_REVISION, dtype=dtype)
    model = model.to(device).eval()
    cfg = model.config
    spec = vq.VideoSpec(
        image_size=cfg.crop_size,
        patch_size=cfg.patch_size,
        tubelet=cfg.tubelet_size,
        clip_len=profile.clip_len,
        clip_stride=profile.clip_stride,
    )

    def encoder_of(m: Any) -> vq.Encoder:
        def encode(video: torch.Tensor) -> torch.Tensor:
            out: torch.Tensor = m.get_vision_features(video.to(device=device, dtype=dtype))
            return out

        return encode

    results: dict[str, Any] = {}
    seconds: dict[str, float] = {}
    for name in ("pretrained", "random_weights"):
        if name == "random_weights":
            del model
            torch.manual_seed(0)
            model = VJEPA2Model(cfg).to(device=device, dtype=dtype).eval()
        for world, fields in worlds.items():
            t = time.perf_counter()
            res = vq.symmetry_controls(
                encoder_of(model), fields, spec, batch_size=args.batch_size, device=device, log=log
            )
            key = name if args.worlds == ["qg"] else f"{world}/{name}"
            results[key] = res
            seconds[key] = time.perf_counter() - t
            log(key + " " + json.dumps({k: v for k, v in res.items() if isinstance(v, dict)}))

    record = {
        **provenance(),
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "dry_run": args.dry_run,
        "model_id": None if args.dry_run else vq.MODEL_ID,
        "model_revision": None if args.dry_run else vq.MODEL_REVISION,
        "profile": profile_name,
        "data_seed": profile.data_seed,
        "worlds": args.worlds,
        "wave_damping": WAVE_DAMPING,
        "video": {
            "image_size": spec.image_size,
            "clip_len": spec.clip_len,
            "clip_stride": spec.clip_stride,
            "tubelet": spec.tubelet,
        },
        "environment": {
            "torch": torch.__version__,
            "device": device,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        },
        "seconds": {**seconds, "wall": time.perf_counter() - t0},
        **results,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    stem = "arrow-symmetry-vjepa2" + ("" if args.worlds == ["qg"] else "-" + "-".join(args.worlds))
    path = args.out / (stem + ("-dry-run" if args.dry_run else "") + ".json")
    path.write_text(json.dumps(record, indent=2) + "\n")
    log(f"wrote {path}")


if __name__ == "__main__":
    main()
