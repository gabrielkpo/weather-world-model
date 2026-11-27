"""Verdicts of the step-2 QG runs (protocol amendments 2 and 3) from the per-seed results."""

from __future__ import annotations

from typing import Any

import numpy as np

from weather_world_model.records import summary
from weather_world_model.worlds.qg import DIAGNOSTIC_NAMES, THRESHOLDED
from weather_world_model.worlds.qg_eval import AMENDMENT3_MODELS, HORIZONS


def _rel_err(value: float | None, truth: float | None) -> float | None:
    if value is None or truth is None:
        return None
    return abs(value - truth) / abs(truth)


def verdicts(truth: dict[str, Any], seeds: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the step-2 criteria (protocol amendment 2) to the per-seed results.

    W-A uses epsilon = 1e-3 (as in the calibration) and must hold for every seed; W-C and
    W-P use the worst seed (minimum accuracy, minimum R² and minimum margin over random).
    """
    key = "eps0.001"
    true_rate = truth[key]["growth_rate"]
    rates = {m: [s[m]["predictability"][key]["growth_rate"] for s in seeds] for m in ("k1", "k8")}
    err1 = [_rel_err(r, true_rate) for r in rates["k1"]]
    err8 = [_rel_err(r, true_rate) for r in rates["k8"]]
    wa_ok = all(
        e8 is not None and e8 <= 0.30 and (e1 is None or e8 < e1)
        for e1, e8 in zip(err1, err8, strict=True)
    )
    arrow = {m: [s[m]["arrow_accuracy"] for s in seeds] for m in ("k1", "k8")}
    mirrored = {m: [s[m]["arrow_accuracy_mirrored"] for s in seeds] for m in ("k1", "k8")}
    wp: dict[str, Any] = {}
    for m in ("k1", "k8"):
        per_q: dict[str, Any] = {}
        for name in DIAGNOSTIC_NAMES:
            r2 = [s[m]["readouts"][name]["test_r2"] for s in seeds]
            rnd = [s["random"]["readouts"][name]["test_r2"] for s in seeds]
            margin = [a - b for a, b in zip(r2, rnd, strict=True)]
            entry: dict[str, Any] = {
                "r2": summary(r2),
                "random_r2": summary(rnd),
                "margin": summary(margin),
            }
            if name in THRESHOLDED:
                entry["passes"] = min(r2) >= 0.7 and min(margin) >= 0.1
            per_q[name] = entry
        wp[m] = {
            "quantities": per_q,
            "supported": all(per_q[q]["passes"] for q in THRESHOLDED),
        }
    return {
        "W-A": {
            "epsilon": 1e-3,
            "true_growth_rate": true_rate,
            "k1_growth_rate": rates["k1"],
            "k8_growth_rate": rates["k8"],
            "k1_relative_error": err1,
            "k8_relative_error": err8,
            "supported": wa_ok,
        },
        "W-C": {
            m: {
                "accuracy": summary(arrow[m]),
                "accuracy_mirrored_control": summary(mirrored[m]),
                "supported": min(arrow[m]) > 0.80,
            }
            for m in ("k1", "k8")
        },
        "W-P": wp,
        "any_collapse": any(s[m]["collapsed"] for s in seeds for m in ("k1", "k8")),
    }


def _gain(readouts: dict[str, Any], horizon: int) -> float:
    """Mean over the thresholded quantities of rollout R² minus persistence R²."""
    at_h = readouts[f"h{horizon}"]
    return float(np.mean([at_h[q]["rollout_r2"] - at_h[q]["persistence_r2"] for q in THRESHOLDED]))


def verdicts_amendment3(truth: dict[str, Any], seeds: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the criteria of protocol amendment 3 to the per-seed results.

    W-R: at h = 10, the mean gain of the rollout over persistence on the four thresholded
    quantities is at least 0.05 for the one-step and the multi-step model, on every seed.
    W-S (epsilon = 1e-3, every seed): the stochastic model's sensitivity growth rate is
    within 30 % of the simulator's and closer to it than the deterministic one-step model.
    The stochastic model's W-C, W-P and W-R scores are reported without thresholds.
    """
    models = [name for name, _, _ in AMENDMENT3_MODELS]
    wr: dict[str, Any] = {}
    for m in models:
        gains = {f"h{h}": [_gain(s[m]["rollout_readouts"], h) for s in seeds] for h in HORIZONS}
        wr[m] = {
            "mean_gain_over_persistence": gains,
            "passes_h10": all(g >= 0.05 for g in gains["h10"]),
            "quantities": {
                f"h{h}": {
                    name: {
                        key: summary([s[m]["rollout_readouts"][f"h{h}"][name][key] for s in seeds])
                        for key in ("rollout_r2", "persistence_r2")
                    }
                    for name in DIAGNOSTIC_NAMES
                }
                for h in HORIZONS
            },
        }
    key = "eps0.001"
    true_rate = truth[key]["growth_rate"]
    det = [s["k1"]["predictability"][key]["growth_rate"] for s in seeds]
    sens = [s["stoch"]["predictability"][key]["growth_rate"] for s in seeds]
    det_err = [_rel_err(r, true_rate) for r in det]
    sens_err = [_rel_err(r, true_rate) for r in sens]
    ws_ok = all(
        se is not None and se <= 0.30 and (de is None or se < de)
        for se, de in zip(sens_err, det_err, strict=True)
    )
    arrow = [s["stoch"]["arrow_accuracy"] for s in seeds]
    return {
        "W-R": {
            "horizons": list(HORIZONS),
            "criterion_quantities": list(THRESHOLDED),
            "models": wr,
            "supported": bool(wr["k1"]["passes_h10"] and wr["k8"]["passes_h10"]),
        },
        "W-S": {
            "epsilon": 1e-3,
            "true_growth_rate": true_rate,
            "deterministic_k1_growth_rate": det,
            "stochastic_sensitivity_growth_rate": sens,
            "stochastic_spread_growth_rate": [
                s["stoch"]["predictability"]["spread"]["growth_rate"] for s in seeds
            ],
            "deterministic_k1_relative_error": det_err,
            "stochastic_relative_error": sens_err,
            "supported": ws_ok,
        },
        "stochastic_reported": {
            "W-C_accuracy": summary(arrow),
            "W-C_accuracy_mirrored_control": summary(
                [s["stoch"]["arrow_accuracy_mirrored"] for s in seeds]
            ),
            "W-P_r2": {
                name: summary([s["stoch"]["readouts"][name]["test_r2"] for s in seeds])
                for name in DIAGNOSTIC_NAMES
            },
            "W-R": "see W-R.models.stoch (mean of sampled rollouts, no threshold)",
        },
        "retrained_k1_k8": verdicts(truth, seeds),
        "any_collapse": any(s[m]["collapsed"] for s in seeds for m in models),
    }
