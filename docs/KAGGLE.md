# Running the W-V measurement on Kaggle

Hypothesis W-V (protocol amendment 2) compares the frozen V-JEPA 2 ViT-L on the QG world with the
small JEPA trained from scratch. The measurement needs a GPU and runs in
[`notebooks/vjepa2_qg_kaggle.ipynb`](../notebooks/vjepa2_qg_kaggle.ipynb) on a free Kaggle GPU
session (T4 or P100). The adapter it uses is `weather_world_model.worlds.vjepa2_qg`, whose docstring
fixes the rendering, the pooling and the probes.

The notebook installs the package in one of two modes, set in its first code cell:

- `INSTALL_MODE = "git"`: `pip install git+https://github.com/gabrielkpo/weather-world-model@<commit>`.
  Only works once the repository is public; set `GIT_COMMIT` to the full SHA to measure.
- `INSTALL_MODE = "wheel"` (default): installs a wheel from a private Kaggle dataset. Steps below.

## Steps (mode "wheel")

1. **Build the dataset folder** from a clean checkout of the commit to measure:

   ```bash
   scripts/build_wheel.sh
   ```

   It writes `dist/kaggle-wheel/` (ignored by Git): the wheel, `build-info.json` (the commit) and
   `records/` (the committed `results/qg/qg-stationary*.json` records the notebook compares against).
2. **Create the private dataset.** On kaggle.com: *Datasets → New Dataset*, title
   `weather-world-model-wheel`, upload the *contents* of `dist/kaggle-wheel/` (keep the `records/`
   folder), visibility *Private*, *Create*. For a later commit, open the dataset and use
   *New Version* with the new folder contents.
3. **Create the notebook.** *Code → New Notebook → File → Import Notebook*, upload
   `notebooks/vjepa2_qg_kaggle.ipynb`.
4. **Attach the dataset.** In the notebook's right panel: *Input → Add Input → Your Work → Datasets*,
   select `weather-world-model-wheel`. It must appear as `/kaggle/input/weather-world-model-wheel/`.
5. **Settings** (right panel, *Session options*): *Accelerator → GPU T4 x1* (or P100), *Internet → On*
   (phone verification of the Kaggle account is required once for internet and GPU).
6. **Run.** *Run All* (or *Save Version → Save & Run All* to run in the background, up to 12 h).
   Expected: about 10 min of simulation on CPU, then 10 to 30 min of encoding on a T4 (an estimate;
   the record stores the measured times and peak GPU memory).
7. **Download the result.** In the *Output* panel (or the saved version's *Output* tab), download
   `qg-vjepa2.json`.
8. **Commit it** as `results/qg/qg-vjepa2.json` in a pull request.

To check the notebook without a GPU, set `DRY_RUN = True`: smoke-sized data and a tiny randomly
initialised V-JEPA 2, a few minutes on CPU, values without meaning.

## What the record holds

Code commit, install mode, model id and pinned revision, data and video configuration, the data
check against the comparison records (same data seed and split; the simulation is chaotic, so another
CPU can produce different snapshots with the same statistics), W-P test R² for the frozen model and a
random-weight control, W-C accuracies (clip-level probe and amendment-1 asymmetry features, plain
reversal and mirrored control, three probe seeds), timings, environment, and the W-V comparison.

The W-V verdict is given only against a record of the small JEPA without normalisation (amendment 6).
A record is recognised as such by a `conv_norm: "none"` entry in its `config`, or by `nonorm` in its
file name. Without one, the record holds the raw values and the comparison with the GroupNorm models,
and the verdict (`W-V.supported`) is `null`.

## Not measured here

- The *adapted* variant of V-JEPA 2 (continued self-supervised pretraining, protocol section 2):
  it needs a V-JEPA 2 training loop and is left to a later change.
- Diagnostic A (W-A) and rollout readouts (W-R) are not part of W-V; the frozen encoder is used
  without its predictor.
