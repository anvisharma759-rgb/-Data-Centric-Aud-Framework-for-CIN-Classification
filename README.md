# Label/image auditing with DST + Grad-CAM: code for the paper

Code behind the paper's CIN1/CIN2/CIN3 colposcopy audit experiments (two-loop training: train → audit → retrain).
Everything runs in Google Colab (GPU) from the notebooks in `notebooks/`.

## What each notebook does

Notebooks `01`–`03` share the same pipeline (`04` reuses its Loop 1 and audit code): 5-fold patient-level `GroupKFold` (zero patient leakage asserted),
10× pre-computed augmentation, a 3-model ensemble (ResNet18, MobileNetV3-Small, EfficientNet-B0; two-phase training with MixUp),
Dempster–Shafer conflict `K` plus a Grad-CAM artifact score per training image, a paired t-test of Loop 2 vs Loop 1 macro-F1,
and last-fold diagnostics.

| Notebook | What it runs |
|---|---|
| `01_exclusion.ipynb` | **Exclusion.** `combined = 0.5·K + 0.5·artifact`; the top 5% (`EXCLUDE_PERCENTILE = 95`) of training originals are dropped before Loop 2. |
| `02_down_weighting.ipynb` | **Down-weighting.** Same flagged top 5% (`DOWNWEIGHT_PERCENTILE = 95`) are kept with `sample_weight = DOWNWEIGHT_FACTOR` (0.2). |
| `03_exclusion_weights_0.7_0.3.ipynb` | **Sensitivity run.** Exclusion as in `01`, with `combined = 0.7·K + 0.3·artifact`. This is the only change from `01`. |
| `04_fold5_comparison_review.ipynb` | **Fold-5 comparison-sample review.** Re-runs Loop 1 and the audit for fold 5 only, then builds a blinded set of 60 Grad-CAM overlays (20 highest-, 20 lowest- and 20 random-artifact-score training images) with a blank rating sheet. After rating, it counts burn-in and anatomically centered attention per group and runs Fisher exact tests. |

The last cell of notebooks `01`–`03` also writes last-fold (fold 5) diagnostics: Loop 1 vs Loop 2 confusion matrices, K-vs-artifact scatter, per-class K histograms, `audit_report.csv`, and Grad-CAM overlays for the 20 highest artifact scores.

Relabelling is implemented (2-of-3 model agreement, mean confidence ≥ 0.6, low K) but flagged zero images in every fold of the runs reported here.

## Implementation details (paper Sec. II.D)

Read these before comparing the code to the paper's description:

1. **Exclusion kept the augmented copies.** Excluding a flagged image removes only the *original* from the training table.
   `expand_with_augmented()` then re-adds all files in the augmented folder for that patient (matched on `(folder, patient_folder)`),
   so the augmented copies of excluded images, including the unmodified copy saved alongside them, stay in Loop 2 training.
2. **Down-weighting missed the identical copy and applied the factor in both the sampler and the loss.**
   - The factor is propagated to augmented copies by matching filenames that start with `<original stem>_`. The identical (unaugmented)
     copy in the augmented folder has no `_k` suffix, so it is **not** matched and keeps weight 1.0.
   - The factor is applied **twice**: in the `WeightedRandomSampler` (class-balance weight × factor) and in the training loss
     (`reduction='none'`, then a weighted mean normalised by the sum of sample weights). Validation and test losses are unweighted.

Note: Loop 1 is retrained inside every notebook, and its fold scores are not identical across notebooks
(for example fold 2 and fold 5 differ between `01` and `02`), so Loop 1 baselines vary between runs.

## Data (not included)

The Malhari images are **not** redistributed here. Download them from Mendeley Data:

> Malhari Dataset (Kalbhor et al., Pimpri Chinchwad College of Engineering), Version 1, DOI [10.17632/m5kxdj7m36.1](https://doi.org/10.17632/m5kxdj7m36.1)
> https://data.mendeley.com/datasets/m5kxdj7m36/1

Please check the dataset's licence/terms on that page. The notebooks expect this layout (patient folders contain the images):

```
/content/data/Malhari_Dataset_V1/colposcopic/
    CIN1/<patient folder>/*.jpg
    CIN2/<patient folder>/*.jpg
    CIN3/<patient folder>/*.jpg
```

Change `DATASET_ROOT` in the configuration cell if you store it elsewhere.

Colab's file panel cannot upload folders. Upload the dataset as a single .zip instead and unzip it:

    !mkdir -p /content/data
    !unzip -q /content/YOUR_ZIP_NAME.zip -d /content/data

Unzipping can add an extra nested folder. Run
`!find /content/data -maxdepth 4 -type d | head -60`
and set `DATASET_ROOT` in the configuration cell to the folder that directly contains CIN1, CIN2 and CIN3.

## Run in Colab

1. Open a notebook from `notebooks/` in Colab (File → Open notebook → GitHub, or upload it) and select **Runtime → Change runtime type → GPU** (the runs used a T4).
2. The notebooks were run on Colab's preinstalled libraries (see below). To match them, check `torch.__version__` etc. against `requirements.txt` and install pinned versions only if they differ (`!pip install -r requirements.txt`, then restart the runtime).
3. Download the dataset (above) into `/content/data/Malhari_Dataset_V1/colposcopic`.
4. **Run all cells in order** (for `04`, follow the rating steps in its first cell; the rating sheet is filled in by hand between Cell 11 and Cell 12). The full pipeline cell trains 3 models × 2 loops × 5 folds, which takes a long time; keep the session alive.
   The augmentation cell (2.5) writes the 10× copies to `/content/mendeley_augmented`; run it once per session.
5. Outputs are written inside the Colab VM and vanish when the session ends, so download them if you need them:
   `/content/audit_loop1/` (`kfold_results.json`, `kfold_comparison.png`, `diagnostic_plots.png`, `audit_report.csv`, `flagged_images/`)
   `/content/outputs/progress_log.csv`, and for `04` the folder `/content/comparison_review_fold5/`.

Seeds are fixed (`SEED = 42`, per fold `SEED + fold`), but GPU training can still differ slightly between runs.

## Library versions (Sec. II.G)

`requirements.txt` pins the versions recorded by Cell 1 of notebook `04` on Colab (Python 3.13.15, torch 2.11.0, torchvision 0.26.0,
scikit-learn 1.6.1, numpy 2.1.3, scipy 1.16.3, pandas 2.2.3, CUDA 13.0, cuDNN 9.27, Tesla T4). Matplotlib and Pillow were not recorded.
Notebook `04` saves `versions.json` on each run so you can compare against your own environment.
