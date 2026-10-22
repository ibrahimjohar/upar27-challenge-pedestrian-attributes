# UPAR 2027 Track 1: Pedestrian Attribute Recognition under Domain Shift

Computer Vision Assignment 2, FAST-NUCES Karachi. Entry to Track 1 of the
[UPAR Challenge 2027](https://www.codabench.org/competitions/18180/) at the Real-World Surveillance
workshop (WACV 2027).

**Task:** for every pedestrian crop, predict 40 binary attributes (age, gender, hair, clothing type
and colour, accessories). Training data comes from Market1501, PA-100K and PETA. The hidden test
set comes from an unseen domain. The ranking metric `Challenge_Avg` is the harmonic mean of
label-based mean accuracy (mA) and instance-based F1, computed per dataset and averaged.

## Repository layout

```
src/
  common.py             attribute order, exact port of the official UPAR metric, threshold calibration
  train.py              training (resumable, crash-safe) + validation analysis
  export_submission.py  checkpoint -> TorchScript -> Codabench zip -> grader-style self-test
submission/
  run.py, metadata.yaml the code that runs inside Codabench (torch + numpy + Pillow only)
notebooks/              Kaggle notebooks (they clone this repo)
  00_data_exploration   label statistics and visual probing of the three datasets (executed, with notes)
  01_download_data      downloads the challenge data
  02_train              trains one backbone, resumable
  03_export_and_package builds the Codabench zip (single model or ensemble)
  04_results_analysis   all runs and submissions, per-attribute errors, ensemble, shift, demo (executed, with notes)
tools/
  push_results.py       copies small result files from Kaggle into results/ and pushes them
  eval_ensemble.py      scores ensembles and offsets from saved validation logits
  make_notebooks.py     regenerates notebooks/
results/                metrics, histories, per-attribute tables, curves, logs (pushed from Kaggle)
report/                 report and slides
```

Weights (`*.pth`), data and submission zips are **not** in git (see `.gitignore`). They live in
the Kaggle notebook outputs.

## Method

| Component | Choice | Why |
|---|---|---|
| Backbone | CLIP ViT-B/16 and DINOv2 ViT-B/14, fine-tuned end to end, 256×128 input. Also compared: ResNet-50, ConvNeXt-S, ViT-B/16 (ImageNet-21k) | Broadly pretrained transformers generalised best. Foundation models not trained for PAR are allowed by the rules |
| Loss | Weighted BCE, w⁺ = e^(1−p), w⁻ = e^p | Class imbalance (e.g. Bald 1%, Adult 96%) |
| Sampling | Each image ∝ n_dataset^(−0.5) | PA-100K is 81% of training data; the score averages datasets equally |
| Augmentation | Flip, ±10 px shift, brightness/contrast ±0.2, small random erasing | No hue or saturation changes, because colours are labels |
| Optimisation | AdamW, layer-wise lr decay 0.65, warmup + cosine, AMP, EMA 0.9995 | Standard ViT fine-tuning recipe |
| Inference | Flip-averaged logits + per-attribute logit offsets tuned on validation | Grader thresholds at 0.5; offsets move each attribute's cut-off |
| Final model | CLIP + DINOv2 logits averaged, offsets refitted, +0.5 added to every logit, int8 weight storage | The two models make different mistakes; the models under-predict positives on new cameras; int8 keeps the zip under the size limit |

## Running it (Kaggle)

### One-time setup
1. **GitHub token.** Go to *GitHub → Settings → Developer settings → Fine-grained tokens → Generate new
   token*. Set Repository access to *Only select repositories → this repo*, Permissions to
   *Contents: Read and write*, and Expiration to 30 days. Copy the token.
2. **Kaggle.** Verify your phone number in *Settings*, which turns on internet for notebooks. Then go to
   *Settings → Secrets* (or, inside a notebook, *Add-ons → Secrets*) and add a secret named
   `GITHUB_TOKEN` holding the token. Attach it to notebooks 02 and 03.
3. **Notebooks.** `GITHUB_REPO` in the first code cell of notebooks 02 and 03 is preset to `ibrahimjohar/upar27-challenge-pedestrian-attributes`.

### Pipeline
| Notebook | Accelerator | Inputs | Produces |
|---|---|---|---|
| `00_data_exploration` | None | 01 | tables and plots in `results/data_exploration/` |
| `01_download_data` | None | none | `upar_data.tar` (all images + annotations) |
| `02_train` | GPU T4 x2 | 01 (+ a previous 02 version to resume) | `run_<backbone>/` with `best.pth`, `last.pth`, `offsets.json`, results; pushes results to `results/` |
| `03_export_and_package` | GPU T4 (or CPU) | 01, 02 | `upar_track1_submission.zip` + manifest; pushes the manifest and log |
| `04_results_analysis` | None | 01, 02, 03 outputs | tables and plots in `results/results_analysis/` |

ConvNeXt is trained on one GPU (the notebook adds `--single-gpu` for it). With the batch split over
two GPUs its host memory grew by about 11 GB per 200 steps and each step was about 20 times slower.

Start long runs with **Save Version → Save & Run All**. They run on Kaggle's servers, unaffected
by your laptop or internet.

### Interruptions and resuming
- `train.py` writes the full training state to `last.pth` (model, EMA, optimizer, LR schedule, AMP
  scaler, RNG, epoch and step) every 20 minutes and after every epoch. It writes atomically, so a
  crash can't corrupt the file.
- The run pauses itself after `TIME_BUDGET_H` (default 11 h), before Kaggle's 12 h session limit.
- **To continue after a pause or crash:** open notebook 02 and choose *Add Input → Your Work →
  Notebooks → 02_train*, picking the version that stopped. Then *Save & Run All*. Training continues
  from the saved epoch and step. If Kaggle won't let a notebook take its own output as input, use
  *New Dataset* on that version's output and add the dataset instead.
- Re-running a finished run doesn't retrain. It only redoes the final analysis.

## Results

Validation (Challenge_Avg, 33,407 images), one training recipe for every backbone:

| Backbone | Pretraining | Plain | Flip + offsets |
|---|---|---|---|
| ResNet-50 | ImageNet-1k | 0.812 | 0.826 |
| ConvNeXt-S | ImageNet-22k | 0.830 | 0.845 |
| ViT-B/16 | ImageNet-21k | 0.863 | 0.873 |
| DINOv2 ViT-B/14 | LVD-142M | 0.873 | 0.890 |
| CLIP ViT-B/16 | 400M image-text pairs | 0.878 | 0.891 |
| CLIP + DINOv2 ensemble | | 0.880 | 0.895 |

Hidden test set (Codabench, 29,248 images):

| Submission | Score | mA | Inst. F1 |
|---|---|---|---|
| CLIP | 0.7766 | 0.7787 | 0.7746 |
| CLIP refit on train+val | 0.7679 | 0.7656 | 0.7702 |
| CLIP + camera augmentation | 0.7701 | 0.7719 | 0.7684 |
| DINOv2 | 0.7733 | 0.7727 | 0.7738 |
| CLIP + DINOv2 (int8) | 0.7816 | 0.7795 | 0.7836 |
| **CLIP + DINOv2 + shift 0.5 (final, ID 1005773)** | **0.7833** | **0.7873** | **0.7794** |

The final submission was 35th on the leaderboard on 20 October 2026. Each run folder in `results/` has
`history.csv`, `results_summary.csv`, `per_attribute.csv`, `offsets.json` and `curves.png`. The
`submission_*` folders hold the export manifests and logs.

## References
- Specker, Cormier, Beyerer. *UPAR: Unified Pedestrian Attribute Recognition and Person Retrieval.* WACV 2023.
- Cormier et al. *UPAR Challenge 2024: Pedestrian Attribute Recognition and Attribute-Based Person Retrieval.* WACVW 2024.
- Radford et al. *Learning Transferable Visual Models From Natural Language Supervision (CLIP).* ICML 2021.
- Oquab et al. *DINOv2: Learning Robust Visual Features without Supervision.* TMLR 2024.
- Liu et al. *A ConvNet for the 2020s (ConvNeXt).* CVPR 2022.
- Li, Chen, Huang. *Multi-attribute Learning for Pedestrian Attribute Recognition in Surveillance Scenarios.* ACPR 2015 (weighted BCE).
- Datasets: PA-100K (Liu et al., ICCV 2017), PETA (Deng et al., ACM MM 2014), Market1501 (Zheng et al., ICCV 2015).

UPAR annotations: CC BY-NC-SA 3.0. The source image datasets are for non-commercial research use only and are not redistributed here.