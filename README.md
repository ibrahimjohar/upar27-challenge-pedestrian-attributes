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
notebooks/              Kaggle notebooks: 01 data, 02 train, 03 export (they clone this repo)
tools/
  push_results.py       copies small result files from Kaggle into results/ and pushes them
  make_notebooks.py     regenerates notebooks/
results/                metrics, histories, per-attribute tables, curves, logs (pushed from Kaggle)
report/                 report and slides
```

Weights (`*.pth`), data and submission zips are **not** in git (see `.gitignore`). They live in
the Kaggle notebook outputs.

## Method

| Component | Choice | Why |
|---|---|---|
| Backbone | CLIP ViT-B/16 (OpenAI weights), fine-tuned end to end, 256×128 input | Strong generalisation to unseen domains. Foundation models not trained for PAR are allowed by the rules |
| Loss | Weighted BCE, w⁺ = e^(1−p), w⁻ = e^p | Class imbalance (e.g. Bald 1%, Adult 96%) |
| Sampling | Each image ∝ n_dataset^(−0.5) | PA-100K is 81% of training data; the score averages datasets equally |
| Augmentation | Flip, ±10 px shift, brightness/contrast ±0.2, small random erasing | No hue or saturation changes, because colours are labels |
| Optimisation | AdamW, layer-wise lr decay 0.65, warmup + cosine, AMP, EMA 0.9995 | Standard ViT fine-tuning recipe |
| Inference | Flip-averaged logits + per-attribute logit offsets tuned on validation | Grader thresholds at 0.5; offsets move each attribute's cut-off |

## Running it (Kaggle)

### One-time setup
1. **GitHub token.** Go to *GitHub → Settings → Developer settings → Fine-grained tokens → Generate new
   token*. Set Repository access to *Only select repositories → this repo*, Permissions to
   *Contents: Read and write*, and Expiration to 30 days. Copy the token.
2. **Kaggle.** Verify your phone number in *Settings*, which turns on internet for notebooks. Then go to
   *Settings → Secrets* (or, inside a notebook, *Add-ons → Secrets*) and add a secret named
   `GITHUB_TOKEN` holding the token. Attach it to notebooks 02 and 03.
3. **Notebooks.** In each notebook, set `GITHUB_REPO = "<your-username>/<repo-name>"` in the first code cell.

### Pipeline
| Notebook | Accelerator | Inputs | Produces |
|---|---|---|---|
| `01_download_data` | None | none | `upar_data.tar` (all images + annotations) |
| `02_train` | GPU T4 x2 | 01 (+ a previous 02 version to resume) | `run_<backbone>/` with `best.pth`, `last.pth`, `offsets.json`, results; pushes results to `results/` |
| `03_export_and_package` | GPU T4 | 01, 02 | `upar_track1_submission.zip` + manifest; pushes the manifest and log |

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

See `results/`. Each training run folder has `history.csv` (per-epoch loss and validation
metrics), `results_summary.csv` (plain → flip-TTA → tuned offsets), `per_attribute.csv`
(per-dataset, per-attribute recall/mA/F1) and `curves.png`. *(Numbers are added here once
the Kaggle runs finish.)*

## References
- Specker, Cormier, Beyerer. *UPAR: Unified Pedestrian Attribute Recognition and Person Retrieval.* WACV 2023.
- Cormier et al. *UPAR Challenge 2024: Pedestrian Attribute Recognition and Attribute-Based Person Retrieval.* WACVW 2024.
- Radford et al. *Learning Transferable Visual Models From Natural Language Supervision (CLIP).* ICML 2021.
- Li, Chen, Huang. *Multi-attribute Learning for Pedestrian Attribute Recognition in Surveillance Scenarios.* ACPR 2015 (weighted BCE).
- Datasets: PA-100K (Liu et al., ICCV 2017), PETA (Deng et al., ACM MM 2014), Market1501 (Zheng et al., ICCV 2015).

UPAR annotations: CC BY-NC-SA 3.0. The source image datasets are for non-commercial research use only and are not redistributed here.
