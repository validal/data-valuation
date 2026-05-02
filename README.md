# Data Valuation

Experimental benchmark for data valuation methods in machine learning, covering instance-level and group-level granularities across diverse datasets and scales. Built on top of [OpenDataVal](https://github.com/opendatval/opendatval).

---

## Repository Structure

```
data-valuation/
├── src/
│   ├── fine_grained/          # Instance-level valuation
│   │   ├── __main__.py        # CLI entry point (Typer)
│   │   ├── dataloader/        # Dataset fetching, noise injection, registration
│   │   ├── dataval/           # 16+ valuation methods (AME, DVRL, Shapley variants, LAVA, ...)
│   │   ├── experiment/        # ExperimentMediator orchestration harness
│   │   └── model/             # Model wrappers (MLP, BERT, LeNet, LogReg)
│   └── coarse_grained/        # Group-level valuation
│       ├── baselines/         # MMD, NTK, OT, RV, DAVINZ
│       ├── models/            # CNN, ResNet, VGG, tabular MLP
│       ├── scripts/           # Bootstrap correlation & robustness experiments
│       └── utilities/         # PyTorch helpers, plotting
├── Scripts/                   # Dataset-specific runners (Adult, CIFAR-10, HEP mass 1K–7M, ...)
├── Plots/                     # Experiment results
│   ├── fine_grained/          # Per-dataset evaluation plots (Adult, CIFAR-10, DogFish, ...)
│   │   └── (tuning inside)    # Hyperparameter tuning results per dataset
│   ├── coarse_grained/
│   │   ├── evaluation/        # Coarse-grained evaluation results
│   │   └── tuning/            # Coarse-grained tuning results
│   └── Scalability/           # Scalability results for fine- and coarse-grained methods
├── data/                      # Datasets (numpy arrays: X/y source, val, test)
├── requirements.txt
└── setup.py
```

---

## Methods

**Fine-Grained (instance-level):** AME, DVRL, InfluenceFunction, KNNShapley, DataShapley, DataBanzhaf, BetaShapley, DataOob, LeaveOneOut, LAVA/SAVA, GAVA, ForgettingEvents, ClassWiseShapley, RobustVolumeShapley, DVRLShap, RandomEvaluator

**Coarse-Grained (group-level):** MMD, NTK, OT (Wasserstein), RV, DAVINZ

---

## Setup

```bash
python -m venv .venv && .\.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

---

## Running Experiments

```bash
# Fine-grained CLI
python -m src.fine_grained --help

# Dataset-specific runner (supports --method and --seed)
python Scripts/run_adult_dataval.py --method DataOob --seed 42
python Scripts/run_cifar10_dataval.py --method KNNShapley --seed 42
python Scripts/run_hepmass_dataval_100k.py --method LAVA --seed 42

# Coarse-grained scripts
python src/coarse_grained/scripts/boostrap.py --help
python src/coarse_grained/scripts/robustness.py --help
```

---

## Results

All plots and figures are stored in `Plots/`, organized by granularity and experiment type:
- **`Plots/fine_grained/`** — per-dataset evaluation and tuning results
- **`Plots/coarse_grained/evaluation/`** and **`/tuning/`** — coarse-grained results
- **`Plots/Scalability/`** — runtime and performance scaling from 1K to 7M samples

---

## License

MIT © 2026 validal
