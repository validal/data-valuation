# Data Valuation

> This is the official code base for **"Data Valuation for Machine Learning: Experiments and Analyses [Experiment, Analysis & Benchmark]"**

Experimental benchmark for data valuation methods in machine learning, covering fine-grained and coarse-grained granularities across diverse datasets, tasks, and scales.

---

## Repository Structure

```
data-valuation/
├── src/
│   ├── fine_grained/          # Fine-grained valuation
│   │   ├── __main__.py        # CLI entry point (Typer)
│   │   ├── dataloader/        # Dataset fetching, noise injection, registration
│   │   ├── dataval/           # Pointwise valuation methods (AME, DVRL, LAVA, ...)
│   │   ├── experiment/        # ExperimentMediator orchestration harness
│   │   └── model/             # Models
│   └── coarse_grained/        # Coarse-grained valuation
│       ├── baselines/         # OT, RVol, DAVINZ (NTK-MMD)
│       ├── models/            # CNN, ResNet, MLP
│       ├── scripts/           # MPerf correlation & robustness & sensitivity experiments
│       └── utilities/         # helpers
├── Scripts/                   # Full data valuation pipeline (Adult, CIFAR-10, HEP mass 1K–7M, ...)
├── Plots/                     # Experiment results
│   ├── fine_grained/          # Per-dataset evaluation plots (Adult, CIFAR-10, DogFish, ...)
│   │   └── (tuning inside)    # parameter tuning results per dataset
│   └── Scalability/           # Scalability results for fine- and coarse-grained methods
├── data/                      # Datasets for reproducibility
├── requirements.txt
└── setup.py
```

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

# Fine-grained scripts
python Scripts/run_adult_dataval.py --method DataOob 
python Scripts/run_cifar10_dataval.py --method KNNShapley 
python Scripts/run_hepmass_dataval_100k.py --method LAVA 

# Coarse-grained scripts
python src/coarse_grained/scripts/boostrap.py 
python src/coarse_grained/scripts/robustness.py
```

---

## Results

All plots and figures are stored in `Plots/`, organized by granularity and experiment type:
- **`Plots/fine_grained/`** — per-dataset evaluation and tuning results (Figures 4-5-6-7-8)
- **`Plots/Scalability/`** — runtime and performance scaling (Figure 9)

---

## Acknowledgements

This project builds upon several open-source works. We thank the authors of [OpenDataVal](https://github.com/opendatval/opendatval), [LAVA](https://github.com/reds-lab/LAVA/), [SAVA](https://github.com/skezle/sava), [DAVINZ](https://github.com/ZhaoxuanWu/DAVINZ-DataValuation), and [KNN-PVLDB](https://github.com/AI-secure/KNN-PVLDB) for making their implementations publicly available.

---

## License

MIT © 2026 validal
