# Fine-Grained Valuation Experiments

This directory contains fine-grained data valuation experiments organized by dataset.

## Structure
```
fine_grained/
├── <Dataset>/
│   ├── tuning/
│   │   ├── high/        # high value tuning (HIGH_CONFIG)
│   │   └── low/         # low value tuning (LOW_CONFIG)
│   ├── evaluation/      # final evaluation plots (high/low vakue removal and noisy detection)
│   └── README.md        # describes HIGH_CONFIG and LOW_CONFIG for the dataset
├── legend.pdf
└── README.md
```

## Running Experiments
- `tuning/high/`: best-performing for high value (HIGH_CONFIG).
- `tuning/low/`: best-performing for low value (LOW_CONFIG).
- `evaluation/`: final plots (`high_value_removal`, `low_value_removal`, `mislabeled_detection`)

Each dataset folder includes a `README.md` describing the exact HIGH_CONFIG and LOW_CONFIG.
