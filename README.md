# Data Valuation: Fine-Grained and Coarse-Grained Evaluation

A comprehensive experimental evaluation of data valuation methods across diverse datasets, granularities, and at scale using OpenDataVal.

## Overview

This repository provides a structured framework for evaluating data valuation methods across two main granularities:
- **Fine-Grained**: Detailed, instance-level data value assessment
- **Coarse-Grained**: Aggregate, group-level data value assessment

Each granularity includes dedicated workflows for hyperparameter tuning and comprehensive evaluation.

## Repository Structure

```
data-valuation/
├── src/                              # Source code
│   ├── fine_grained/                # Fine-grained valuation methods
│   │   ├── __init__.py
│   │   └── methods.py              # Fine-grained implementation
│   ├── coarse_grained/             # Coarse-grained valuation methods
│   │   ├── __init__.py
│   │   └── methods.py              # Coarse-grained implementation
│   └── __init__.py
│
├── experiments/                     # Experimental workflows
│   ├── fine_grained/
│   │   ├── tuning/                 # Hyperparameter tuning
│   │   ├── evaluation/             # Evaluation results
│   │   └── README.md
│   ├── coarse_grained/
│   │   ├── tuning/                 # Hyperparameter tuning
│   │   ├── evaluation/             # Evaluation results
│   │   └── README.md
│   ├── configs/                    # Experiment configurations
│   └── datasets/                   # Dataset specifications
│
├── data/                           # Data management
│   ├── raw/                        # Original datasets
│   ├── processed/                  # Processed datasets
│   └── results/                    # Experiment results
│
├── notebooks/                      # Jupyter notebooks
├── tests/                          # Unit tests
├── .gitignore                      # Python & data science config
├── requirements.txt                # Dependencies (with opendatval)
├── setup.py                        # Package installation
├── README.md                       # Comprehensive documentation
└── LICENSE                         # MIT License
```

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/validal/data-valuation.git
cd data-valuation
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install the package in development mode:
```bash
pip install -e .
```

## Usage

### Fine-Grained Experiments

For instance-level data valuation using OpenDataVal:

```python
from src.fine_grained import methods

# Your fine-grained valuation code here
```

Experiments and results go in:
- `experiments/fine_grained/tuning/` - Tuning experiments
- `experiments/fine_grained/evaluation/` - Evaluation results

### Coarse-Grained Experiments

For group-level data valuation using OpenDataVal:

```python
from src.coarse_grained import methods

# Your coarse-grained valuation code here
```

Experiments and results go in:
- `experiments/coarse_grained/tuning/` - Tuning experiments
- `experiments/coarse_grained/evaluation/` - Evaluation results

## Data Organization

- **`data/raw/`**: Store original, unmodified datasets
- **`data/processed/`**: Store cleaned and processed datasets
- **`data/results/`**: Store evaluation results and metrics

## Datasets

Dataset specifications and metadata should be placed in `experiments/datasets/`.

## Framework

This project uses [OpenDataVal](https://github.com/opendatval/opendatval) as the main framework for data valuation methods.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this repository in your research, please cite it appropriately:

```bibtex
@repository{validal2026datavaluatione,
  title={Data Valuation: Fine-Grained and Coarse-Grained Evaluation},
  author={Validal},
  year={2026},
  url={https://github.com/validal/data-valuation}
}
```

## Contact

For questions or inquiries, please open an issue on the repository.