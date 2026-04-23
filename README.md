

```markdown
# SyNQA: Synergistic Network QUBO Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## About SyNQA
Synergistic Network QUBO Analysis (SyNQA) is a computational framework that reframes biomarker discovery from a conventional univariate ranking to an interaction-aware combinatorial optimization problem. 

By formulating disease-specific interaction rewiring as an energy minimization problem of a Quadratic Unconstrained Binary Optimization (QUBO) model, SyNQA effectively extracts robust, non-redundant, and mechanistically interpretable feature subsets without relying on black-box machine learning algorithms. While originally applied to human gut microbiome data for colorectal cancer, its data-agnostic nature allows for potential adaptation to various high-dimensional omics modalities.

## Repository Structure
```text
.
├── data/
│   ├── demonstration/          # Dummy/demonstration datasets for testing
│   └── raw/                    # (Empty) Directory for user's full datasets
├── results/                    # Output directory for results and figures
├── SyNQA.py  # Main execution script
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
└── README.md
```

## Requirements and Installation
The pipeline is implemented in Python 3.10+. The optimization process utilizes Simulated Annealing provided by D-Wave's `dwave-neal` package.

1. Clone this repository:
   ```bash
   git clone https://github.com/YourUsername/SyNQA.git
   cd SyNQA
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

*(Note: `requirements.txt` includes `numpy`, `pandas`, `scipy`, `scikit-learn`, `dwave-neal`, `networkx`, `matplotlib`, `seaborn`, etc.)*

## Usage
To execute the entire SyNQA pipeline sequentially (from data residualization, QUBO optimization, to cross-validation and figure generation), run the master script:

```bash
python SyNQA.py
```

Results, including selected microbial taxa lists, performance metrics, and network visualization figures, will be automatically saved in the `Final_Results_SyNQA_Strict/` directory.

## Data Availability
The full metagenomic datasets analyzed in our study are publicly available in the `curatedMetagenomicData` repository (Bioconductor). To facilitate immediate testing and reproducibility, a small-scale demonstration dataset is provided in the `data/demonstration/` directory.

## Citation
If you use SyNQA in your research, please cite our paper:

> Arita, K., Nakano, Y., & Miyazaki, S. (2026). Synergistic Network QUBO Analysis (SyNQA): A combinatorial optimization framework for interaction-aware microbiome feature selection. *BMC Bioinformatics* (Under Review).

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```
