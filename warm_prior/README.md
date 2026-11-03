# Reliability-adjusted warm priors for Bayesian active learning

The repository provides code and prepared data for acid–amine, Suzuki and
Buchwald–Hartwig reaction classification.

```text
warm_prior/
├── README.md
├── requirements.txt
├── run_experiments.py
├── active_learning_bnn_reliability_adjust_warm_prior.py
├── src/              # Required training modules
├── configs/          # Three dataset configurations
├── data/             # Six prepared pool/test .pkl.gz files
└── initial_indices/  # Nine fixed initial-GT .npy files
```

## Install

```bash
cd warm_prior
conda create -n warm-prior python=3.9 -y
conda activate warm-prior
python -m pip install -r requirements.txt
```

## Run

All commands below run from `warm_prior/`. Data are included.

```bash
# Check that the supplied inputs exist
python run_experiments.py --dataset all --check-data

# Run all 36 settings sequentially on GPU 0
python run_experiments.py --dataset all --gpu 0
```

Datasets: `acidamine`, `suzuki`, `buchwald`; seeds: `41 42 43`.
Methods: `vanilla` (no warm prior), `naive` (uncorrected pseudo-label prior),
`RA` (reliability-adjusted prior), `random` (RA prior with random acquisition). Use a fresh
`--output-dir` to repeat experiments; existing run directories are not overwritten.

## Results

Results are written to `outputs/<campaign>/seed<seed>/<method-directory>/`.
Use `metrics_by_round.csv`, filtered to `split == "test"`, for held-out
Macro-F1. Prediction summaries are in `test/`; acquisition histories are in
`selected_indices_by_round.npy`. 

## Data sources

Acid–amine: [Zhong et al.](https://www.nature.com/articles/s41467-025-59812-0).
Suzuki: [Ahneman et al.](https://www.science.org/doi/10.1126/science.aar5169) and
[King-Smith et al.](https://www.nature.com/articles/s41557-023-01393-w).
Buchwald: [Neves et al.](https://www.nature.com/articles/s43588-026-01017-6).
