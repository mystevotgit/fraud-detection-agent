# Intelligent Cost-Sensitive Credit Card Fraud Detection Agent

This repository is a CS 582 Machine Learning project for cost-sensitive fraud-detection prototype. The proposed system compares supervised and unsupervised models, calibrates fraud probabilities, and uses a Genetic Algorithm to learn two thresholds for `APPROVE`, `MANUAL REVIEW`, and `BLOCK/ESCALATE` decisions under configurable operational costs.

## Dataset

Use the public Kaggle Credit Card Fraud Detection dataset commonly attributed to the Worldline/ULB Machine Learning Group:

<https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud>

Expected local file:

```bash
data/raw/creditcard.csv
```

The full dataset is intentionally ignored by Git. See [data/raw/README.md](/Users/olatunjioye/Documents/Fraud%20Detection%20Agent/data/raw/README.md) for setup instructions.

## Reproducible Commands

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

```bash
python scripts/prepare_data.py --config configs/default.yaml
python scripts/train_models.py --config configs/default.yaml
python scripts/optimize_thresholds.py --config configs/default.yaml
python scripts/evaluate_test_set.py --config configs/default.yaml
python scripts/run_all.py --config configs/default.yaml
streamlit run app/streamlit_app.py
pytest
```

Commands that require the dataset will fail clearly if `data/raw/creditcard.csv` is missing. They will not substitute another dataset or invent results.

## Main Artifacts

- Architecture: [docs/architecture.md](/Users/olatunjioye/Documents/Fraud%20Detection%20Agent/docs/architecture.md)
