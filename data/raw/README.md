# Raw Dataset Setup

Download `creditcard.csv` from the Kaggle Credit Card Fraud Detection dataset:

<https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud>

Place the file here:

```text
data/raw/creditcard.csv
```

The dataset contains anonymized transactions from European cardholders in September 2013. The features `V1` through `V28` are PCA-transformed, while `Time`, `Amount`, and `Class` are directly provided. `Class` is the target label: `0` for legitimate and `1` for fraud.

Kaggle may require authentication or manual browser download. This project will not silently substitute another dataset.

