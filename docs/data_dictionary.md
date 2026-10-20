# Data Dictionary

Dataset: Kaggle Credit Card Fraud Detection dataset attributed to Worldline and the ULB Machine Learning Group.

| Column | Meaning |
|---|---|
| `Time` | Seconds elapsed between this transaction and the first transaction in the dataset. |
| `V1`-`V28` | PCA-transformed anonymized transaction features. Original feature semantics are unavailable for confidentiality reasons. |
| `Amount` | Transaction amount. Used for amount-weighted experimental fraud-loss assumptions. |
| `Class` | Target label: `0` legitimate, `1` fraudulent. |

Important interpretation note: because `V1`-`V28` are anonymized PCA features, explanations can identify predictive influence but cannot provide ordinary business-causal interpretations.

