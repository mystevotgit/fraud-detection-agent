from __future__ import annotations

import pandas as pd
import pytest

from fraud_agent.data.validate_data import EXPECTED_COLUMNS


@pytest.fixture()
def tiny_creditcard_df() -> pd.DataFrame:
    rows = []
    for idx in range(12):
        row = {column: 0.0 for column in EXPECTED_COLUMNS}
        row["Time"] = float(idx)
        row["Amount"] = float(10 + idx)
        row["Class"] = 1 if idx in {2, 9} else 0
        row["V1"] = 2.0 if row["Class"] == 1 else -0.5
        row["V2"] = -1.0 if row["Class"] == 1 else 0.2
        rows.append(row)
    return pd.DataFrame(rows, columns=EXPECTED_COLUMNS)

