from typing import List, Optional

import numpy as np
import pandas as pd 


def make_full_dates(first: str, last: str, date_name: str) -> pd.DataFrame:
    dates = pd.date_range(first, last, freq=pd.DateOffset(months=1), closed="left")
    dates = pd.DataFrame(index=dates).reset_index().rename(columns={"index": date_name})
    dates[date_name] = dates[date_name].astype(str)
    return dates


def calc_lags(
    df: pd.DataFrame,
    key_col: str,
    target_col: str,
    lags: List,
    sort_col: Optional[str] = None,
) -> pd.DataFrame:
    if sort_col:
        df = df.sort_values(by=[key_col, sort_col], ascending=True)
    for lag in lags:
        df[f'{target_col}__lag_{lag}'] = df.groupby(key_col)[target_col].shift(lag)

    return df


def calc_box_cox(x, lmbda=0):
    if lmbda == 0:
        return np.log(x)
    else:
        return (x ** lmbda - 1) / lmbda


def calc_inverse_box_cox(x, lmbda=0):
    if lmbda == 0:
        return np.exp(x)
    else:
        x = lmbda * x + 1
        return np.sign(x) * np.abs(x) ** (1 / lmbda)
