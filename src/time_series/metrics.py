from typing import Callable

import numpy as np
import pandas as pd 


def calc_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calc and return percentage mean absolute deviation."""
    return np.nansum(abs(y_pred - y_true)) / np.nansum(abs(y_true))


def calc_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calc and return percentage mean absolute deviation."""
    return np.nansum((y_pred - y_true) / np.nansum(y_true))


def calc_cum_metric(df: pd.DataFrame, time_col: str, y_col: str, yhat_col: str, metric: Callable):
    times, metrics = [], []
    for ds in sorted(df[time_col].unique()):
        metric_ = metric(df.loc[df[time_col] <= ds, y_col].values,
                         df.loc[df[time_col] <= ds, yhat_col].values)
        times.append(ds)
        metrics.append(metric_)

    return metrics, times
