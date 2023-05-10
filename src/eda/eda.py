from functools import reduce
import os
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.time_series import metrics



def plot_ecdf(
        data: np.ndarray,
        figsize: Tuple[int, int] = (5, 3),
        title: str = 'ecdf',
):
    """Build empirical cumulative distribution function plot."""
    x, y = np.sort(data), np.arange(1, len(data)+1) / len(data)
    plt.figure(figsize=figsize)
    plt.scatter(x, y)
    plt.grid()
    plt.title(title)
    plt.show()


def plot_forecast_metrics_by_group(
        df: pd.DataFrame,
        key_col: str,
        y_col: str = 'y',
        yhat_col: str = 'yhat',
        fig_title: str = 'Forecast metrics by group',
        fig_ylabel: str = 'BIAS, WAPE',
        fig_path: Optional[str] = None,
        fig_size: Tuple[int] = (15, 10),
        xticks_rotation: float = 90,
):
    """
    Plots BIAS and WAPE metrics aggregated by groups
    :param df: input data
    :param key_col: column to group data by
    :param y_col: factual target column name
    :param yhat_col: forecasted target column name
    :param fig_title: figure title
    :param fig_ylabel: y name
    :param fig_path: path to save figure
    :param fig_size: size of figure
    :param xticks_rotation: x ticks rotation
    """
    df_metrics = (
        df.groupby(key_col)
        .apply(lambda x: metrics.calc_bias(x[y_col].values, x[yhat_col].values))
        .reset_index()
        .rename(columns={0: 'bias'})
    )
    df_metrics = (
        df_metrics
        .merge(
            df.groupby(key_col)
            .apply(lambda x: metrics.calc_wape(x[y_col].values, x[yhat_col].values))
            .reset_index()
            .rename(columns={0: 'wape'})
        )
        .sort_values(by='bias')
    )
    plt.style.use('ggplot')
    plt.rc('figure', figsize=fig_size)
    plt.bar(np.arange(df_metrics.shape[0]), df_metrics['bias'].values, color='red')
    plt.plot(df_metrics[key_col].values, df_metrics['wape'].values, color='green')
    plt.xticks(fontsize=12, rotation=xticks_rotation)
    plt.axhline(0.0, color='blue')
    plt.axhline(0.1, color='black')
    plt.axhline(-0.1, color='black')
    plt.axhline(0.2, color='black', linestyle='--')
    plt.axhline(-0.2, color='black', linestyle='--')
    plt.ylabel(fig_ylabel)
    plt.title(fig_title)
    if fig_path:
        plt.savefig(os.path.join(fig_path))
    else:
        plt.show()

        
def plot_train_val_pred(
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        df_pred: pd.DataFrame,
        date_col: str = 'ds',
        y_col: str = 'y',
        yhat_col: str = 'yhat',
        plot_title: str = 'Train, forecast and validation',
        legend_loc: str = 'upper left',
        plot_validation_start_date: bool = True,
        vline_col: str = 'green',
):
    plt.figure()
    sns.lineplot(y=y_col, x=date_col, data=df_train, color='blue', lw=2, label='Train')
    sns.lineplot(y=y_col, x=date_col, data=df_val, color='black', lw=2, label='Validation')
    sns.lineplot(y=yhat_col, x=date_col, data=df_pred, color='red', lw=2, label='Forecast')
    
    if plot_validation_start_date:
        plt.axvline(x=df_val[date_col].min(), color=vline_col, label='Validation start date', linestyle='dashed')

    plt.title(plot_title)
    plt.legend(loc=legend_loc)
    plt.show()

    
def plot_forecast_dashboard(
    df: pd.DataFrame,
    time_col: str,
    y_col: str,
    yhat_col: str,
    figsize: Tuple[int] = (20, 11),
):
    cum_metric, cum_time = metrics.calc_cum_metric(df, time_col, y_col, yhat_col,  metrics.calc_wape)
    cum_bias, _ = metrics.calc_cum_metric(df, time_col, y_col, yhat_col,  metrics.calc_bias)

    fig = plt.figure(figsize=figsize)
    fig.tight_layout(pad=10)

    plt.subplot(221)
    sns.lineplot(cum_time, cum_bias)
    plt.title('cumulative BIAS')
    plt.ylabel('BIAS')

    plt.subplot(222)
    sns.lineplot(cum_time, cum_metric)
    plt.title('cumulative WAPE')
    plt.ylabel('WAPE')
    
    plt.subplot(223)
    plt.scatter(df[y_col].values, df[yhat_col].values)
    plt.axline([0, 0], [np.nanmax(df[y_col].values), np.nanmax(df[yhat_col].values)], color='black')
    plt.title('prediction on target')
    plt.xlabel(y_col)
    plt.ylabel(yhat_col)
                        
    plt.subplot(224)
    plt.scatter(df[time_col].values, df[yhat_col].values-df[y_col].values)
    plt.axhline(0, color='black')
    plt.title('residuals on time')
    plt.ylabel('residual')
                        
    plt.show() 

    
def count_nans(df: pd.DataFrame) -> pd.DataFrame:
    df_len = df.shape[0]
    nan_cnt = [df[col].isna().sum() for col in df.columns]
    nan_ratio = [x / df_len for x in nan_cnt]
    return pd.DataFrame({"colname": df.columns, "nan_cnt": nan_cnt, "nan_ratio": nan_ratio})


def calc_distinct_values(df: pd.DataFrame, key: List[str], cols: Optional[List[str]] = None) -> pd.DataFrame:
    if cols is None:
        cols = set(df.columns).difference(set(key))
    
    df_cnt = []
    for col in cols:
        tmp = (
            df.groupby(key)[col]
            .unique()
            .map(len)
            .reset_index()
            .rename(columns={col: f"{col}_distinct_cnt"})
        )
        df_cnt.append(tmp)

    df_cnt = reduce(lambda x, y: x.merge(y, on=key), df_cnt)
    return df_cnt


def plot_feature_importance_weight(
    model,
    importance_type="gain",
    figsize=(7,3),
    title="Feature importance",
    xlabel="gain weight",
    ylabel="feature",
    top_n=None,
):
    booster = model.booster_   
    importance = booster.feature_importance(importance_type=importance_type)
    feature_name = booster.feature_name()
    importance /= importance.sum()    
    tuples = sorted(zip(feature_name, importance), key=lambda x: -x[1])
    xs = [x[1] for x in tuples]
    ys = [x[0] for x in tuples]
    if top_n:
        xs = xs[:top_n]
        ys = ys[:top_n]
        
    plt.figure(figsize=figsize)
    sns.barplot(xs, ys, color="royalblue")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.show()
