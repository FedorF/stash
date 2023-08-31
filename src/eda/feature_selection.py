import numpy as np
import pandas as pd


def generate_bins(data, col, num_bins=20, duplicated_bins='raise'):
    return pd.qcut(data[col].values, num_bins, retbins=False, duplicates=duplicated_bins)


def calc_woe(data, feat_col, target_col, num_bins=20, keep_bins_col=False, duplicated_bins='raise'):
    assert set(data[target_col].values) == {0, 1}
    data[f'{feat_col}__bins'] = generate_bins(data, feat_col, num_bins, duplicated_bins)
    df_bins = data.groupby([f'{feat_col}__bins', target_col])[target_col].count().unstack()
    df_bins[0] /= df_bins[0].sum()
    df_bins[1] /= df_bins[1].sum()
    df_bins['woe'] = (df_bins[1]/df_bins[0]).map(np.log)
    if not keep_bins_col:
        data.drop(columns=f'{feat_col}__bins', inplace=True)
    return df_bins


def calc_iv(data, feat_col, target_col, num_bins=20, keep_bins_col=False, duplicated_bins='raise'):
    df_bins = calc_woe(data, feat_col, target_col, num_bins, keep_bins_col, duplicated_bins)
    df_bins['prob_diff'] = df_bins[1] - df_bins[0]
    df_bins['iv'] = df_bins['prob_diff'] * df_bins['woe']
    iv = df_bins['iv'].sum()
    return iv, df_bins


def calc_features_iv(data, features, target_col, num_bins=20, duplicated_bins='raise'):
    ivs = []
    for col in features:
        try:
            iv, _ = calc_iv(data, col, target_col, num_bins, False, duplicated_bins)
        except:
            raise Exception(f'problem with feature: {col}')
        ivs.append(iv)
    return pd.DataFrame(data={'feature': features, 'iv': ivs})
