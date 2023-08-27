from copy import deepcopy
from typing import List, Optional

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor


class PropensityModel:
    def __init__(self, base_model: Optional = None):
        if base_model:
            assert hasattr(base_model, 'predict_proba')
            self.base_model = base_model
        else:
            self.base_model = LGBMClassifier()
        self.is_fitted = False

    def fit(self,
            data: pd.DataFrame,
            features: List[str],
            t_col: str = 'treatment'):
        assert set(data[t_col].values) == {0, 1}
        self.base_model.fit(data[features].values, data[t_col].values)
        self.is_fitted = True

    def predict(self,
                data: pd.DataFrame,
                features: List[str]) -> np.ndarray:
        assert self.is_fitted
        propensity_score = self.base_model.predict_proba(data[features].values)[:, 1]
        return propensity_score


class SModel:
    def __init__(self, base_model: Optional = None):
        if base_model:
            self.base_model = base_model
        else:
            self.base_model = LGBMRegressor()
        self.is_fitted = False

    def fit(self,
            data: pd.DataFrame,
            features: List[str],
            y_col: str,
            t_col: str = 'treatment'):
        assert set(data[t_col].values) == {0, 1}
        self.base_model.fit(data[features+[t_col]].values, data[y_col].values)
        self.is_fitted = True

    def predict(self,
                data: pd.DataFrame,
                features: List[str]) -> np.ndarray:
        assert self.is_fitted
        data_treated = data.assign(treatment=1)[features+['treatment']].values
        data_not_treated = data.assign(treatment=0)[features+['treatment']].values
        uplift = self.base_model.predict(data_treated) - self.base_model.predict(data_not_treated)
        return uplift


class TModel:
    def __init__(self, base_model: Optional = None, models: Optional[List] = None):
        if models:
            assert len(models) == 2
            self.control_model, self.target_model = deepcopy(models)
        elif base_model:
            self.control_model, self.target_model = deepcopy(base_model), deepcopy(base_model)
        else:
            self.control_model, self.target_model = LGBMRegressor(), LGBMRegressor()

        self.is_fitted = False

    def fit(self,
            data: pd.DataFrame,
            features: List[str],
            y_col: str,
            t_col: str = 'treatment'):
        assert set(data[t_col].values) == {0, 1}
        data_control = data.loc[data[t_col] == 0]
        data_target = data.loc[data[t_col] == 1]
        self.control_model.fit(data_control[features].values, data_control[y_col].values)
        self.target_model.fit(data_target[features].values, data_target[y_col].values)
        self.is_fitted = True

    def predict(self,
                data: pd.DataFrame,
                features: List[str]) -> np.ndarray:
        assert self.is_fitted
        uplift = self.target_model.predict(data[features].values) - self.control_model.predict(data[features].values)
        return uplift


class XLearner:
    def __init__(self,
                 base_model: Optional = None,
                 models: Optional[List] = None,
                 weight: Optional[float] = None,
                 ):
        """
        Parameters:
            weight: second-level models weight. Should be in [0, 1] set.
            If control >> target, should use first-level model fitted on control, so weight = 0.
            If target >> control, so weight = 1.

        """
        if models:
            self.control_model, self.target_model, self.model0, self.model1 = deepcopy(models)
        elif base_model:
            self.control_model = deepcopy(base_model)
            self.target_model = deepcopy(base_model)
            self.model0 = deepcopy(base_model)
            self.model1 = deepcopy(base_model)
        else:
            self.control_model = LGBMRegressor()
            self.target_model = LGBMRegressor()
            self.model0 = LGBMRegressor()
            self.model1 = LGBMRegressor()

        self.is_fitted = False
        self.weight = weight
        self.use_propensity_model = False
        if not weight:
            self.use_propensity_model = True
            self.propensity_model = PropensityModel()

    def fit(self,
            data: pd.DataFrame,
            features: List[str],
            y_col: str,
            t_col: str = 'treatment'):
        assert set(data[t_col].values) == {0, 1}
        data_control = data.loc[data[t_col] == 0]
        data_target = data.loc[data[t_col] == 1]
        self.control_model.fit(data_control[features].values, data_control[y_col].values)
        self.target_model.fit(data_target[features].values, data_target[y_col].values)

        uplift_control = self.target_model.predict(data_control[features].values) - data_control[y_col].values
        uplift_target = data_target[y_col].values - self.control_model.predict(data_target[features].values)

        self.model0.fit(data_control[features].values, uplift_control)
        self.model1.fit(data_target[features].values, uplift_target)

        if self.use_propensity_model:
            self.propensity_model.fit(data, features, t_col)
        self.is_fitted = True

    def predict(self,
                data: pd.DataFrame,
                features: List[str]) -> np.ndarray:
        assert self.is_fitted

        uplift0 = self.model0.predict(data[features].values)
        uplift1 = self.model1.predict(data[features].values)

        if self.use_propensity_model:
            self.weight = self.propensity_model.predict(data, features)

        uplift = self.weight * uplift0 + (1-self.weight) * uplift1

        return uplift
