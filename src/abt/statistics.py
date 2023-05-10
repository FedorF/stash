from typing import Callable, Tuple

import numpy as np


def bootstrap_confidence_interval(
        target:np.ndarray,
        control: np.ndarray,
        func: Callable = lambda t, c: t-c,
        confidence: float =.95,
        n_iter: int = 1000,
) -> Tuple[float, float]:
    """
    Calculates confidence interval for function of two samples using bootstrap
    :param target: target sample
    :param control: control sample
    :param func: function to apply
    :param confidence: width of interval
    :param n_iter: number of iterations
    """
    n_target = len(target)
    n_control = len(control)

    samples = []
    for i in range(n_iter):
        t = target[np.random.randint(low=0, high=n_target, size=n_target)].mean()
        c = control[np.random.randint(low=0, high=n_control, size=n_control)].mean()
        samples.append(func(t, c))
    
    low = (1 - confidence) / 2
    high = 1 - low
    return np.percentile(samples, 100*low), np.percentile(samples, 100*high)
