from typing import Any, Dict

import numpy as np

from utils import create_register_fn

_LAMBDA_STRAT_REGISTRY = {}
lambda_strategy = create_register_fn(_LAMBDA_STRAT_REGISTRY)


class LambdaStrategy:
    def set_params(self, arms, alpha, seed):
        self.alpha = alpha
        self.arms = arms
        self.seed = seed

    def update(self, i, x) -> None:
        pass

    def get_lambda(self, i) -> np.ndarray:
        pass

    def reset(self) -> None:
        pass


@lambda_strategy("Hoeffding")
class Hoeffding(LambdaStrategy):
    def __init__(self, var):
        self.var = var

    def set_params(self, arms, alpha, seed):
        super(Hoeffding, self).set_params(arms, alpha, seed)

    def reset(self):
        self.t = np.repeat(
            2, self.arms)  # consistency with series defined by old experiments

    def update(self, i, x):
        self.t[i] += 1

    def get_lambda(self, i):
        return np.minimum(
            np.sqrt((2 / self.var * np.log(2 / self.alpha)) /
                    (self.t[i] * np.log(self.t[i] + 1))), 1)


@lambda_strategy("SampleMean")
class SampleMean(LambdaStrategy):
    def __init__(self, arms, alpha, seed, coef):
        self.coef = coef

    def set_params(self, arms, alpha, seed):
        super(SampleMean, self).set_params(arms, alpha, seed)

        self.means = np.zeros(arms)
        self.t = np.ones(arms)

    def reset(self):
        self.means = np.zeros(self.arms)
        self.t = np.zeros(self.arms)

    def update(self, i, x):
        self.means[i] = (self.means[i] * self.t[i] + x) / (self.t[i] + 1)
        self.t[i] += 1

    def get_lambda(self, i):
        return self.coef * self.means[i]
