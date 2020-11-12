from typing import Any, Dict

import numpy as np

from agents.evidence.base import evidence, Evidence
from agents.evidence.e.lambdastrategy import _LAMBDA_STRAT_REGISTRY


@evidence("PMH")
class PMH(Evidence):
    """Predictably mixed Hoeffding e-values."""

    def __init__(self, mean: float, var: float, lambda_strat: str,
                 lambda_kwargs: Dict[str, Any]):
        """
        :mean: null mean
        :var: variance of reward
        :lambda_start: how to construct the series of lambda_t for the predictable mixing
        :lambda_kwargs: kwargs for the lambda construction method"""
        self.mean = mean
        self.var = var
        self.lambda_strat_name = lambda_strat
        self.lambda_kwargs = lambda_kwargs
        self.lambda_strat = _LAMBDA_STRAT_REGISTRY[lambda_strat](
            **lambda_kwargs)

    def set_params(self, arms: int, alpha: float, seed: float) -> None:
        """Set bandit and error parameters
        :arms: number of arms
        :alpha: threshold of correction
        :seed: seed for randomization"""
        super(PMH, self).set_params(arms, alpha, seed)
        self.lambda_strat.set_params(arms, alpha, seed)

    def _cur_e_value(self, cur_lambda, x):
        """Calculates the PM-H e-value."""
        return cur_lambda * (x -
                             self.mean) - self.var * np.square(cur_lambda) / 2

    def update(self, i: int, x: float) -> None:
        """Update e-values based on rewards.

        :i: index of arm
        :x: reward of arm
        """
        cur_lambda = self.lambda_strat.get_lambda(i)
        self.t[i] += 1
        self.log_e_values[i] += self._cur_e_value(cur_lambda, x)
        self.lambda_strat.update(i, x)

    def values(self, i) -> float:
        """Get e-values so far for an arm
        :i: index of arm
        :return: e-value of arm i"""
        return np.exp(self.log_e_values[i])

    def all_values(self) -> np.ndarray:
        """Get all e-values of all arms
        :return: e-values of all arms"""
        return np.exp(self.log_e_values)

    def reset(self) -> None:
        """Reset evidence module to initial, blank state."""
        self.log_e_values = np.zeros(self.arms)
        self.t = np.zeros(self.arms)
        self.lambda_strat.reset()
