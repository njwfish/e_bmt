import numpy as np

from agents.evidence.base import evidence, Evidence
from agents.phi import get_p_value, _BRACKET_MAP, _PHI_REGISTRY


@evidence("LCB")
class LCB(Evidence):
    """LCB p-value used by JJ that is applicable to bounded RVs on [0, 1].

    Note that all p-values output 1 / p, so "EBH" can be applied to get
    BH in a purely operational definition
    """

    def __init__(self, mean: float, running_min=True, phi='kaufmann'):
        """
        :mean: null mean
        :running_min: use the running min if true
        :phi: choice of phi function to define the LCB p-value"""
        self.mean = mean
        self.running_min = running_min
        self.phi = phi

    def set_params(self, arms: int, alpha: float, seed: float) -> None:
        """Set bandit and error parameters
        :arms: number of arms
        :alpha: threshold of correction
        :seed: seed for randomization"""
        super(LCB, self).set_params(arms, alpha, seed)
        self.sample_means = np.zeros(arms)
        self.t = np.zeros(arms)
        self.mins = np.ones(arms)
        self.tolerance = self.alpha / (arms * 10)

    def reset(self) -> None:
        """Reset evidence module to initial, blank state."""
        self.sample_means = np.zeros(self.arms)
        self.t = np.zeros(self.arms)
        self.mins = np.ones(self.arms)

    def update(self, i: int, x: float) -> None:
        """Update p-values based on rewards.

        :i: index of arm
        :x: reward of arm
        """
        self.sample_means[i] = (self.sample_means[i] * self.t[i] +
                                x) / (self.t[i] + 1)
        self.t[i] += 1
        if self.sample_means[i] > self.mean + _PHI_REGISTRY[self.phi](
                self.t[i], _BRACKET_MAP[self.phi][1]):
            p = get_p_value(self.sample_means[i], self.mean, self.t[i],
                            self.phi, self.tolerance)
            if not self.running_min or p < self.mins[i]:
                self.mins[i] = p

    def values(self, i: int) -> float:
        """Returns p-values at arm i
        :i: index of arm
        :return: ith p-value at the current time step"""
        return 1 / self.mins[i]

    def all_values(self) -> np.ndarray:
        """Returns p-values at all arms
        :return: all p-values at current time step"""
        return 1 / self.mins
