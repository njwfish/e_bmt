import numpy as np

from agents.evidence.base import evidence, Evidence, _EVIDENCE_REGISTRY


@evidence('EtoP')
class EtoP(Evidence):
    """P-values gotten by calibrating e-values to p-values by taking 1 / e."""

    def __init__(self, e_evidence, e_kwargs):
        """
        :e_evidence: type of e-value based :code:`agents.evidence.base` registry
        :e_kwargs: kwargs for constructing e-value"""
        self.e_evidence_name = e_evidence
        self.e_kwargs = e_kwargs
        self.e_evidence = _EVIDENCE_REGISTRY[e_evidence](**e_kwargs)

    def set_params(self, arms: int, alpha: float, seed) -> None:
        """Set bandit and error parameters
        :arms: number of arms
        :alpha: threshold of correction
        :seed: seed for randomization"""
        super(EtoP, self).set_params(arms, alpha, seed)
        self.e_evidence.set_params(arms, alpha, seed)
        self.maxes = np.zeros(arms)

    def reset(self) -> None:
        """Reset evidence module to initial, blank state."""
        self.e_evidence.reset()
        self.maxes = np.zeros(self.arms)

    def update(self, i: int, x: float) -> None:
        """Update e-values based on rewards.

        :i: index of arm
        :x: reward of arm
        """
        self.e_evidence.update(i, x)
        self.maxes[i] = max(self.maxes[i], self.e_evidence.values(i))

    def values(self, i: int) -> None:
        """Returns p-values at arm i
        :i: index of arm
        :return: ith p-value at the current time step"""
        return self.maxes[i]

    def all_values(self) -> np.ndarray:
        """Returns p-values at all arms
        :return: all p-values at current time step"""
        return self.maxes
