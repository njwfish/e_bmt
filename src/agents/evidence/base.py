import numpy as np

from utils import create_register_fn

_EVIDENCE_REGISTRY = {}
evidence = create_register_fn(_EVIDENCE_REGISTRY)


class Evidence:
    """Abstract class for keeping track of evidence accumulated over run of
    bandit e.g. p-values or e-values and their corresponding BH/e-BH rejection
    sets."""

    def set_params(self, arms: int, alpha: float, seed: float) -> None:
        """Set params for evidence module.

        :arms: number of arms to keep track of evidence for
        :alpha: level of correction for BH/e-BH
        :seed: random seed
        """
        self.alpha = alpha
        self.arms = arms
        self.seed = seed

    def update(self, i: int, x: float) -> None:
        """Update with reward from arm.

        :i: index of arm
        :x: reward of arm
        """
        pass

    def values(self, i: int) -> float:
        """Get value for arm i.

        :i: index of arm
        :return: value
        """
        pass

    def all_values(self) -> np.ndarray:
        """Get all value for all arms.

        :return: values of all arms
        """
        pass

    def reset(self) -> None:
        """Reset evidence module to initial, blank state."""
        pass
