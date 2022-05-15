from typing import Any, Dict, List, Union

from utils import create_register_fn

_SAMPLING_REGISTRY = {}
_REJECTION_REGISTRY = {}

# Register as a sampling function
sampling_strategy = create_register_fn(_SAMPLING_REGISTRY)
# Register as a rejection function
rejection_strategy = create_register_fn(_REJECTION_REGISTRY)


def get_sampler(name):
    """Get sampling algorithm by name.

    :param name: name of sampling algorithim
    :return: sampling algorithm constructor
    """
    return _SAMPLING_REGISTRY[name]


def get_rejector(name):
    """Get rejection algorithm by name.

    :param name: name of rejection algorithim
    :return: rejection algorithm constructor
    """
    return _REJECTION_REGISTRY[name]


class Sampler:
    """Base class for sampling algorithms."""

    def select_arm(self, rejection_indices: List[int]) -> int:
        """Abstract method for selecting an arm to sample based on rejected arm
        indices.

        :param rejection_indices: list of indices of arms that have been rejected
        :return: arm index to sample
        """
        raise NotImplementedError

    def set_params(self, arms: int, alpha: float, seed: int):
        """Set parameters of sampling algorithm.

        :param arms: number of arms in bandit
        :param alpha: level of FDR control wanted
        :param seed: random seed for initialization
        """
        self.seed = seed
        self.arms = arms
        self.alpha = alpha

    def update_state(self, idx: int, reward: float) -> None:
        """Update state of sampler with reward.

        :param idx: index of arm being sampled
        :param reward: reward sampled from bandit
        """
        raise NotImplementedError

    def reset(self):
        """Reset sampling algorithm."""
        raise NotImplementedError

    @staticmethod
    def from_dict(in_dict: Dict[str, Any]) -> 'Sampler':
        """Build sampler from dictionary of arguments.

        :param in_dict: dictionry of arguments
        :return: Sampler
        """
        return get_sampler(in_dict['name'])(**(in_dict['kwargs']))


class Rejector:

    def reject(self) -> List[int]:
        raise NotImplementedError

    def get_true_rej_ct(self, true_ct) -> int:
        return len([i for i in self.reject() if i < self.true_ct])

    def set_params(self, arms: int, alpha: float, seed: int):
        self.seed = seed
        self.arms = arms
        self.alpha = alpha

    def update_state(self, idx: int, reward: float) -> None:
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    @staticmethod
    def from_dict(in_dict):
        return get_rejector(in_dict['name'])(**(in_dict['kwargs']))


class Agent:

    def __init__(self,
                 sampler: Sampler,
                 rejector: Rejector,
                 name: Union[None, str] = None):
        self.sampler = sampler
        self.rejector = rejector
        self.name = name

    def set_params(self, arms: int, alpha: float, sampler_seed: int,
                   rejector_seed: int):
        self.sampler.set_params(arms, alpha, sampler_seed)
        self.rejector.set_params(arms, alpha, rejector_seed)

    def select_arm(self):
        arm = self.sampler.select_arm(self.rejector.reject())
        return arm

    def update_state(self, idx: int, reward: float) -> None:
        self.sampler.update_state(idx, reward)
        self.rejector.update_state(idx, reward)

    def reset(self):
        self.sampler.reset()
        self.rejector.reset()

    @property
    def identifier(self):
        if self.name is not None:
            return self.name
        return (self.sampler.name, self.rejector.name)

    @staticmethod
    def from_dict(in_dict, idx=None):
        sampler = Sampler.from_dict(in_dict['sampler'])
        rejector = Rejector.from_dict(in_dict['rejector'])
        name = in_dict[
            'name'] if 'name' in in_dict else f'{sampler.name}_{rejector.name}' + (
                f'_{idx}' if idx is not None else "")
        return Agent(sampler, rejector, name)
