from typing import Any, Callable, Dict, List, Tuple, Union
import numpy as np
from tqdm import tqdm

from agents.base import Agent

SHOW_PROGRESS = False

_ARM_SIGNAL_REGISTRY = {}
_DIST_REGISTRY = {}


def arm_signal(name: str):
    """Register a function as generating signals for non-null arms.

    :param name: name to register function under

    :return: annotation function
    """

    def register(fn):
        _ARM_SIGNAL_REGISTRY[name] = fn
        return fn

    return register


def arm_dist(name):
    """Register a function generating reward distribution from signal
    parameter.

    :param name: name to register function under

    :return: annotation function
    """

    def register(fn):
        _DIST_REGISTRY[name] = fn
        return fn

    return register


@arm_signal('constant')
def constant_arm_signal(signal: float, non_null_param: Union[str, int],
                        arms: int) -> Tuple[np.ndarray, int]:
    """Create constant signal value for set of arms.

    :param signal: The signal parameter to be set for non-null arms.
    :param non_null_param: Number of non-null arms as a function of number of arms. Either :code:`sqrt`, :code:`log`, or integer specifying number of arms"
    :param arms: number of arms in bandit

    :return: array of signal values and number of non-null arms
    """

    if non_null_param == 'sqrt':
        ct = np.floor(np.sqrt(arms))
    elif non_null_param == 'log':
        ct = max(np.floor(np.log(arms)), 1)
    elif non_null_param < 1:
        ct = np.floor(non_null_param * arms)
    else:
        ct = non_null_param
    return np.repeat(signal, ct), int(ct)


@arm_signal('scaling')
def scaling_arm_signal(max_signal: float, min_signal: float, scale: float,
                       arms: int) -> Tuple[np.ndarray, int]:
    """Create scaling signal value for set of arms. Number of non-null arms is
    number of arms required to reduce signal to 0.

    :param max_signal: max signal parameter for non-null arm (should be nonnegative and larger than :code:`min_signal`).
    :param min_signal: min signal parameter for non-null arm (should be nonnegative).
    :param scale: scale parameter i.e. how much tne signal changes per non-null arm
    :param arms: number of arms in bandit

    :return: array of signal values and number of non-null arms
    """

    idx_ub = (max_signal - min_signal) * arms / scale
    if idx_ub == np.floor(idx_ub):
        max_idx = int(idx_ub) - 1
    else:
        max_idx = np.floor(idx_ub)
    indices = np.arange(0, max_idx + 1)
    return max_signal - indices / (arms / scale), len(indices)


@arm_dist('gaussian')
def gaussian_arm(signal: float) -> float:
    """Sample Gaussian random variable with mean= :code:`signal` and variance
    1.

    :param signal: signal parameter of arm

    :return: random sample from standard normal with mean :code:`signal`
    """
    return np.random.normal(signal, 1)

@arm_dist('bernoulli')
def bernoulli_arm(signal: float) -> float:
    """Sample Bernoulli random variable with mean= :code:`signal`.

    :param signal: signal parameter of arm

    :return: random sample from bernoulli with mean :code:`signal`
    """
    return np.random.binomial(1, signal)


RoundInfo = Tuple[
    int, int, float, 'Bandit',
    Agent]  # round_idx, arm idx, reward, Bandit, Agent playing bandit
AccumNext = Callable[[Any, RoundInfo], Any]


class Bandit:
    """Multi-armed bandit class.

    :param name: Name of bandit - for id use in later analysis but not vital to code
    :param arms: Number of arms
    :param null_signal: value of signal for null arms
    :param arm_signal: string specifying method (:code:`@arm_signal`) for generating signals for non-null arms
    :param arm_kwargs: kwargs passed to method for non-null arm signals specified by :code:`arm_signal`
    :param arm_dist: string specifying reward distribution (:code:`@arm_dist`) of each arm
    :param graph_bandit: is graph bandit or not (models combinatorial bandit)

    :return: Bandit instance
    """

    def __init__(self,
                 name: str,
                 arms: int,
                 null_signal: int,
                 arm_signal: str,
                 arm_kwargs: Dict[str, Any],
                 arm_dist: str,
                 graph_bandit: bool = False):
        self.name = name
        self.arms = arms
        self.null_signal = null_signal
        self.arm_signal = arm_signal
        self.arm_kwargs = arm_kwargs
        self.arm_dist = arm_dist
        self.graph_bandit = graph_bandit
        self._make_signal_array()

    def to_dict(self) -> Dict[str, Any]:
        """Convert bandit instance to dictionary.

        :return: dict of bandit parameters
        """
        return {
            'arms': self.arms,
            'null_signal': self.null_signal,
            'arm_signal': self.arm_signal,
            'arm_kwargs': self.arm_kwargs,
            'arm_dist': self.arm_dist,
        }

    @staticmethod
    def from_dict(in_dict: Dict[str, Any]) -> "Bandit":
        """Initialize bandit for parameters in dictionary.

        :param in_dict: dictionary of parameters

        :return: Bandit instance
        """
        return Bandit(**in_dict)

    def _make_signal_array(self):
        """Make array of signals for all arms.

        :meta private:
        """

        non_null_signals, self.non_null_ct = _ARM_SIGNAL_REGISTRY[
            self.arm_signal](**(self.arm_kwargs), arms=self.arms)
        self.signal_arr = np.concatenate(
            (non_null_signals,
             np.zeros(self.arms - self.non_null_ct, dtype=np.float)))

    def initialize_seed(self, seed: int):
        """Initialize instance to use specific seed.

        :param seed: seed value
        """
        self.seed = seed
        np.random.seed(seed)
        self.internal_random_state = np.random.get_state()

    def pull_arm(self, i: int) -> float:
        """Pull an arm and get reward from that arm.

        :param i: index of arm being pulled
        :return: reward
        """
        np.random.set_state(self.internal_random_state)
        result = _DIST_REGISTRY[self.arm_dist](self.signal_arr[i])
        self.internal_random_state = np.random.get_state(
            self.internal_random_state)
        return result

    def run(self,
            agent: Agent,
            rounds: Union[str, int, float],
            seed: int,
            accumulators: Dict[str, Tuple[Any, AccumNext]],
            verbose=False) -> Dict[str, Any]:
        """Run the bandit algorithm with an agent acting on it.

        :param agent: agent algorithm interacting with bandit
        :param rounds:
            the stopping rule - a string input will have the
            algorithm stop when the non-nulls arms are all rejected,
            otherwise an integer input will be the number of rounds
            to run.
        :param seed: seed for random initialization
        :param accumulators: key mapping to accumulating functions that calculate statistics of the run
        :param verbose: verbose messaging of round info

        :return: dictionary mapping of key to value generated by accumulators based on :code:`accumulators` key
        """
        self.initialize_seed(seed)
        accumulated_values = {
            key: base
            for key, (base, _) in accumulators.items()
        }

        def do_round(round_idx):
            if verbose:
                print(round_idx, agent.sampler.ucb)
            arm = agent.select_arm()
            if self.graph_bandit:
                for i in arm:
                    reward = self.pull_arm(i)
                    agent.update_state(i, reward)
            else:
                reward = self.pull_arm(arm)
                agent.update_state(arm, reward)

            for key, (_, fn) in accumulators.items():
                accumulated_values[key] = fn(
                    accumulated_values[key],
                    (round_idx, arm, reward, self, agent))

        if isinstance(rounds, int):
            for idx in range(rounds):
                do_round(idx)
        else:
            tpr_target = 1. if isinstance(rounds, str) else rounds
            idx = 0
            if SHOW_PROGRESS:
                pbar = tqdm(leave=False)
            while agent.rejector.get_true_rej_ct(
                    self.non_null_ct) < tpr_target * self.non_null_ct:
                do_round(idx)
                if SHOW_PROGRESS:
                    pbar.update(1)
                idx += 1
        return accumulated_values
