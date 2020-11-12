from typing import Any, Callable, Dict, Tuple

from agents.base import Agent
from bandit import AccumNext, Bandit

AccumInit = Callable[['Trial'], Any]
Accumulator = Tuple[AccumInit, AccumNext]


class Trial:
    def __init__(self, name: str, bandit: Bandit, agent: Agent, rounds,
                 seed: int, alpha: float):
        self.name = name
        self.bandit = bandit
        self.agent = agent
        self.rounds = rounds
        self.seed = seed
        self.alpha = alpha

    @staticmethod
    def run_trial(trial, accumulators: Dict[str, Accumulator]):
        trial.agent.set_params(trial.bandit.arms, trial.alpha, trial.seed,
                               trial.seed)
        trial.agent.reset()
        init_accums = {
            label: (init(trial), fn)
            for label, (init, fn) in accumulators.items()
        }
        res = trial.bandit.run(trial.agent, trial.rounds, trial.seed,
                               init_accums)
        return res
