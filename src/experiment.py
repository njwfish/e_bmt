from typing import Any, Dict, List, Tuple

from copy import deepcopy
import os
import pickle

# import multiprocess
from multiprocess import get_context

import numpy as np
from tqdm import tqdm

from agents.base import Agent, get_sampler, get_rejector
from agents.sampling import EUCB, PUCB
from bandit import Bandit
from trial import Trial

ExpResult = List[Dict[str, Any]]


class Experiment:
    set_keys = [
        'rejector_name', 'rejector_kwargs', 'sampler_name', 'sampler_kwargs'
    ]

    def __init__(self, name: str, bandit: Bandit, agent: Agent, rounds: int,
                 trials: int, seed: int, alpha: float):
        self.name = name
        self.bandit = bandit
        self.agent = agent
        self.rounds = rounds
        self.trials = trials
        self.seed = seed
        self.alpha = alpha

    def make_trials(self):
        return [
            Trial(f'{self.name}_trial={trial}', self.bandit, self.agent,
                  self.rounds, self.seed + trial, self.alpha)
            for trial in range(self.trials)
        ]

    @staticmethod
    def run_exps(exps: List['Experiment'],
                 accumulators,
                 threads,
                 display_progress_bar=False,
                 save_along=None) -> List[Tuple['Experiment', ExpResult]]:
        exp_res_pairs = []
        filtered_exps = [
            exp for exp in exps if (save_along is None) or (
                not os.path.exists(f'{save_along}/{exp.name}.pkl'))
        ]
        with get_context('spawn').Pool(threads) as p:
            for exp in tqdm(filtered_exps,
                            desc="Number of experiments completed"):
                all_trials = [(trial, accumulators)
                              for trial in exp.make_trials()]
                res_iter = p.imap(lambda args: Trial.run_trial(*args),
                                  all_trials,
                                  chunksize=1)
                if display_progress_bar:
                    res = list(
                        tqdm(res_iter, total=len(all_trials), leave=False))
                else:
                    res = list(res_iter)

                if save_along is not None:
                    with open(f'{save_along}/{exp.name}.pkl', 'wb') as out_f:
                        pickle.dump((exp, res), out_f)

                exp_res_pairs.append((exp, res))
        return exp_res_pairs
