from typing import Any, Callable, List, Tuple
import os

import matplotlib.pyplot as plt
import numpy as np

from agents.sampling import EUCB, EBH, PUCB
from bandit import AccumNext, Bandit, RoundInfo
from trial import AccumInit, Accumulator, Trial
from plots import e_rejection_perf_plot
from utils import create_register_fn

_ACCUM_REGISTRY = {}


def register_accum(name):
    def assign_pair(pair):
        _ACCUM_REGISTRY[name] = pair

    return assign_pair


def get_accum(name):
    return _ACCUM_REGISTRY[name]


def list_accum_maker(get_fn: Callable[[RoundInfo], Any]) -> Accumulator:
    def init(trial: Trial) -> List[Any]:
        return []

    def accum(existing: List[Any], round_info: RoundInfo) -> [Any]:
        existing.append(get_fn(round_info))
        return existing

    register_accum(get_fn.__name__)((init, accum))
    return (init, accum)


@list_accum_maker
def rejection_set(round_info: RoundInfo):
    return round_info[4].rejector.reject()


@list_accum_maker
def e_values(round_info: RoundInfo):
    return round_info[4].rejector.log_e_values


register_accum('mean_true_time')((lambda trial: 0,
                                  lambda _, round_info: round_info[0] + 1))


def tpr_accum(_: None, round_info: RoundInfo):
    bandit, agent = round_info[3], round_info[4]
    rejections = agent.rejector.reject()
    true_rej_ct = np.sum(np.array(rejections) < bandit.non_null_ct)
    return true_rej_ct / bandit.non_null_ct


register_accum('tpr')((lambda trial: None, tpr_accum))


def init_e_threshs(out_dir):
    def init(trial):
        out_subdir = f'{out_dir}/{trial.name}'
        if not os.path.exists(out_subdir):
            os.makedirs(out_subdir)
        return out_subdir

    return init


def plot_e_threshs(state, round_info):
    out_dir = state
    cur_round, arm, reward, bandit, agent = round_info

    prefix = f'{out_dir}/round_{cur_round}'
    if isinstance(agent.rejector, EBH):
        fig = plt.figure(figsize=(6, 4))
        ax = plt.gca()
        e_rejection_perf_plot(ax, bandit, agent.rejector.log_e_values)
        ax.axvspan(arm - 0.5, arm + 0.5, color='grey', alpha=0.4)
        fig.savefig(f'{prefix}_rejector.png')
        plt.close(fig)
    if isinstance(agent.sampler, EUCB) or isinstance(agent.sampler, PUCB):
        fig = plt.figure(figsize=(6, 4))
        ax = plt.gca()
        e_rejection_perf_plot(ax, bandit, agent.sampler.ucb)
        if isinstance(agent.sampler, PUCB):
            e_rejection_perf_plot(ax, bandit, agent.sampler.means)
        ax.axvspan(arm - 0.5, arm + 0.5, color='grey', alpha=0.4)
        fig.savefig(f'{prefix}_sampler.png')
        plt.close(fig)
    return out_dir
