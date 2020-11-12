import os

import matplotlib.pyplot as plt

from agents.sampling import EUCB, EBH
from plots import e_rejection_perf_plot


def rejection_set(agent):
    return agent.rejector.reject()


def e_values(cur_vals, data):
    _, _, _, agent = data
    cur_vals.append(agent.rejector.log_e_values)
    return cur_vals


def init_e_threshs(out_dir):
    def init(bandit, agent, rounds, trial, seed, alpha, name):
        return f'{out_dir}/{name}/{trial}', 0

    return init


def plot_e_threshs(state, data):
    out_dir, cur_round = state
    arm, reward, bandit, agent = data

    out_subdir = f'{out_dir}'
    if not os.path.exists(out_subdir):
        os.makedirs(out_subdir)
    prefix = f'{out_subdir}/round_{cur_round}'
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
        ax.axvspan(arm - 0.5, arm + 0.5, color='grey', alpha=0.4)
        fig.savefig(f'{prefix}_sampler.png')
        plt.close(fig)
    return out_dir, cur_round + 1
