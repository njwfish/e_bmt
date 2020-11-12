import numpy as np
import matplotlib.pyplot as plt
from toolz.itertoolz import groupby

from agents.sampling import EBH, EUCB, lambda_t
from bandit import Bandit
from experiment import ExpResult

# from stats import Stats


def cmap_style_fn_factory(cmap_name):
    style_map = {}
    counter = 0
    cmap_fn = plt.get_cmap(cmap_name)
    linestyles = ['solid', 'dashed', 'dotted', 'dashdot']

    def style_fn(name):
        nonlocal counter
        nonlocal style_map
        nonlocal cmap_fn

        if name not in style_map:
            style_map[name] = {
                'color': cmap_fn(counter),
                'linestyle': linestyles[(counter % len(linestyles))]
            }
            counter += 1
        return style_map[name]

    return style_fn


def arms_vs_true_time_plot(ax_list,
                           exp_res_pairs,
                           style_fn,
                           rel_method=None,
                           bandit_title=True):
    exp_time_pairs = [(exp, np.mean([res['mean_true_time']
                                     for res in results]))
                      for exp, results in exp_res_pairs]

    bandit_name_list = []
    for (bandit_name, bandit_group), ax in zip(
            groupby(lambda x: x[0].bandit.name, exp_time_pairs).items(),
            ax_list):
        print(bandit_name)

        agent_grouped_data = groupby(lambda x: x[0].agent.name, bandit_group)

        agent_times_map = {}
        for agent_name, exp_time_data in agent_grouped_data.items():
            exps, mean_times = zip(*exp_time_data)
            arm_cts = [exp.bandit.arms for exp in exps]
            sorted_idxs = np.argsort(arm_cts)
            agent_times_map[agent_name] = (np.array(arm_cts)[sorted_idxs],
                                           np.array(mean_times)[sorted_idxs])

        if rel_method is not None:
            abs_times = agent_times_map[rel_method][1]
            for agent_name in agent_times_map:
                arm_cts, times = agent_times_map[agent_name]

                min_time_ct = min(times.shape[0], abs_times.shape[0])
                agent_times_map[agent_name] = (arm_cts[:min_time_ct],
                                               times[:min_time_ct] /
                                               abs_times[:min_time_ct])

        for agent_name, (arm_cts, times) in agent_times_map.items():
            ax.plot(arm_cts,
                    times,
                    label=agent_name.replace('_', ' ').replace(
                        'PUCB', 'UCB').replace('EBH',
                                               'e-BH').replace('PBH', 'BH'),
                    **(style_fn(agent_name)))

        ax.set_xlabel('Arms')
        # ax.set_ylabel(f'Time' + (
        #     f' (relative to {rel_method})' if rel_method is not None else ''))
        ax.set_ylabel(f'Time')
        bandit_name_list.append(bandit_name)
    return bandit_name_list


def e_rejection_perf_plot(ax, bandit: Bandit, log_e_values, last_arm=None):
    """Assumes EBH has set_params called w.r.t.

    to bandit params.
    """

    ax.scatter(np.arange(bandit.non_null_ct),
               log_e_values[:bandit.non_null_ct],
               color='red',
               label='non-null')
    ax.scatter(np.arange(bandit.non_null_ct, bandit.arms),
               log_e_values[bandit.non_null_ct:],
               color='blue',
               label='null')
    ax.set_xlabel(f'Arms - total arms: {bandit.arms}')
    ax.set_ylabel(f'Value')
    if last_arm is not None:
        ax.axvspan(last_arm - 0.5, last_arm + 0.5, color='grey', alpha=0.4)


def make_gif(subdir, idx_regex, name):
    paths = [re.fullmatch(idx_regex, path) in os.listdir(subdir)]
    paths = sorted([path for path in paths if path is not None],
                   key=lambda x: int(x.group(1)))
    images = []
    for path in paths:
        file_path = os.path.join(subdir, path)
        images.append(imageio.imread(file_path))
    imageio.mimsave(f'{subdir}/{name}.gif', images)


def width_plot(ax, start, end, alpha):
    e_lambdas = lambda_t(np.arange(start, end), alpha)
    e_width = (np.log(1 / alpha) +
               np.cumsum(np.square(e_lambdas) / 8)) / np.cumsum(e_lambdas)
    p_width = phi(np.arange(start, end), alpha)
    ax.plot(e_width, label='e width')
    ax.plot(p_width, label='p width')


# def trial_vs_time_plot(ax, res: ExpResult):
