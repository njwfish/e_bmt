from itertools import product
import json
import os
import pickle
import re
import shutil

import click
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

from accumulators import get_accum, init_e_threshs, plot_e_threshs
from experiment import Experiment
from loader import load_experiments
from plots import *

matplotlib.rc('font', size=11)
matplotlib.rc('figure', dpi=300)


@click.group()
def main():
    pass


def repair_old_exp(exp):
    """Set default name for bandits without bandit name.

    :param exp: experiment object
    :return: reparied experiment object
    """
    repair = False
    if not hasattr(exp.bandit, 'name'):
        # set default exp name for older experiments
        exp.bandit.name = 'constant_2'
        repair = True
    return repair


def layout(fig, ax, midpoint):
    """Fix layout/legend of matplotlib figure to place legend at midpoint.

    :param fig: Matplotlib figure
    :param ax: axes of plot
    :param midpoint: midpoint to place legend
    """
    handles, labels = ax.get_legend_handles_labels()
    labels = [label.replace("Uniform", "Uni") for label in labels]
    fig.legend(handles,
               labels,
               loc="upper center",
               bbox_to_anchor=(0, midpoint, 1, 1 - midpoint),
               ncol=2,
               columnspacing=0.5)
    fig.tight_layout(rect=(0.083, 0, 1 - 0.083, midpoint))


@main.command()
@click.option('--exp_path',
              multiple=True,
              type=click.Path(),
              help="path to experiment")
@click.option('--exp_dir',
              multiple=True,
              type=click.Path(),
              help="path to directory of experiments")
@click.option('--out_dir',
              type=click.Path(),
              help="path to directory to output all results to")
@click.option('--repair/--no-repair',
              default=False,
              help="whether the input experiments need to be repaired")
@click.option('--rel_method',
              type=str,
              default=None,
              help="the label of the method to compare other methods to")
@click.option(
    '--style_map',
    type=str,
    default=None,
    help="path to style map in json format (map of label names to styles)")
@click.option(
    '--log_scale/--no_log_scale',
    type=bool,
    default=False,
    help=
    "true puts the plot on a logarithmic scale, and otherwise false does not")
def parse_results(exp_path: click.Path, exp_dir: click.Path,
                  out_dir: click.Path, repair: bool, rel_method: str,
                  style_map: str, log_scale: bool):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    exps = [pickle.load(open(path, 'rb')) for path in exp_path]
    dir_paths = [
        os.path.join(dir_path, path) for dir_path in exp_dir
        for path in os.listdir(dir_path) if re.fullmatch(r'.*\.pkl', path)
    ]
    dir_exps = [pickle.load(open(dir_path, 'rb')) for dir_path in dir_paths]
    if repair:
        for path, (exp, res) in zip(dir_paths, dir_exps):
            if repair_old_exp(exp):
                pickle.dump((exp, res), open(path, 'wb'))

    all_exps = exps + dir_exps

    bandit_ct = len({exp.bandit.name for exp, _ in all_exps})

    figsize = (3, 3)
    midpoint = 0.8

    fig_list = []
    ax_list = []
    for _ in range(bandit_ct):
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()
        fig_list.append(fig)
        ax_list.append(ax)
        ax.set_xscale('log')
    if style_map is None:
        style_map = cmap_style_fn_factory('tab10')
    else:
        style_map_back = json.load(open(style_map))
        for value in style_map_back.values():
            if isinstance(value['linestyle'], list):
                value['linestyle'] = (value['linestyle'][0],
                                      tuple(value['linestyle'][1]))

        style_map = lambda x: style_map_back[x]
    bandit_names = arms_vs_true_time_plot(ax_list, all_exps, style_map)
    for bandit_name, fig, ax in zip(bandit_names, fig_list, ax_list):
        layout(fig, ax, midpoint)
        fig.savefig(os.path.join(out_dir, f'abs_{bandit_name}.png'))
    if rel_method is not None:
        fig_list = []
        ax_list = []
        for _ in range(bandit_ct):
            fig = plt.figure(figsize=figsize)
            ax = plt.gca()
            fig_list.append(fig)
            ax_list.append(ax)
            ax.set_xscale('log')
            if log_scale:
                ax.set_yscale('log')
        bandit_names = arms_vs_true_time_plot(ax_list,
                                              all_exps,
                                              style_map,
                                              rel_method=rel_method)
        for bandit_name, fig, ax in zip(bandit_names, fig_list, ax_list):
            layout(fig, ax, midpoint)
            fig.savefig(
                os.path.join(out_dir,
                             f'rel_{bandit_name},rel_method={rel_method}.png'))

    # fig = plt.figure(figsize=(6, 4))
    # ax = plt.gca()
    # arms_vs_true_time_plot(ax,
    #                        all_exps,
    #                        cmap_style_fn_factory('tab10'),
    #                        rel_method='EUCB_EBH')
    # fig.legend()
    # fig.tight_layout()
    # fig.savefig(os.path.join(out_dir, 'arms_v_true_time_rel.png'))

    # for idx, (exp, result, accumulator) in enumerate(
    #         zip(stats.exps, stats.results, stats.accumulators)):
    #     if isinstance(exp.agent.sampler, EUCB):
    #         for trial in range(exp.trials):
    #             e_val_series = accumulator[trial]['e-values']
    #             for step, ((arm, _, _), log_e_values) in enumerate(
    #                     zip(result[trial], e_val_series)):
    #                 fig = plt.figure(figsize=(6, 4))
    #                 ax = plt.gca()
    #                 e_rejection_perf_plot(ax, exp.bandit, log_e_values)
    #                 fig.savefig(
    #                     f'{out_dir}/e_values_exp={idx}_trial={trial}_step={step}.png'
    #                 )
    #                 plt.close(fig)


cur_accums = ['mean_true_time']
accum_dict = {key: get_accum(key) for key in cur_accums}


@main.command()
@click.argument('config_path', type=click.Path())
@click.argument('out_dir', type=click.Path())
@click.option('--processes',
              type=int,
              default=1,
              help="number of processes to use for concurrent processing")
@click.option(
    '--dry-run/--no-dry-run',
    default=False,
    help="whether the run is a dry run (:code:`True`) or not (:code:`False`).")
def exp(config_path: click.Path, out_dir: click.Path, processes: int,
        dry_run: bool):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    exps = load_experiments(config_path)
    results = []

    if dry_run:
        for exp in exps:
            print(f'name={exp.name},seed={exp.seed}')

    else:
        shutil.copyfile(config_path, f'{out_dir}/exp_config.json')
        exp_res_pairs = Experiment.run_exps(exps,
                                            accum_dict,
                                            processes,
                                            display_progress_bar=True,
                                            save_along=out_dir)
        for exp, res in exp_res_pairs:
            with open(f'{out_dir}/{exp.name}.pkl', 'wb') as out_f:
                pickle.dump((exp, res), out_f)


if __name__ == '__main__':
    from multiprocess import set_start_method
    set_start_method("spawn")
    main()
