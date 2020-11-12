import matplotlib.pyplot as plt
import numpy as np


def metric_vs_step(names, metrics):
    fig = plt.figure(figsize=(8, 6))
    ax = plt.gca()

    max_steps = 0
    for name, metric in zip(names, metrics):
        steps = metric.shape[0]
        max_steps = max(steps, max_steps)
        ax.plot(np.arange(steps), metric, label=name)
    ax.set_xlim(0, steps)
    ax.set_ylim(0, max_val)
    ax.set_xlabel('$t$')
    ax.set_ylabel(name)
    fig.legend()
    fig.savefig(f'{arms}_{name}.png')
