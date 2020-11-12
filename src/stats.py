from typing import List

import numpy as np

from bandit import BanditEntry
from experiment import Experiment


def mean_true_time(result: List[List[BanditEntry]]):
    return np.mean([len(trial) for trial in result])


def mean_tpr(non_null_ct, result: List[List[BanditEntry]]):
    rejection_sets = [[metrics['rejections'] for _, _, metrics in trial]
                      for trial in result]

    truths = set(range(non_null_ct))
    true_pos_ct = [[
        len(set(rejections).intersection(truths)) for rejections in trial
    ] for trial in rejection_sets]
    return np.mean(np.stack(true_pos_ct), axis=1) / truths


def mean_fdr(non_null_ct, result: List[List[BanditEntry]]):
    rejection_sets = [[metrics['rejections'] for _, _, metrics in trial]
                      for trial in result]

    truths = set(range(non_null_ct))
    fdps = [[
        len(set(rejections).difference(truths)) / len(set(rejections))
        for rejections in trial
    ] for trial in rejection_sets]
    max_length = max([len(trial) for trial in fdps])

    fdrs = []
    for idx in range(max_length):
        fdr = np.mean([trial[idx] for trial in fdps if len(trial) > idx])
        fdrs.append(fdr)

    return np.array(fdrs)


class Stats:
    def __init__(self, exps: List[Experiment], results, accumulators=None):
        self.exps = exps
        self.results = results
        self.accumulators = accumulators

    @property
    def mean_true_times(self):
        if self.accumulators is not None:
            means = []
            for metrics, accums in zip(self.results, self.accumulators):
                total = 0
                for metric, accum in zip(metrics, accums):
                    if metric:
                        total += len(metric)
                    else:
                        total += accum['runtime']
                means.append(total / len(accums))
            return means
        else:
            return [mean_true_time(result) for result in self.results]

    @property
    def mean_tprs(self):
        return [
            mean_tpr(exp.bandit.non_null_ct, result)
            for exp, result in zip(self.exps, self.results)
        ]

    @property
    def mean_fdrs(self):
        return [
            mean_fdr(exp.bandit.non_null_ct, result)
            for exp, result in zip(self.exps, self.results)
        ]
