import bisect
from copy import copy

import numpy as np
from scipy.special import lambertw
from agents.base import Sampler, Rejector, sampling_strategy, rejection_strategy
from agents.evidence.base import _EVIDENCE_REGISTRY
from agents.phi import _PHI_REGISTRY
from utils import create_register_fn

_CAL_REGISTRY = {}
_E_REGISTRY = {}

cal_strategy = create_register_fn(_CAL_REGISTRY)
e_strategy = create_register_fn(_E_REGISTRY)


def lambda_t(t, alpha):
    return np.minimum(np.sqrt(2 * np.log(2 / alpha) / (t * np.log(t + 1))), 1)


@e_strategy('sub_gaussian')
def sub_gaussian_log_e(x, t, alpha, mu_0, var=1):
    return lambda_t(t, alpha) * (x - mu_0) - var * np.square(lambda_t(
        t, alpha)) / 2


@cal_strategy('power_cal')
def power_cal(p_value, kappa):
    assert 0 <= kappa and kappa <= 1
    return kappa * np.power(p_value, kappa - 1)


@cal_strategy('shafer_cal')
def shafer_cal(p_value):
    return np.power(p_value, -0.5) - 1


@cal_strategy('average_cal')
def average_cal(p_value):
    num = 1 - p_value + p_value * np.log(p_value)
    den = p_value * np.square(np.log(p_value))
    return num / den


@cal_strategy('big_boost')
def big_boost(p_value):
    e_value = 1 / p_value  # only b/c we know p_value is 1 / e_value
    num = (e_value**2) * np.log(2)
    den = (1 + e_value) * np.square(np.log(1 + e_value))
    return num / den


@sampling_strategy('Uniform')
class UniformSampling(Sampler):
    def set_params(self, arms, alpha, seed):
        super(UniformSampling, self).set_params(arms, alpha, seed)
        np.random.seed(seed)
        self.state = np.random.get_state()

    def update_state(self, idx, reward):
        pass

    def select_arm(self, rejected_indices):
        np.random.set_state(self.state)
        res = np.random.randint(self.arms)
        self.state = np.random.get_state()
        return res

    def reset(self):
        pass


@sampling_strategy('Elimination')
class EliminationSampling(Sampler):
    def set_params(self, arms, alpha, seed):
        super(EliminationSampling, self).set_params(arms, alpha, seed)
        np.random.seed(seed)
        self.state = np.random.get_state()
        self.rejected_indices = set()

    def update_state(self, idx, reward):
        pass

    def select_arm(self, rejected_indices):
        np.random.set_state(self.state)
        self.state = np.random.get_state()
        res = np.random.randint(self.arms)
        while res in self.rejected_indices:
            res = np.random.randint(self.arms)
        self.state = np.random.get_state()
        self.rejected_indices.update(rejected_indices)
        return res

    def reset(self):
        pass


@sampling_strategy('EUCB')
class EUCB(Sampler):
    def _init_(self, sequence='pm', sequence_kwargs={}):
        self.sequence = sequence
        self.sequence_kwargs = sequence_kwargs

    def reset(self):
        self.ucb = np.zeros(self.arms)
        self.evidence = np.zeros(self.arms)
        self.lambda_sum = np.zeros(self.arms)
        self.cgf_sum = np.zeros(self.arms)
        self.ts = np.zeros(self.arms)
        self.round = 0

    def update_state(self, i, x):
        self.ts[i] += 1
        lam = lambda_t(self.ts[i], self.alpha)
        self.evidence[i] += lam * x
        self.lambda_sum[i] += lam
        self.cgf_sum[i] += np.square(lam) / 2
        self.ucb[i] = (self.evidence[i] + np.log(2 / self.alpha) +
                       self.cgf_sum[i]) / self.lambda_sum[i]
        self.round += 1

    def select_arm(self, rejected_indices):
        if self.round < self.arms:
            return self.round

        m = np.zeros(self.arms, dtype=bool)
        m[rejected_indices] = True
        a = np.ma.array(self.ucb, mask=m)
        return a.argmax()


@rejection_strategy('EBHNew')
class EBHNew(Rejector):
    _correction_map = {
        'jj':
        lambda alpha, k: alpha / (6.4 * np.log(36 / alpha)),
        'su':
        lambda alpha, k:
        (-1 * alpha) / lambertw(z=-1 * alpha / np.exp(1), k=-1
                                ),  # inverse of alog(1 + log(1 / a)) = alpha
        'dep':
        lambda alpha, k: alpha / np.log(k)
    }

    def __init__(self, evidence, evidence_kwargs, correction=None):
        self.evidence_name = evidence
        self.evidence_kwargs = evidence_kwargs
        self.evidence = _EVIDENCE_REGISTRY[evidence](**evidence_kwargs)
        self.correction = correction

    def set_params(self, arms, alpha, seed):
        super(EBHNew, self).set_params(arms, alpha, seed)
        if self.correction is None:
            self.correct_alpha = alpha
        else:
            self.correct_alpha = EBHNew._correction_map[self.correction](alpha,
                                                                         arms)
        self.evidence.set_params(arms, self.correct_alpha, seed)

        self.rejections = []
        self.true_ct = None
        self.true_rej_ct = None

    def reset(self):
        self.evidence.reset()

        self.rejections = []
        self.true_ct = None
        self.true_rej_ct = None

    def _rej_thresh(self, rej_size):
        return self.arms / (self.correct_alpha * rej_size)

    def _update_true_rej_ct(self):
        self.true_rej_ct = len([idx < self.true_ct for idx in self.rejections])

    def _ebh(self):
        sorted_indices = np.argsort(self.evidence.all_values())[::-1]

        end = 0
        for ct, i in enumerate(sorted_indices):
            # if self.evidence.values(i) < self._rej_thresh(ct + 1):
            if self.evidence.values(i) * (
                    ct + 1) / self.arms >= 1 / self.correct_alpha:
                end = ct + 1
            else:
                break
        self.rejections = list(sorted_indices[:end])
        if self.true_ct is not None:
            self._update_true_rej_ct()

    def update_state(self, i, x):
        self.evidence.update(i, x)
        # if self.evidence.values(i) >= self._rej_thresh(
        #         len(self.rejections) + 1):
        if (len(self.rejections) + 1
            ) * self.evidence.values(i) / self.arms >= 1 / self.correct_alpha:
            self._ebh()

    def reject(self):
        return copy(self.rejections)

    def get_true_rej_ct(self, true_ct):
        if self.true_ct is None or self.true_ct != true_ct:
            self.true_ct = true_ct
            self._update_true_rej_ct()
        return self.true_rej_ct


@rejection_strategy('EBH')
class EBH(Rejector):
    def __init__(self, max_cal=None, max_cal_kwargs={}):
        self.e_strategy = "sub_gaussian"
        self.max_cal = max_cal
        self.max_cal_kwargs = max_cal_kwargs
        self.true_ct = None
        self.true_rej_ct = None

    def reset(self):
        self.mu_0 = 0
        self.log_e_values = np.zeros(shape=self.arms)
        self.t = np.ones(shape=self.arms)
        self.rejections = []
        self.reject_ct = 0
        self.round = 0
        self.true_ct = None
        self.true_rej_ct = None

        self.max_log_e_values = np.zeros(shape=self.arms)

    def initialize_state(self, initial_rewards):
        for i in range(self.arms):
            self.log_e_values[i] += _E_REGISTRY[self.e_strategy](
                initial_rewards[i], self.t[i], self.alpha, self.mu_0)
        self._ebh()

    def update_state(self, i, x):
        not_rejected = i not in self.reject()

        self.t[i] += 1
        self.log_e_values[i] += _E_REGISTRY[self.e_strategy](x, self.t[i],
                                                             self.alpha,
                                                             self.mu_0)

        self.max_log_e_values[i] = max(self.log_e_values[i],
                                       self.max_log_e_values[i])
        if not_rejected:
            self._step_ebh(i)
        else:
            self._ebh()
        self.round += 1

    def _get_e_value(self, i):
        if self.max_cal is None:
            return np.exp(self.log_e_values[i])
        else:
            return _CAL_REGISTRY[self.max_cal](p_value=min(
                1, np.exp(-self.max_log_e_values[i])),
                                               **self.max_cal_kwargs)

    def _step_ebh(self, i):
        """Assumes arm i has not been rejected yet."""
        if (len(self.rejections) +
                1) * self._get_e_value(i) / self.arms >= 1 / self.alpha:
            # self.rejections.append(i)
            # self.reject_ct += 1
            # self._add_rej_to_true_rej_ct(i)
            self._ebh()

    def _ebh(self):
        if self.max_cal is None:
            sorted_indices = np.argsort(self.log_e_values)[::-1]
        else:
            sorted_indices = np.argsort(self.max_log_e_values)[::-1]

        open_end = 0
        for ct, i in enumerate(sorted_indices):
            if self._get_e_value(i) * (ct + 1) / self.arms >= 1 / self.alpha:
                open_end = ct + 1
            else:
                break
        self.rejections = list(sorted_indices[:open_end])
        self.reject_ct = len(self.rejections)
        self._recount_true_rej_ct()

    def reject(self):
        return self.rejections

    def _add_rej_to_true_rej_ct(self, rej_idx):
        if self.true_ct is not None and rej_idx < self.true_ct:
            self.true_rej_ct += 1

    def _recount_true_rej_ct(self):
        if self.true_ct is not None:
            self.true_rej_ct = len(
                [i for i in self.reject() if i < self.true_ct])

    def get_true_rej_ct(self, true_ct):
        if self.true_ct is not None and self.true_ct == true_ct:
            return self.true_rej_ct
        else:
            self.true_ct = true_ct
            self._recount_true_rej_ct()
            return self.true_rej_ct


# def phi(t, delta):
#     return np.sqrt((2 * np.log(1 / delta) + 6 * np.log(np.log(1 / delta)) +
#                     3 * np.log(np.log(np.exp(1) * t / 2))) / t)


@sampling_strategy('PUCB')
class PUCB(Sampler):
    def __init__(self, phi='kaufmann'):
        self.phi = phi

    def reset(self):
        self.ucb = np.zeros(self.arms)
        self.means = np.zeros(self.arms)
        self.ts = np.zeros(self.arms)
        self.round = 0

    def update_state(self, i, x):
        self.ts[i] += 1
        self.means[i] = (self.means[i] * (self.ts[i] - 1) + x) / self.ts[i]
        self.ucb[i] = self.means[i] + _PHI_REGISTRY[self.phi](self.ts[i],
                                                              self.alpha)
        self.round += 1

    def select_arm(self, rejected_indices):
        if self.round < self.arms:
            return self.round
        m = np.zeros(self.arms, dtype=bool)
        m[rejected_indices] = True
        a = np.ma.array(self.ucb, mask=m)
        return np.argmax(a)


@rejection_strategy('PBH')
class PBH(Sampler):

    _compliance_map = {
        'jj': lambda alpha: alpha / (6.4 * np.log(36 / alpha)),
        'su': lambda alpha:
        (-1 * alpha) / lambertw(z=-1 * alpha / np.exp(1), k=-1
                                )  # inverse of alog(1 + log(1 / a)) = alpha
    }

    def __init__(self, running_min=False, compliance='su'):
        self.running_min = running_min
        self.compliance = compliance

    def set_params(self, arms, alpha, seed):
        super(PBH, self).set_params(arms, alpha, seed)
        self.thresh = PBH._compliance_map[self.compliance](alpha)
        self.true_ct = None
        self.true_rej_ct = None

    def reset(self):
        self.means = np.zeros(shape=self.arms)
        self.t = np.ones(shape=self.arms)  # number of pulls of each arm so far
        self.rejections = []
        self.reject_ct = 0
        self.round = 0
        self.true_ct = None
        self.true_rej_ct = None

    def initialize_state(self, initial_rewards):
        for i in range(self.arms):
            self.means[i] = initial_rewards[i]
        self._bh()

    def update_state(self, i, x):
        not_rejected = i not in self.reject()
        self.means[i] = (self.means[i] * self.t[i] + x) / (self.t[i] + 1)
        self.t[i] += 1
        if not_rejected:
            self._step_bh(i)
        else:
            self._bh()
        self.round += 1

    def _step_bh(self, i):
        if self.means[i] - phi(
                self.t[i],
            (self.reject_ct + 1) * self.thresh / self.arms) >= 0:
            # self.reject_ct += 1
            # self.rejections.append(i)
            # self._add_rej_to_true_rej_ct(i)
            self._bh()

    def _bh(self):
        frontier = list(range(self.arms))
        new_frontier = []
        rejections = []
        for i in range(1, self.arms + 1):
            for idx in frontier:
                if self.means[idx] - phi(self.t[idx],
                                         i * self.thresh / self.arms) < 0:
                    new_frontier.append(idx)
                else:
                    rejections.append(idx)
            if len(rejections) < i:
                break
            else:
                frontier = new_frontier
                new_frontier = []
        self.rejections = rejections
        self.reject_ct = len(self.rejections)
        self._recount_true_rej_ct()

    def reject(self):
        return self.rejections

    def _add_rej_to_true_rej_ct(self, rej_idx):
        if self.true_ct is not None and rej_idx < self.true_ct:
            self.true_rej_ct += 1

    def _recount_true_rej_ct(self):
        if self.true_ct is not None:
            self.true_rej_ct = len(
                [i for i in self.reject() if i < self.true_ct])

    def get_true_rej_ct(self, true_ct):
        if self.true_ct == true_ct:
            return self.true_rej_ct
        else:
            self.true_ct = true_ct
            self._recount_true_rej_ct()
            return self.true_rej_ct
