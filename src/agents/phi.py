import numpy as np
from scipy.optimize import root_scalar

from utils import create_register_fn

_PHI_REGISTRY = {}

phi = create_register_fn(_PHI_REGISTRY)


@phi('kaufmann')
def kaufmann(t, delta):
    return np.sqrt((2 * np.log(1 / delta) + 6 * np.log(np.log(1 / delta)) +
                    3 * np.log(np.log(np.exp(1) * t / 2))) / t)


@phi('IS')
def IS(t, delta):
    return np.sqrt(
        (2.89 * np.log(np.log(2.041 * t)) + 2.065 * np.log(4.983 / delta)) / t)


_BRACKET_MAP = {'kaufmann': (0, 1 / np.e), 'IS': (0, 1)}


def get_p_value(mu_hat, mu_0, t, phi, tolerance):

    phi_fn = _PHI_REGISTRY[phi]

    def root_fn(p):
        if p == 0:
            return np.NINF
        return mu_hat - (mu_0 + phi_fn(t, p))

    result = root_scalar(root_fn,
                         bracket=_BRACKET_MAP[phi],
                         method='bisect',
                         xtol=tolerance,
                         x0=0.01)

    assert result.converged, result
    p = result.root
    return p
