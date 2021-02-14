import numpy as np

from .rustem import test, butwhy, EM


def restructure(em):
    # assign each data point to its closest cluster
    max_vals = np.argmax(em.get_gamma(), axis=1)
    return em.get_mu()[max_vals, :]
