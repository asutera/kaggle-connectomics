# Authors: Aaron Qiu <zqiu@ulg.ac.be>,
#          Antonio Sutera <a.sutera@ulg.ac.be>,
#          Arnaud Joly <a.joly@ulg.ac.be>,
#          Gilles Louppe <g.louppe@ulg.ac.be>,
#          Vincent Francois <v.francois@ulg.ac.be>
#
# License: BSD 3 clause
from __future__ import division, print_function, absolute_import

from itertools import chain

import numpy as np
from sklearn.externals.joblib import Parallel, delayed, cpu_count

from utils import scale


def _partition_X(X, n_jobs):
    """Private function used to partition X between jobs."""
    n_nodes = X.shape[1]

    # Compute the number of jobs
    n_jobs = min(cpu_count() if n_jobs == -1 else n_jobs, n_nodes)

    # Partition estimators between jobs
    n_node_per_job = (n_nodes // n_jobs) * np.ones(n_jobs, dtype=np.int)
    n_node_per_job[:n_nodes % n_jobs] += 1
    starts = np.cumsum(n_node_per_job)

    return n_jobs, [0] + starts.tolist()


def _parallel_count(X, start, end):
    """Private function used to compute a batch of score within a job."""
    count = np.zeros((end - start, X.shape[1]))

    for index, jx in enumerate(range(start, end)):
        X_jx_bot = X[:-1, jx] + 0.2
        X_jx_top = X[:-1, jx] + 0.5

        for j in range(X.shape[1]):
            if j == jx:
                continue

            count[index, j] = ((X[1:, j] > X_jx_bot) &
                               (X[1:, j] < X_jx_top)).sum()

    return count


def make_prediction_directivity(X, threshold=0.12, n_jobs=1):
    """Score neuron connectivity using a precedence measure

    Parameters
    ----------
    X : numpy array of shape (n_samples, n_nodes)
        Fluorescence signals

    threshold : float, (default=0.11)
        Threshold value for hard thresholding filter:
        x_new[i] = x[i] if x[i] >= threshold else 0.

    n_jobs : integer, optional (default=1)
        The number of jobs to run the algorithm in parallel.
        If -1, then the number of jobs is set to the number of cores.

    Returns
    -------
    score : numpy array of shape (n_nodes, n_nodes)
        Pairwise neuron connectivity score.

    """

    # Perform filtering
    X_new = np.zeros((X.shape))
    for i in range(1, X.shape[0] - 1):
        for j in range(X.shape[1]):
            X_new[i, j] = (X[i, j] + 1 * X[i - 1, j] + 0.8 * X[i - 2, j] +
                           0.4 * X[i - 3, j])

    X_new = np.diff(X_new, axis=0)
    thresh1 = X_new < threshold * 1
    thresh2 = X_new >= threshold * 1
    X_new[thresh1] = 0
    X_new[thresh2] = pow(X_new[thresh2], 0.9)

    # Score directivity
    n_jobs, starts = _partition_X(X, n_jobs)
    all_counts = Parallel(n_jobs=n_jobs)(
        delayed(_parallel_count)(X_new, starts[i], starts[i + 1])
        for i in range(n_jobs))
    count = np.vstack(list(chain.from_iterable(all_counts)))

    return scale(count - np.transpose(count))
