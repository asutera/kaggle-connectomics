# Authors: Aaron Qiu <zqiu@ulg.ac.be>,
#          Antonio Sutera <a.sutera@ulg.ac.be>,
#          Arnaud Joly <a.joly@ulg.ac.be>,
#          Gilles Louppe <g.louppe@ulg.ac.be>,
#          Vincent Francois <v.francois@ulg.ac.be>
#
# License: BSD 3 clause
from __future__ import division, print_function, absolute_import

import numpy as np


def min_diagonal(X):
    np.fill_diagonal(X, X.min())
    return X


def min_max(X):
    X_scale = X.ravel() - X.min()
    X_scale /= X_scale.max()
    return X_scale.reshape(X.shape)


def scale(X):
    return min_max(min_diagonal(X))
