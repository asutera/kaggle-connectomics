# Authors: Aaron Qiu <zqiu@ulg.ac.be>,
#          Antonio Sutera <a.sutera@ulg.ac.be>,
#          Arnaud Joly <a.joly@ulg.ac.be>,
#          Gilles Louppe <g.louppe@ulg.ac.be>,
#          Vincent Francois <v.francois@ulg.ac.be>
#
# License: BSD 3 clause

from __future__ import division, print_function, absolute_import

import numpy as np
from sklearn.decomposition import PCA

from utils import scale


# #########################################
# ####### SIMPLIFIED METHOD ###############
# #########################################

def f1(X):
    return X + np.roll(X, -1, axis=0) + np.roll(X, 1, axis=0)


def f2(X):
    return (X + np.roll(X, 1, axis=0) + 0.8 * np.roll(X, 2, axis=0)
            + 0.4 * np.roll(X, 3, axis=0))


def g(X):
    return np.diff(X, axis=0)


def h(X, threshold=0.11):
    threshold1 = X < threshold * 1
    threshold2 = X >= threshold * 1
    X_new = np.zeros_like(X)
    X_new[threshold1] = 0
    X_new[threshold2] = X[threshold2]
    return X_new


def w(X):
    X_new = X
    Sum_X_new = np.sum(X_new, axis=1)
    Sum4 = Sum_X_new
    normalization = np.max(Sum4)
    for i in range(X_new.shape[0]):
        r = Sum4[i] / normalization
        if Sum4[i] != 0:
            X_new[i, :] = ((X_new[i, :] + 1) ** (1 + (1. / Sum4[i])))
        else:
            X_new[i, :] = 1
    return X_new


def simple_filter(X, LP='f1', threshold=0.11, weights=True):
    if LP == 'f1':
        X = f1(X)
    elif LP == 'f2':
        X = f2(X)
    else:
        raise ValueError("Unknown filter, got %s." % LP)
    X = g(X)
    X = h(X)
    if weights:
        X = w(X)
    return X


def tuned_filter(X, LP='f1', threshold=0.11, weights=True):
    if LP == 'f1':
        X = f1(X)
    elif LP == 'f2':
        X = f2(X)
    elif LP == 'f3':
        X = f3(X)
    elif LP == 'f4':
        X = f4(X)
    else:
        raise ValueError("Unknown filter, got %s." % LP)
    X = g(X)
    X = h(X)
    X = r(X)

    if weights:
        X = w_star(X)
    return X


def make_simple_inference(X):

    print('Making simple inference...')

    t = [0.100, 0.101, 0.102, 0.103, 0.104, 0.105, 0.106, 0.107, 0.108, 0.109,
         0.110, 0.111, 0.112, 0.113, 0.114, 0.115, 0.116, 0.117, 0.118, 0.119,
         0.120, 0.121, 0.122, 0.123, 0.124, 0.125, 0.126, 0.127, 0.128, 0.129,
         0.130, 0.131, 0.132, 0.133, 0.134, 0.135, 0.136, 0.137, 0.138, 0.139,
         0.140, 0.141, 0.142, 0.143, 0.144, 0.145, 0.146, 0.147, 0.148, 0.149,
         0.150, 0.151, 0.152, 0.154, 0.155, 0.156, 0.157, 0.158, 0.159, 0.160,
         0.161, 0.162, 0.163, 0.164, 0.165, 0.166, 0.167, 0.168, 0.169, 0.170,
         0.171, 0.172, 0.173, 0.174, 0.175, 0.176, 0.177, 0.178, 0.179, 0.180,
         0.181, 0.182, 0.183, 0.184, 0.185, 0.186, 0.187, 0.188, 0.189, 0.190,
         0.191, 0.192, 0.193, 0.194, 0.195, 0.196, 0.197, 0.198, 0.199, 0.200,
         0.201, 0.202, 0.203, 0.204, 0.205, 0.206, 0.207, 0.208, 0.209, 0.200,
         0.201, 0.202, 0.203, 0.204, 0.205, 0.206, 0.207, 0.208, 0.209, 0.210]

    weight = 0

    n_samples, n_nodes = X.shape
    y_pred_agg = np.zeros((n_nodes, n_nodes))

    for threshold in t:
        for filtering in ["f1", "f2"]:

            print('Current: %0.3f, %s' % (threshold, filtering))

            X_new = simple_filter(X, LP=filtering, threshold=t, weights=True)
            pca = PCA(whiten=True, n_components=int(0.8 * n_nodes)).fit(X_new)
            y_pred = - pca.get_precision()

            if filtering == 'f1':
                y_pred_agg += y_pred
                weight += 1
            elif filtering == 'f2':
                y_pred_agg += y_pred * 0.9
                weight += 0.9

    return scale(y_pred_agg / weight)

# #########################################
# ########### TUNED METHOD ################
# #########################################


def f3(X):
    return (X + np.roll(X, -1, axis=0) + np.roll(X, -2, axis=0)
            + np.roll(X, 1, axis=0))


def f4(X):
    return (X + np.roll(X, -1, axis=0) + np.roll(X, -2, axis=0)
            + np.roll(X, -3, axis=0))


def r(X):
    return X**0.9


def w_star(X, filtering="f1"):

    X_new = X

    Sum_X_new = np.sum(X_new, axis=1)

    Sum4 = Sum_X_new + 0.5 * np.roll(Sum_X_new, 1)

    normalization = np.max(Sum4)

    for i in range(X_new.shape[0]):

        r = Sum4[i] / normalization

        if filtering == "f1":

            if Sum4[i] > 0 and r < 0.23 and r > 0.05:
                X_new[i, :] = (((X_new[i, :] + 1)
                                ** (1 + (1. / Sum4[i]))) ** 1.9)

            elif Sum4[i] > 0 and r < 0.75:
                X_new[i, :] = (((X_new[i, :] + 1)
                                ** (1 + (1. / Sum4[i]))) ** 1.6)

            elif Sum4[i] != 0:
                X_new[i, :] = (((X_new[i, :] + 1)
                                ** (1 + (1. / Sum4[i]))) ** 1.4)

            else:
                X_new[i, :] = 1

        elif filtering == "f2":

            if Sum4[i] > 0 and r < 0.23 and r > 0.05:
                X_new[i, :] = (((X_new[i, :] + 1)
                                ** (1 + (1. / Sum4[i]))) ** 1.9)

            elif Sum4[i] > 0 and r < 0.75:
                X_new[i, :] = (((X_new[i, :] + 1)
                                ** (1 + (1. / Sum4[i]))) ** 1.6)

            elif Sum4[i] != 0:

                X_new[i, :] = (((X_new[i, :] + 1)
                                ** (1 + (1. / Sum4[i]))) ** 1.4)

            else:
                X_new[i, :] = 1

        elif filtering == "f3":

            if Sum4[i] > 0 and r < 0.22 and r > 0.04:
                X_new[i, :] = (((X_new[i, :] + 1)
                                ** (1 + (1. / Sum4[i]))) ** 1.9)

            elif Sum4[i] > 0 and r < 0.75:
                X_new[i, :] = (((X_new[i, :] + 1)
                                ** (1 + (1. / Sum4[i]))) ** 1.7)

            elif Sum4[i] != 0:
                X_new[i, :] = (((X_new[i, :] + 1)
                                ** (1 + (1. / Sum4[i]))) ** 1.5)

            else:
                X_new[i, :] = 1

        elif filtering == "f4":

            if Sum4[i] > 0 and r < 0.22 and r > 0.08:
                X_new[i, :] = (((X_new[i, :] + 1)
                                ** (1 + (1. / Sum4[i]))) ** 1.9)

            elif Sum4[i] != 0:
                X_new[i, :] = (((X_new[i, :] + 1)
                                ** (1 + (1. / Sum4[i]))) ** 1.5)

            else:
                X_new[i, :] = 1

        else:
            if Sum4[i] != 0:
                X_new[i, :] = (((X_new[i, :] + 1)
                                ** (1 + (1. / Sum4[i]))) ** 1.6)
            else:
                X_new[i, :] = 1

    return X_new


def make_tuned_inference(X):
    print('Making tuned inference...')

    t = [0.100, 0.101, 0.102, 0.103, 0.104, 0.105, 0.106, 0.107, 0.108, 0.109,
         0.110, 0.111, 0.112, 0.113, 0.114, 0.115, 0.116, 0.117, 0.118, 0.119,
         0.120, 0.121, 0.122, 0.123, 0.124, 0.125, 0.126, 0.127, 0.128, 0.129,
         0.130, 0.131, 0.132, 0.133, 0.134, 0.135, 0.136, 0.137, 0.138, 0.139,
         0.140, 0.141, 0.142, 0.143, 0.144, 0.145, 0.146, 0.147, 0.148, 0.149,
         0.150, 0.151, 0.152, 0.154, 0.155, 0.156, 0.157, 0.158, 0.159, 0.160,
         0.161, 0.162, 0.163, 0.164, 0.165, 0.166, 0.167, 0.168, 0.169, 0.170,
         0.171, 0.172, 0.173, 0.174, 0.175, 0.176, 0.177, 0.178, 0.179, 0.180,
         0.181, 0.182, 0.183, 0.184, 0.185, 0.186, 0.187, 0.188, 0.189, 0.190,
         0.191, 0.192, 0.193, 0.194, 0.195, 0.196, 0.197, 0.198, 0.199, 0.200,
         0.201, 0.202, 0.203, 0.204, 0.205, 0.206, 0.207, 0.208, 0.209, 0.200,
         0.201, 0.202, 0.203, 0.204, 0.205, 0.206, 0.207, 0.208, 0.209, 0.210]

    weight = 0

    n_samples, n_nodes = X.shape
    y_pred_agg = np.zeros((n_nodes, n_nodes))

    for threshold in t:
        for filtering in ["f1", "f2", "f3", "f4"]:
            print('Current: %0.3f, %s' % (threshold, filtering))

            X_new = tuned_filter(X, LP=filtering, threshold=t, weights=True)
            pca = PCA(whiten=True, n_components=int(0.8 * n_nodes)).fit(X_new)
            y_pred = - pca.get_precision()

            if filtering == 'f1':
                y_pred_agg += y_pred
                weight += 1
            elif filtering == 'f2':
                y_pred_agg += y_pred * 0.9
                weight += 0.9
            elif filtering == 'f3':
                y_pred_agg += y_pred * 0.01
                weight += 0.01
            elif filtering == 'f4':
                y_pred_agg += y_pred * 0.7
                weight += 0.7

    return scale(y_pred_agg / weight)
