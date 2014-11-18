#!/usr/bin/env python

# Authors: Aaron Qiu <zqiu@ulg.ac.be>,
#          Antonio Sutera <a.sutera@ulg.ac.be>,
#          Arnaud Joly <a.joly@ulg.ac.be>,
#          Gilles Louppe <g.louppe@ulg.ac.be>,
#          Vincent Francois <v.francois@ulg.ac.be>
#
# License: BSD 3 clause

from __future__ import division, print_function, absolute_import

import argparse
from itertools import product

import numpy as np

from PCA import make_simple_inference, make_tuned_inference

from directivity import make_prediction_directivity

# Cache accelerator may be removed to save disk space
from sklearn.externals.joblib import Memory
memory = Memory(cachedir="cachedir", verbose=0)
np.loadtxt = memory.cache(np.loadtxt)


if __name__ == "__main__":
    # Process arguments
    parser = argparse.ArgumentParser(description='Perform Connectomics '
                                                 'modelling')
    parser.add_argument('-f', '--fluorescence', type=str, required=True,
                        help='Path to the fluorescence file')
    parser.add_argument('-p', '--position', type=str, required=False,
                        help='Path to the network position file')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Path of the output/prediction file')
    parser.add_argument('-n', '--network', type=str, required=True,
                        help='Network name')
    parser.add_argument('-m', '--method', type=str, required=False,
                        default='simple', help='Simplified or tuned method?')
    parser.add_argument('-d', '--directivity', type=int, required=False,
                        default=0,
                        help='''Take into account information about
                                directivity?''')
    parser.add_argument('-s', '--submission', type=int, required=False,
                        default=0, help='Submission file?')
    args = vars(parser.parse_args())

    # Loading data
    print('Loading data...')
    X = np.loadtxt(args["fluorescence"], delimiter=",")

    X = np.asfortranarray(X, dtype=np.float32)
    # pos = np.loadtxt(args["position"], delimiter=",")

    # # Producing the prediction matrix ##

    if args["method"] == 'tuned':
        y_pca = make_tuned_inference(X)
    else:
        y_pca = make_simple_inference(X)

    if bool(args["directivity"]):
        print('Using information about directivity...')
        y_directivity = make_prediction_directivity(X)
        # Perform stacking
        score = 0.997 * y_pca + 0.003 * y_directivity
    else:
        score = y_pca

    if bool(args["submission"]):
        # Generate the submission file ##
        with open(args["output"], 'w') as fname:
            fname.write("NET_neuronI_neuronJ,Strength\n")

            for i, j in product(range(score.shape[0]), range(score.shape[1])):
                line = "{0}_{1}_{2},{3}\n".format(args["network"],
                                                  i + 1, j + 1, score[i, j])
                fname.write(line)

        print("Infered connectivity score is saved at %s"
              % args["output"])
