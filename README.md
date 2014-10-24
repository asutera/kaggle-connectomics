Kaggle Connectomics
===================

If you use this code, please cite :
Sutera, A., Joly, A., François-Lavet, V., Qiu, Z. A., Louppe, G., Ernst, D., & Geurts, P. (2014). Simple connectome inference from partial correlation statistics in calcium imaging. JMLR: Workshop and Conference Proceedings of ECML2014 (Nancy).

If you are interested in more explanations, have a look at http://orbi.ulg.ac.be/handle/2268/169767 (Mirror: http://arxiv.org/abs/1406.7865).

https://www.kaggle.com/c/connectomics

Team AAAGV (#1)

Contact
-------

- Antonio Sutera <a.sutera@ulg.ac.be>
- Arnaud Joly <a.joly@ulg.ac.be>
- Aaron Qiu <zqiu@ulg.ac.be>
- Gilles Louppe <g.louppe@ulg.ac.be>
- Vincent François <v.francois@ulg.ac.be>


Dependencies
------------

The following programs and packages were used for the contest:

    - Python 2.7
    - NumPy >= 1.6.2
    - SciPy >= 0.11
    - scikit-learn == master branch (last update, the hash commit was `8d04380d474723467b5a717328efd0c9fc5bd898`)

with appropriate blas and lapack binding such as MKL, accelerate or ATLAS.
In order to test the code, we recommend you to use the Anaconda python
distribution (https://store.continuum.io/cshop/anaconda/).

Code ran on MacOsx 10.9.2 and Linux (version 2.6.18-194.26.1.el5
(brewbuilder@norob.fnal.gov) (gcc version 4.1.2 20080704 (Red Hat 4.1.2-48))
1 SMP Tue Nov 9 12:46:16 EST 2010).


How to train your model
-----------------------

No model is learnt to produce the connectivity score matrix.


How to make predictions on a new test set
-----------------------------------------
In order to reproduce the result, you can launch the main.py file.
The usage is the following:

    usage: main.py [-h] -f FLUORESCENCE [-p POSITION] -o OUTPUT -n NETWORK

    optional arguments:
      -h, --help            show this help message and exit
      -f FLUORESCENCE, --fluorescence FLUORESCENCE
                            Path to the fluorescence file
      -p POSITION, --position POSITION
                            Path to the network position file
      -o OUTPUT, --output OUTPUT
                            Path of the output/prediction file
      -n NETWORK, --network NETWORK
                            Network name

For example on the "test" dataset, you would use the following command:

    python main.py -f fluorescence_test.txt -p networkPositions_test.txt -o score_test.csv -n test

To run the script, you will need a machine with at least 8GB RAM, a fast
processor (> 2.5 GHz), 4 cores and sufficient disk space. On our last
test, it took around 10 hours (with 7 process) on normal-1 and +-2 hours on small-1.

The performance obtained on normal-1 and small-1 are the following:

    On normal-1: 0.94356018640593564
    On small-1: 0.71027913026472989
     
Note that all parameters have been optimized for a big dataset, i.e. 1000 neurons, and it explains the poor result on small-1.
