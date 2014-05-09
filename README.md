Kaggle Connectomics
===================

Contact
-------

Antonio Sutera <a.sutera@ulg.ac.be>
Arnaud Joly <a.joly@ulg.ac.be>


The hardware / OS platform you used
-----------------------------------

Code ran on MacOsx 10.9.2 and Linux (version 2.6.18-194.26.1.el5
(brewbuilder@norob.fnal.gov) (gcc version 4.1.2 20080704 (Red Hat 4.1.2-48))
1 SMP Tue Nov 9 12:46:16 EST 2010).


Any necessary 3rd-party software (+ installation steps)
-------------------------------------------------------

The following programs and packages were used for the contest:

    - Python 2.7
    - NumPy >= 1.6.2
    - SciPy >= 0.11
    - scikit-learn >= 0.14

with appropriate blas and lapack binding such as MKL, accelerate or ATLAS.
In order to test the code, we recommend you to use the Anaconda python
distribution (https://store.continuum.io/cshop/anaconda/).


How to train your model
-----------------------

No model is learnt to produce the connectivity score matrix.


How to make predictions on a new test set.
------------------------------------------
In order to reproduce the result, you can launch the main.py file.
The usage is the following:

    usage: main.py [-h] -f FLUORESCENCE [-p POSITION] -o OUTPUT -n NETWORK

    Perform Connectomics modelling

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

To run the script, you will need a machine with at least 8 Giga ram, a fast
processor (> 2.5 GHz), 4 cores and sufficient disk space. On our last
test, it took around XXX hours (with 7 process) on YYYY and XXX hours on ZZZ.

The performance we obtained on normal-1 are the following XXXX.
