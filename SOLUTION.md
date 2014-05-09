Solution
========

Summary
-------

We solve the challenge problem by infering an undirected network by estimating partial correlation [1] for every pair of neurons. In order to increase the performance, we preprocess the data by filtering the time series and weighting the samples to take into account the number (and the intensity) of burst peaks and deal with network bursts [2].
To achieve the best results, we asymmetrize our score matrix by trying to determine (heuristically) the causality.


Feature Selection / Extraction
------------------------------

Filering: We apply a (handmade) low-pass filter and a high-pass filter on the data.
Sample weighting : We look after the sum of values for the current time step and the previous time step and we weighted the current step value by a coefficient depending on the calculated sum and some parameters optimized (on normal-1 and normal-4 datasets).  


Modelling Techniques and Training
---------------------------------

Once the preprocessing is performed on the data, we estimate the partial correlation by computing the inverse of the correlation matrix, i.e. the precision matrix.

We improve the total score by considering the 800 most important components through a PCA analysis.

Code Description
----------------

See Readme.md

Dependencies
------------

The following programs and packages were used for the contest:

    - Python 2.7
    - NumPy >= 1.6.2
    - SciPy >= 0.11
    - scikit-learn >= 0.14

with appropriate blas and lapack binding such as MKL, accelerate or ATLAS.
In order to test the code, we recommend you to use the Anaconda python
distribution (https://store.continuum.io/cshop/anaconda/).

How To Generate the Solution
----------------------------

See Readme.md

Additional Comments and Observations
------------------------------------

We will provide more detailed information in a paper that we plan to submit to the workshop dedicated to the Connectomics challenge.

Simple Features and Methods
---------------------------

----

Figures
-------

----

References
----------

[1] De La Fuente, A., Bing, N., Hoeschele, I., & Mendes, P. (2004). Discovery of meaningful associations in genomic data using partial correlation coefficients. Bioinformatics, 20(18), 3565-3574.

[2] Stetter, O., Battaglia, D., Soriano, J., & Geisel, T. (2012). Model-free reconstruction of excitatory neuronal connectivity from calcium imaging signals. PLoS computational biology, 8(8), e1002653.
