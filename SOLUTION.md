Solution
========

Method
------

The core principle of our approach is to recover an undirected network by
estimating partial correlations [1] for every pair of neurons. In particular,
this approach is well-known for identifying first-order interactions (i.e.,
direct connections in the network) from higher-order interactions (i.e.,
indirect connections).

In order to increase the performance, the raw data is preprocessed by i) filtering the
time series and ii) re-weighting the samples to take into account the number
(and the intensity) of burst peaks [2].

As a last step, and to obtain slightly better results, we asymmetrize our
score matrix by trying to determine (heuristically) the causality.


Feature Selection / Extraction
------------------------------

The data were filtered using a low-pass filter, a high-pass filter and a hard tresholding filter (see `model/PCA.py:_preprocess`). 

For the partial correlation method, we apply one more non-linear filter based on the overall neuron activity
(`model/PCA.py:_weights_fast`): for each sample, i.e. each time 
interval, we weight the samples depending on the global neuron activity at current and 
previous time steps, thereby lowering the effect of the end of high global burst periods in the correlation calculation. 

Parameters of those filters have been optimized on normal-1 and normal-4 datasets.

Modeling Techniques and Training
---------------------------------

`model/PCA.py`: Once preprocessing is done, partial correlations are estimated 
by computing the inverse of the correlation matrix (also known as the precision 
matrix).  To filter out noise, the inverse of the correlation matrix is recovered
from Principal Component Analysis (PCA) using the 800 first components (out
of 1000). 

`model/directivity.py`: Some causal information (directivity of the links) were retrieved from the data by comparing activity of each couple of neurons between two subsequent time steps. The directivity method tries to detect variation of fluorescence signal of a neuron `j` due to a neuron `i`. Let us denote the fluorescence signal of a neuron `l` at time `t` by `x_l[t]`, this method counts the number of time that  `x_j[t+1] - x_i[t]` is in `[f_1, f_2]` where `f_1` and `f_2` are parameters of the method. 

Both solutions are combined together through averaging. 



Code description
----------------

See `README.md`.


Additional Comments and Observations
------------------------------------

We will provide motivation and details in a paper that we plan to submit 
to the Neural Connectomics Workshop organized at ECML'14.


References
----------

[1] De La Fuente, A., Bing, N., Hoeschele, I., & Mendes, P. (2004). 
    Discovery of meaningful associations in genomic data using partial 
    correlation coefficients. Bioinformatics, 20(18), 3565-3574.

[2] Stetter, O., Battaglia, D., Soriano, J., & Geisel, T. (2012). 
    Model-free reconstruction of excitatory neuronal connectivity from 
    calcium imaging signals. PLoS computational biology, 8(8), e1002653.
