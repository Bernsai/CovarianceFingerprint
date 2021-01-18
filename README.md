# Covariance Fingerprints

_Trends, periodicities and local variations are among the main recognizable patterns in time series data. While humans 
are able to quickly explore the superimposition of such patterns, mining algorithms are often faced with the challenge 
of finding (i) a suitable time series representation model and (ii) an expressive query model which adapt to diverse 
application domains and information needs. [...] We propose a supervised stochastic approach which facilitates 
interval-based pattern analysis of time series data. Our proposal is based on non-parametric Gaussian Processes and is 
able to interrelate interesting patterns within single and across multiple time series. Our performance evaluation in 
different real-world application domains indicates that our approach is able to expose interesting patterns and 
knowledge._ 

-- abstract of [1], which proposes the concept of covariance fingerprints.

This repository provides the code used for evaluating the respective concept of _Covariance Fingerprints_ throughout [1]. 
In order to avoid any copyright issues, we omitted the used datasets for this repository. Still, they may be retrieved 
using the cited sources of the corresponding paper [1].

## Implementation
We implemented the given Algorithms and the general problem of automatic GPM retrieval using Python 3.8, Tensorflow 2.4 
and Tensorflow-Probability 0.11 (and other auxiliary libraries). We used our package 
[gpbasics](https://github.com/Bernsai/GaussianProcessFundamentals) for basic Gaussian process model functionalities 
and our package [gpmretrieval](https://github.com/Bernsai/GaussianProcessModelRetrieval) for Gaussian process model 
inference [2].

## Example
In `test/Example.py` we provide an exemplary application of our proposed concept of _Covariance Fingerprints_ to a 
financial dataset covering the _DAX Performance Index_ (which needs to be separately retrieved via 
[Yahoo Finance](https://finance.yahoo.com/quote/%5EN225/history)). The resulting matching regions may be further analyzed 
and interwoven using our interval-based implementation of Allen relations [3] (cf. `main/AllenRelations.py`). 

## References

[1] F. Berns and C. Beecks, Stochastic Time Series Representation for Interval Pattern Mining via Gaussian Processes, 
in SDM, SIAM, 2021.

[2] F. Berns and C. Beecks, Complexity-adaptive gaussian process model inference for large-scale data, in SDM, SIAM, 2021.

[3] J. F. Allen, Maintaining knowledge about temporal intervals, Commun. ACM, 26 (1983), pp. 832–843.
