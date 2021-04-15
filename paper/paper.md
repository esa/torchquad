---
title: 'torchquad: a Python package for multidimensional numerical integration on the GPU'
tags:
  - Python
  - multidimensional
  - numerical integration
  - deterministic
  - stochastic
  - gpu
authors:
  - name: Gabriele Meoni
    orcid: 0000-0001-9311-6392
    affiliation: 1
  - name: Håvard Hem Toftevaag
    orcid: 0000-0003-4692-5722
    affiliation: 1
  - name: Pablo Gómez
    orcid: 0000-0002-5631-8240
    affiliation: 1
affiliations:
 - name: Advanced Concepts Team, European Space Agency
   index: 1
date: 15 April 2021
bibliography: paper.bib

---

# Summary

\texttt{torchquad} is a Python module for $n$-dimensional numerical integration on the graphics processing unit (GPU).
Various deterministic and stochastic integration methods are available, such as Simpson's rule and Monte Carlo integration, allowing of high-efficiency integration for any dimensionality, $n$.
Because it is based on PyTorch [@PyTorch2019], \texttt{torchquad} provides full automatic differentiation throughout the integration, which is essential for machine learning purposes.

# Statement of Need

Multidimensional integration is needed in many fields, such as physics (everything from particle physics [@ParticlePhysics-Paper] to astrophysics [@Astrophysics-Paper]), applied finance [@AppliedFinance-Paper], medical statistics [@MedicalStatistics-Paper], and machine learning [@VEGASinMachineLearning-Paper]. 
Many of the conventional Python packages for multidimensional integration, such as \texttt{quadpy} [@quadpy] and \texttt{nquad} [@scipy], rely on the utilization of the central processing unit (CPU). However, CPUs have demonstrated limited performances compared with GPUs for the implementation of integration methods, which suffer from the so-called \textit{curse of dimensionality} [@ZMCintegral]. 
The complexity of the integration grows, indeed, exponentially with the number of dimensions [@CurseOfDim-Book]. Because of that, CPU-based implementations feature bad trade-offs between accuracy and speed in the calculation. Limiting the error in the integration value requires increasing the number of function evaluation points $N$, which increases the latency of the computation, especially for higher dimensions.
Previous work has demonstrated that this problem can be mitigated by leveraging on \textit{single instruction, multiple data} parallelization of GPUs [@ZMCintegral].

Although some GPU-based implementations exist, such as \texttt{VegasFlow} [@VegasFlow-Paper; @VegasFlow-Package] and \texttt{ZMCintegral} [@ZMCintegral; @ZMCintegral-code], their implementation is based on TensorFlow [@Tensorflow], which is just one of the existing deep-learning softwares in Python. 
Additionally, the available GPU-based Python packages rely solely on Monte Carlo [@ZMCintegral] and \mbox{\texttt{VEGAS}} [@VegasFlow-Paper] methods. Even though such methods offer good speed/accuracy trade-offs for high values of $n$, the efficiency of other methods are better for small values of $n$ [@Vegas-paper].

In contrast, \texttt{torchquad} is, to the best of the authors' knowledge, the first implementation based on PyTorch. 
Furthermore, it incorporates several deterministic and stochastic methods, including \mbox{\texttt{VEGAS}} and Monte Carlo, which allow obtaining high accuracy for various dimensions and values of $N$. 
In addition, the value of $n$ used is programmable for each method.

Finally, being PyTorch-based, \texttt{torchquad} is fully differentiable, extending its usability to applications such as those based on machine learning.


# Implemented integration methods

The following deterministic integration methods are available in \texttt{torchquad} (version 0.2):  

* Trapezoid Rule [@sag1964numerical],  
* Simpson's Rule [@sag1964numerical], and  
* Boole's Rule [@ubale2012numerical],  

while the stochastic integration methods implemented in \texttt{torchquad} so far are: 

* Monte Carlo Integrator [@caflisch1998monte], and  
* VEGAS Enhanced (\mbox{\texttt{VEGAS+}}) method [@VegasEnhanced-paper].  

As previously specified, the number of domain evaluation points can be tuned for each integration method to ensure flexible trade-offs between computational complexity and accuracy. 

# Installation \& Contribution

The \texttt{torchquad} package is implemented in Python 3.8 and is openly accessible under a GPL-3 license. Users wishing to contribute to \texttt{torchquad} can submit issues or pull requests to our GitHub repository following the contribution guidelines outlined there.

# Tutorials 

The \texttt{torchquad} Github repository includes the ```Example_Notebook.ipynb```, which provides some examples of the use of \texttt{torchquad} for one-dimensional and multidimensional integration. Moreover, a benchmark with \texttt{nquad} [@scipy] is shown in terms of runtime and absolute error compared to a ground truth value.

# References
