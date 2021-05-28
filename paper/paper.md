---
title: 'torchquad: High-performance numerical Integration on GPUs with PyTorch'
tags:
  - Python
  - multidimensional
  - numerical integration
  - GPU
  - automatic differentiation
  - PyTorch
  - high-performance computing
authors:
  - name: Pablo Gómez
    orcid: 0000-0002-5631-8240
    affiliation: 1
  - name: Håvard Hem Toftevaag
    orcid: 0000-0003-4692-5722
    affiliation: 1
  - name: Gabriele Meoni
    orcid: 0000-0001-9311-6392
    affiliation: 1
affiliations:
 - name: Advanced Concepts Team, European Space Agency, Noordwijk, The Netherlands
   index: 1
date: 28 May 2021
bibliography: paper.bib

---

# Summary

\texttt{torchquad} is a Python module for $n$-dimensional numerical integration optimized for graphics processing units (GPUs).
Various deterministic and stochastic integration methods, such as Newton\textendash Cotes formulas and Monte Carlo integration methods like VEGAS Enhanced [@VegasEnhanced-paper], are available for computationally efficient integration for any dimensionality $n_{\mathrm{d}}$.
As it is implemented using PyTorch [@PyTorch2019], one of the most popular machine learning frameworks, \texttt{torchquad} provides fully automatic differentiation throughout the integration, which is essential for many machine learning applications.

# Statement of Need

Multidimensional integration is needed in many fields, such as physics (ranging from particle physics [@ParticlePhysics-Paper] to astrophysics [@Astrophysics-Paper]), applied finance [@AppliedFinance-Paper], medical statistics [@MedicalStatistics-Paper], and machine learning [@VEGASinMachineLearning-Paper]. 
Most of the conventional Python packages for multidimensional integration, such as \texttt{quadpy} [@quadpy], \texttt{nquad} [@scipy] and Cuba [@hahn2005cuba], only target and are optimized for central processing units (CPUs). 
However, as many numerical integration methods are embarrassingly parallel, GPUs can offer superior computational performance in computing them. 
Furthermore, especially for higher dimensionality, numerical integration methods typically suffer from the so-called \textit{curse of dimensionality} [@ZMCintegral]. 
This phenomenon refers to the fact that the computational complexity of the integration grows exponentially with the number of dimensions [@CurseOfDim-Book]. Reducing the error of the integration value requires increasing the number of function evaluation points $N$, which significantly increases the runtime of the computation, especially for higher dimensions.
Previous work has demonstrated that this problem can be mitigated by leveraging the \textit{single instruction, multiple data} parallelization of GPUs [@ZMCintegral].

Although GPU-based implementations for multidimensional numerical integration in Python exist, some of these packages do not allow fully automatic differentiation [@borowka2019gpu], which is crucial for many machine learning applications [@Baydin2018autodiffinML]. Recently, to fill this gap, the packages \texttt{VegasFlow} [@VegasFlow-Paper; @VegasFlow-Package] and \texttt{ZMCintegral} [@ZMCintegral; @ZMCintegral-code] were developed. Both of these implementations are, however, based on the TensorFlow library [@Tensorflow], and there are currently no packages available that enable more than one-dimensional integration in PyTorch.
Additionally, the available GPU-based Python packages that allow fully automatic differentiation rely solely on Monte Carlo [@ZMCintegral] or VEGAS [@VegasFlow-Paper] methods. 
Even though such methods offer good speed to accuracy trade-offs for problems of higher dimensionality $n_{\mathrm{d}}$, the efficiency of deterministic methods, such as the Newton\textendash Cotes formulas, is often superior for lower dimensionality [@Vegas-paper].

Thus, to the authors' knowledge, \texttt{torchquad} is the first PyTorch-based module for $n$-dimensional numerical integration. 
Furthermore, it incorporates several deterministic and stochastic methods, including Newton\textendash Cotes formulas and VEGAS Enhanced, which allow obtaining high-accuracy estimates for varying dimensionality at configurable computational cost as controlled by the maximum number of function evaluations $N$. It is, to the authors' knowledge, also the first GPU-capable implementation of VEGAS Enhanced [@VegasEnhanced-paper], which improves on its predecessor VEGAS by introducing an adaptive stratified sampling strategy.

Finally, being PyTorch-based, \texttt{torchquad} is fully differentiable, extending its applicability to use-cases such as those based on machine learning. In these applications, it is typically necessary to compute the gradient of some parameters with regard to input variables to perform updates of the trainable parameters in the machine learning model. With \texttt{torchquad}, e.g., the utilized loss function can contain integrals without breaking the automatic differentiation required for training.


# Implemented Integration Methods

\texttt{torchquad} features fully vectorized implementations of various deterministic and stochastic methods to perform $n$-dimensional integration over cubical domains.
In particular, the following deterministic integration methods are available in \texttt{torchquad} (version 0.2):  

* Trapezoid Rule [@sag1964numerical] 
* Simpson's Rule [@sag1964numerical]  
* Boole's Rule [@ubale2012numerical]  

The stochastic integration methods implemented in \texttt{torchquad} so far are: 

* Classic Monte Carlo Integrator [@caflisch1998monte]  
* VEGAS Enhanced (\mbox{\texttt{VEGAS+}}) integration method [@VegasEnhanced-paper]  

The functionality and the convergence of all the methods are ensured through automatic unit testing, which relies on an extensible set of different test functions.
Both single and double precision are supported to allow different trade-offs between accuracy and memory utilization. Even though it is optimized for GPUs, \texttt{torchquad} can also be employed without a GPU without any functional limitations.

# Installation \& Contribution

The \texttt{torchquad} package is implemented in Python 3.8 and is openly available under a GPL-3 license. Installation with either pip (PyPi) or conda is available for all releases. Our public GitHub repository provides users with direct access to the main development branch. Users wishing to contribute to \texttt{torchquad} can submit issues or pull requests to our GitHub repository following the contribution guidelines outlined there.

# Tutorials 

The \texttt{torchquad} documentation, hosted on Read the Docs, provides some examples of the use of \texttt{torchquad} for one-dimensional and multidimensional integration utilizing a variety of the implemented methods.

# References
