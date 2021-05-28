---
title: 'torchquad: a Python package for *n*-dimensional numerical integration optimized on the GPU'
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

\texttt{torchquad} is a Python module for $n$-dimensional numerical integration optimized for graphics processing units (GPUs).
Various deterministic and stochastic integration methods, such as Simpson's rule and Monte Carlo integration, are available for computationally efficient integration for any dimensionality, $n_{\mathrm{d}}$.
Because it is based on PyTorch [@PyTorch2019], \texttt{torchquad} provides fully automatic differentiation throughout the integration, which is essential in many machine learning applications.

# Statement of Need

Multidimensional integration is needed in many fields, such as physics (ranging from particle physics [@ParticlePhysics-Paper] to astrophysics [@Astrophysics-Paper]), applied finance [@AppliedFinance-Paper], medical statistics [@MedicalStatistics-Paper], and machine learning [@VEGASinMachineLearning-Paper]. 
Many of the conventional Python packages for multidimensional integration, such as \texttt{quadpy} [@quadpy] and \texttt{nquad} [@scipy], only target and are optimized for central processing units (CPUs). 
However, as many numerical integration methods are embarassingly parallel, GPUs can offer a superior computational performance for their implementation. 
Note also that, especially for higher dimensionality, numerical integration methods typically suffer from the so-called \textit{curse of dimensionality} [@ZMCintegral]. 
This phenomenon refers to the fact that the complexity of the integration grows exponentially with the number of dimensions [@CurseOfDim-Book]. Limiting the error in the integration value requires increasing the number of function evaluation points $N$, which increases the runtime of the computation, especially for higher dimensions.
Previous work has demonstrated that this problem can be mitigated by leveraging on \textit{single instruction, multiple data} parallelization of GPUs [@ZMCintegral].

Although some GPU-based implementations exist, such as \texttt{VegasFlow} [@VegasFlow-Paper; @VegasFlow-Package] and \texttt{ZMCintegral} [@ZMCintegral; @ZMCintegral-code], these implementations are based on TensorFlow [@Tensorflow]. Currently, there are no packages available that enable more than one-dimensional integration in PyTorch.
Additionally, the available GPU-based Python packages rely solely on Monte Carlo [@ZMCintegral] and \mbox{\texttt{VEGAS}} [@VegasFlow-Paper] methods. 
Even though such methods offer good speed/accuracy trade-offs for high dimensionality $n_{\mathrm{d}}$, 
the efficiency of other methods are superior for lower dimensionality $n_{\mathrm{d}}$ [@Vegas-paper].

In contrast, \texttt{torchquad} is, to the authors' knowledge, the first PyTorch-based module for $n$-dimensional numerical integration. 
It incorporates several deterministic and stochastic methods, including \mbox{\texttt{VEGAS}} and classic Monte Carlo integration, which allow obtaining high accuracy estimates for varying dimensionality at configurable computational cost as controlled by the maximum number of function evaluations $N$. It is, to the authors' knowledge also the first GPU-capable implementation of VEGAS Enhanced [@VegasEnhanced-paper], which improves its predecessor VEGAS by introducing an adaptive stratified sampling strategy, called \textit{importance sampling}.

Finally, being PyTorch-based, \texttt{torchquad} is fully differentiable, extending its usability to applications such as those based on machine learning. In these applications it is typically necessary to compute the gradient of some parameter in regard to some other input parameter to perform updates of the trainable parameters in the machine learning model. With \texttt{torchquad}, e.g., the utilized loss function can contain integrals without breaking the automatic differentation that is required for training.


# Implemented integration methods

\texttt{Torchquad} features a fully vectorized implementation of various deterministic and stochastic methods to perform $n$-dimensional integration over cubical domains.
In particular, the following deterministic integration methods are available in \texttt{torchquad} (version 0.2):  

* Trapezoid Rule [@sag1964numerical],  
* Simpson's Rule [@sag1964numerical], and  
* Boole's Rule [@ubale2012numerical],  

while the stochastic integration methods implemented in \texttt{torchquad} so far are: 

* Classic Monte Carlo Integrator [@caflisch1998monte], and  
* VEGAS Enhanced (\mbox{\texttt{VEGAS+}}) method [@VegasEnhanced-paper].  

The functionality and the convergence of all the methods are proven through automatic unit testing, which relies on an extensible set of different test functions.
Both single and double precision are supported to ensure different trade-offs between accuracy and memory utilisation. 

# Installation \& Contribution

The \texttt{torchquad} package is implemented in Python 3.8 and is openly accessible under a GPL-3 license. Users wishing to contribute to \texttt{torchquad} can submit issues or pull requests to our GitHub repository following the contribution guidelines outlined there.

# Tutorials 

The \texttt{torchquad} GitHub repository includes the Jupyter notebook ```Example_Notebook.ipynb```, which provides some examples of the use of \texttt{torchquad} for one-dimensional and multidimensional integration. Moreover, a benchmark Jupyter notebook with \texttt{nquad} [@scipy] is shown in terms of runtime and absolute error compared to a ground truth value. Additionally, we refer to our document at TODO for more details.

# References