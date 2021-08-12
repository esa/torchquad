---
title: 'torchquad: Numerical Integration in Arbitrary Dimensions with PyTorch'
tags:
  - Python
  - n-dimensional
  - numerical integration
  - GPU
  - automatic differentiation
  - PyTorch
  - high-performance computing
  - machine learning
authors:
  - name: Pablo Gómez^[corresponding author]
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
date: 15 June 2021
bibliography: paper.bib

---

# Summary

\texttt{torchquad} is a \texttt{Python} module for $n$-dimensional numerical integration optimized for graphics processing units (GPUs).
Various deterministic and stochastic integration methods, such as \texttt{Newton\textendash Cotes} formulas and \texttt{Monte} \texttt{Carlo} integration methods like \texttt{VEGAS} \texttt{Enhanced} [@VegasEnhanced-paper], are available for computationally efficient integration for arbitrary dimensionality $n_{\mathrm{d}}$.
As it is implemented using \texttt{PyTorch} [@PyTorch2019], one of the most popular machine learning frameworks, \texttt{torchquad} provides fully automatic differentiation throughout the integration, which is essential for many machine learning applications.

# Statement of Need

Multidimensional integration is needed in many fields, such as physics (ranging from particle physics [@ParticlePhysics-Paper] to astrophysics [@izzo2021geodesy]), applied finance [@AppliedFinance-Paper], medical statistics [@MedicalStatistics-Paper], and machine learning [@VEGASinMachineLearning-Paper]. 
Most of the conventional \texttt{Python} packages for multidimensional integration, such as \texttt{quadpy} [@quadpy] and \texttt{nquad} [@scipy], only target and are optimized for central processing units (CPUs). 
However, as many numerical integration methods are embarrassingly parallel, GPUs can offer superior computational performance in their computation. 
Furthermore, numerical integration methods typically suffer from the so-called \textit{curse of dimensionality} [@ZMCintegral]. 
This phenomenon refers to the fact that the computational complexity of the integration grows exponentially with the number of dimensions [@CurseOfDim-Book]. Reducing the error of the integration value requires increasing the number of function evaluation points $N$ exponentially, which significantly increases the runtime of the computation, especially for higher dimensions.
Previous work has demonstrated that this problem can be mitigated by leveraging the \textit{single instruction, multiple data} parallelization of GPUs [@ZMCintegral].

Although GPU-based implementations for multidimensional numerical integration in \mbox{\texttt{Python}} exist, some of these packages do not allow fully automatic differentiation [@borowka2019gpu], which is crucial for many machine learning applications [@Baydin2018autodiffinML]. Recently, to fill this gap, the packages \texttt{VegasFlow} [@VegasFlow-Paper] and \mbox{\texttt{ZMCintegral}} [@ZMCintegral] were developed. Both of these implementations are, however, based on \texttt{TensorFlow} [@Tensorflow], and there are currently no packages available that enable more than one-dimensional integration in \texttt{PyTorch}.
Additionally, the available GPU-based \texttt{Python} packages that allow fully automatic differentiation rely solely on \texttt{Monte} \texttt{Carlo} methods [@ZMCintegral; @VegasFlow-Paper]. 
Even though such methods offer good speed\textendash accuracy trade-offs for problems of high dimensionality $n_{\mathrm{d}}$, the efficiency of deterministic methods, such as the \texttt{Newton\textendash Cotes} formulas, is often superior for lower dimensionality [@Vegas-paper].

In summary, to the authors' knowledge, \texttt{torchquad} is the first \texttt{PyTorch}-based module for $n$-dimensional numerical integration. 
Furthermore, it incorporates several deterministic and stochastic methods, including \texttt{Newton\textendash Cotes} formulas and \texttt{VEGAS} \texttt{Enhanced}, which allow obtaining high-accuracy estimates for varying dimensionality at configurable computational cost as controlled by the maximum number of function evaluations $N$. It is, to the authors' knowledge, also the first GPU-capable implementation of \texttt{VEGAS} \texttt{Enhanced} [@VegasEnhanced-paper], which improves on its predecessor \texttt{VEGAS} by introducing an adaptive stratified sampling strategy.

Finally, being \texttt{PyTorch}-based, \texttt{torchquad} is fully differentiable, extending its applicability to use cases such as those in machine learning. In these applications, it is typically necessary to compute the gradient of some parameters with regard to input variables to perform updates of the trainable parameters in the machine learning model. With \texttt{torchquad}, e.g., the employed loss function can contain integrals without breaking the automatic differentiation required for training.


# Implemented Integration Methods

\texttt{torchquad} features fully vectorized implementations of various deterministic and stochastic methods to perform $n$-dimensional integration over cubical domains.
In particular, the following deterministic integration methods are available in \texttt{torchquad} (version 0.2.1):  

* \texttt{Trapezoid} \texttt{Rule} [@sag1964numerical] 
* \texttt{Simpson's} \texttt{Rule} [@sag1964numerical] 
* \texttt{Boole's} \texttt{Rule} [@ubale2012numerical] 

The stochastic integration methods implemented in \texttt{torchquad} so far are: 

* \texttt{Classic} \texttt{Monte} \texttt{Carlo} \texttt{Integrator} [@caflisch1998monte] 
* \texttt{VEGAS} \texttt{Enhanced} (\mbox{\texttt{VEGAS+}}) integration method [@VegasEnhanced-paper] 

The functionality and the convergence of all the methods are ensured through automatic unit testing, which relies on an extensible set of different test functions.
Both single and double precision are supported to allow different trade-offs between accuracy and memory utilization. Even though it is optimized for GPUs, \texttt{torchquad} can also be employed without a GPU without any functional limitations.

# Installation \& Contribution

The \texttt{torchquad} package is implemented in \texttt{Python} \texttt{3.8} and is openly available under a GPL-3 license. Installation with either \texttt{pip} (\texttt{PyPi})\footnote{\texttt{torchquad} package on \texttt{PyPi}, \url{https://pypi.org/project/torchquad/}} or \texttt{conda}\footnote{\texttt{torchquad} package on \texttt{conda}, \url{https://anaconda.org/conda-forge/torchquad}} is available. Our public \texttt{GitHub} repository\footnote{\texttt{torchquad} \texttt{GitHub} repository, \url{https://github.com/esa/torchquad}} provides users with direct access to the main development branch. Users wishing to contribute to \texttt{torchquad} can submit issues or pull requests to our \texttt{GitHub} repository following the contribution guidelines outlined there.

# Tutorials 

The \texttt{torchquad} documentation, hosted on \texttt{Read} \texttt{the} \texttt{Docs},\footnote{\texttt{torchquad} documentation on \texttt{Read} \texttt{the} \texttt{Docs}, \url{https://torchquad.readthedocs.io/}} provides some examples of the use of \texttt{torchquad} for one-dimensional and multidimensional integration utilizing a variety of the implemented methods.

# References
