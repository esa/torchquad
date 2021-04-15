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
date: 1 July 2021
bibliography: paper.bib

---

# Summary

\texttt{torchquad} is a Python module for $n$-dimensional numerical integration on the graphics processing unit (GPU).


# Statement of Need

The world needs torchquad because ...

PyTorch [@NEURIPS2019].
such as \texttt{quadpy} [@quadpy] and \texttt{nquad} [@scipy]
\textit{curse of dimensionality} [@ZMCintegral].
such as \texttt{VegasFlow} [@Carrazza2020rdn; @vegasflow-package] and \texttt{ZMCintegral} [@ZMCintegral; @ZMCintegral-code], their implementation is based on TensorFlow [@Tensorflow].

The following deterministic integration methods are available in \texttt{torchquad} (version 0.2):  
  - Trapezoid Rule [@sag1964numerical]  
  - Simpson's Rule [@sag1964numerical]  
  - Boole's Rule [@ubale2012numerical]  

The stochastic integration methods implemented in \texttt{torchquad} are:  
  - Monte Carlo Integrator [@caflisch1998monte]
  - VEGAS Enhanced (\texttt{VEGAS+}) method [@lepage2020adaptive]


\begin{itemize}
    \item Monte Carlo Integrator [@caflisch1998monte]
    \item VEGAS Enhanced (VEGAS+) method [@lepage2020adaptive]
\end{itemize}

# Acknowledgements

We would like to thank ...

# References