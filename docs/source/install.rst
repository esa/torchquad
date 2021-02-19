.. _installation:

Getting started
===============

This is a brief example of setting up *torchquad*.

Prerequisites 
--------------

*torchquad* is built with

- `PyTorch <https://pytorch.org/>`_
- `conda <https://docs.conda.io/en/latest/>`_

We recommend using `conda <https://docs.conda.io/en/latest/>`_, especially if you want to utilize the GPU. 
It will automatically set up CUDA and the cudatoolkit for you in that case.
Note that torchquad also works on the CPU. However, it is optimized for GPU usage.

- `conda <https://docs.conda.io/en/latest/>`_, which will take care of all requirements for you. For a detailed list of required packages, please refer to the `conda environment file <https://github.com/esa/torchquad/blob/main/environment.yml>`_.

Installation
-------------

First we must make sure we have `torchquad <https://github.com/esa/torchquad>`_ installed:

1. Get `miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ or similar
2. Clone the repo

   .. code-block:: bash

      git clone https://github.com/esa/torchquad.git

3. Setup the environment. This will create a conda environment called ``torchquad``

   .. code-block:: bash

      conda env create -f environment.yml

Alternatively you can use

.. code-block:: bash

      pip install torchquad

Usage
-----

A brief example of how *torchquad* can be used to compute a simple integral can be found on our `GitHub <https://github.com/esa/torchquad>`_. 
For a more thorough introduction please refer to the `example notebook <https://github.com/esa/torchquad/blob/main/notebooks/Example_notebook.ipynb>`_.
