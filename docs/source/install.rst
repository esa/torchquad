.. _installation:

Getting started
===============

This is a brief introduction on how to set up *torchquad*.

Prerequisites
--------------

*torchquad* is built with

- `autoray <https://github.com/jcmgray/autoray>`_, which means the implemented quadrature supports `NumPy <https://numpy.org/>`_ and can be used for machine learning with modules such as `PyTorch <https://pytorch.org/>`_, `JAX <https://github.com/google/jax/>`_ and `Tensorflow <https://www.tensorflow.org/>`_, where it is fully differentiable
- `conda <https://docs.conda.io/en/latest/>`_, which will take care of all requirements for you

We recommend using `conda <https://docs.conda.io/en/latest/>`_, especially if you want to utilize the GPU.
With PyTorch it will automatically set up CUDA and the cudatoolkit for you, for example.
Note that *torchquad* also works on the CPU; however, it is optimized for GPU usage.
torchquad's GPU support is tested only on NVIDIA cards with CUDA. We are investigating future support for AMD cards through `ROCm <https://pytorch.org/blog/pytorch-for-amd-rocm-platform-now-available-as-python-package/>`_.

For a detailed list of required packages and packages for numerical backends,
please refer to the conda environment files `environment.yml <https://github.com/esa/torchquad/blob/main/environment.yml>`_ and
`environment_all_backends.yml <https://github.com/esa/torchquad/blob/main/environment_all_backends.yml>`_.
torchquad has been tested with JAX 0.2.25, NumPy 1.19.5, PyTorch 1.10.0 and Tensorflow 2.7.0; other versions of the backends should work as well.


Installation
-------------

First, we must make sure we have `torchquad <https://github.com/esa/torchquad>`_ installed.
The easiest way to do this is simply to

   .. code-block:: bash

      conda install torchquad -c conda-forge

Alternatively, it is also possible to use

   .. code-block:: bash

      pip install torchquad

The PyTorch backend with CUDA support can be installed with

   .. code-block:: bash

      conda install "cudatoolkit>=11.1" "pytorch>=1.9=*cuda*" -c conda-forge -c pytorch

Note that since PyTorch is not yet on *conda-forge* for Windows, we have
explicitly included it here using ``-c pytorch``.
Note also that installing PyTorch with *pip* may **not** set it up with CUDA
support.
Therefore, we recommend to use *conda*.

Here are installation instructions for other numerical backends:

   .. code-block:: bash

      conda install "tensorflow>=2.6.0=cuda*" -c conda-forge
      pip install "jax[cuda]>=0.2.22" --find-links https://storage.googleapis.com/jax-releases/jax_releases.html # linux only
      conda install "numpy>=1.19.5" -c conda-forge

More installation instructions for numerical backends can be found in
`environment_all_backends.yml <https://github.com/esa/torchquad/blob/main/environment_all_backends.yml>`__
and at the backend documentations, for example
https://pytorch.org/get-started/locally/,
https://github.com/google/jax/#installation and
https://www.tensorflow.org/install/gpu, and often there are multiple
ways to install them.


Usage
-----

Now you are ready to use *torchquad*.
A brief example of how *torchquad* can be used to compute a simple integral can be found on our `GitHub <https://github.com/esa/torchquad#usage>`_.
For a more thorough introduction, please refer to the `tutorial <https://torchquad.readthedocs.io/en/main/tutorial.html>`_.
