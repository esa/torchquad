.. _installation:

Getting started
===============

This is a brief introduction on how to set up *torchquad*.

Prerequisites
--------------

*torchquad* is built with

- `autoray <https://github.com/jcmgray/autoray>`_, which means the implemented quadrature supports `NumPy <https://numpy.org/>`_ and can be used for machine learning with modules such as `PyTorch <https://pytorch.org/>`_, `JAX <https://github.com/google/jax/>`_ and `Tensorflow <https://www.tensorflow.org/>`_, where it is fully differentiable

torchquad has no backend pinned as a hard dependency — install the numerical
backend(s) you want alongside it, or use NumPy alone. Any of `pip
<https://pip.pypa.io/>`_, `uv <https://docs.astral.sh/uv/>`_, or `conda
<https://docs.conda.io/en/latest/>`_ works.
Note that *torchquad* also works on the CPU; however, it is optimized for GPU usage.
GPU support is tested only on NVIDIA cards with CUDA. For GPU installs, follow
each framework's own install guide — the CPU-only convenience extras below
cannot select GPU wheels, and JAX/TensorFlow GPU builds are Linux/WSL2-only.

For a detailed list of required packages and packages for numerical backends,
please refer to the conda environment files `environment.yml <https://github.com/esa/torchquad/blob/main/environment.yml>`_ and
`environment_all_backends.yml <https://github.com/esa/torchquad/blob/main/environment_all_backends.yml>`_.
torchquad requires Python 3.10 or newer. Its CI suite runs on Python 3.12 with JAX 0.4.35, NumPy 2.3, PyTorch 2.5 and TensorFlow 2.18; other versions of the backends should work as well.


Installation
-------------

Install *torchquad* from PyPI:

   .. code-block:: bash

      pip install torchquad
      # or, with uv:
      uv pip install torchquad

It is also available on conda-forge:

   .. code-block:: bash

      conda install torchquad -c conda-forge

**Adding a backend.** *torchquad* ships convenience extras that pull a backend
from the default package index. ``jax`` and ``tensorflow`` install CPU builds;
``torch`` resolves to the CUDA build on Linux (that is what PyTorch publishes to
PyPI), and CPU on macOS/Windows. For a guaranteed-CPU torch, install it from the
PyTorch CPU index first.

   .. code-block:: bash

      pip install "torchquad[torch]"        # PyTorch (CUDA on Linux, CPU elsewhere)
      pip install "torchquad[jax]"          # JAX (CPU)
      pip install "torchquad[tensorflow]"   # TensorFlow (CPU)
      pip install "torchquad[all]"          # all three

**Adding a backend (GPU).** These extras cannot select GPU wheels (a Python
package cannot encode the CUDA-specific index URLs each framework needs), so for
GPU support install the backend from its own guide first, then
``pip install torchquad``:

- PyTorch: https://pytorch.org/get-started/locally/
- JAX (Linux/WSL2 only): https://docs.jax.dev/en/latest/installation.html
- TensorFlow (Linux/WSL2 only): https://www.tensorflow.org/install/gpu

For a full multi-backend setup, the conda file
`environment_all_backends.yml <https://github.com/esa/torchquad/blob/main/environment_all_backends.yml>`__
installs every backend (CPU) in one step:

   .. code-block:: bash

      conda env create -f environment_all_backends.yml
      conda activate torchquad


Usage
-----

Now you are ready to use *torchquad*.
A brief example of how *torchquad* can be used to compute a simple integral can be found on our `GitHub <https://github.com/esa/torchquad#usage>`_.
For a more thorough introduction, please refer to the `tutorial <https://torchquad.readthedocs.io/en/main/tutorial.html>`_.
