# torchquad
<!--
*** Based on https://github.com/othneildrew/Best-README-Template
-->

![Read the Docs (version)](https://img.shields.io/readthedocs/torchquad/main?style=flat-square) ![GitHub Workflow Status (branch)](https://img.shields.io/github/workflow/status/esa/torchquad/Running%20tests/main?style=flat-square) ![GitHub last commit](https://img.shields.io/github/last-commit/esa/torchquad?style=flat-square) 
![GitHub](https://img.shields.io/github/license/esa/torchquad?style=flat-square) ![Conda (channel only)](https://img.shields.io/conda/vn/conda-forge/torchquad?style=flat-square) ![PyPI](https://img.shields.io/pypi/v/torchquad?style=flat-square) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/torchquad?style=flat-square) 

![GitHub contributors](https://img.shields.io/github/contributors/esa/torchquad?style=flat-square)
![GitHub issues](https://img.shields.io/github/issues/esa/torchquad?style=flat-square) ![GitHub pull requests](https://img.shields.io/github/issues-pr/esa/torchquad?style=flat-square) 
![Conda](https://img.shields.io/conda/dn/conda-forge/torchquad?style=flat-square) ![PyPI - Downloads](https://img.shields.io/pypi/dm/torchquad?style=flat-square) 
[![JOSS](https://joss.theoj.org/papers/d6f22f83f1a889ddf83b3c2e0cd0919c/status.svg)](https://joss.theoj.org/papers/d6f22f83f1a889ddf83b3c2e0cd0919c?style=flat-square)

<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/esa/torchquad">
    <img src="logos/torchquad_white_background_PNG.png" alt="Logo" width="280" height="120">
  </a>
  <p align="center">
    High-performance numerical integration on the GPU with PyTorch
    <br />
    <a href="https://torchquad.readthedocs.io"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/esa/torchquad/blob/master/notebooks/Torchquad%20-%20Example%20notebook.ipynb">View Example notebook</a>
    ·
    <a href="https://github.com/esa/torchquad/issues">Report Bug</a>
    ·
    <a href="https://github.com/esa/torchquad/issues">Request Feature</a>
  </p>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li><a href="#goals">Goals</a></li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#test">Test</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#FAQ">FAQ</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

The torchquad module allows utilizing GPUs for efficient numerical integration with [PyTorch](https://pytorch.org/). 
The software is free to use and is designed for the machine learning community and research groups focusing on topics requiring high-dimensional integration.

### Built With

This project is built with the following packages:

* [PyTorch](https://pytorch.org/), which means it is fully differentiable and can be used for machine learning, and
* [conda](https://docs.conda.io/en/latest/), which will take care of all requirements for you.


<!-- GOALS -->
## Goals

* **Supporting science**:  Multidimensional numerical integration is needed in many fields, such as physics (from particle physics to astrophysics), in applied finance, in medical statistics, and others. torchquad aims to assist research groups in such fields, as well as the general machine learning community.
* **Withstanding the curse of dimensionality**: The [curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality) makes deterministic methods in particular, but also stochastic ones, computationally expensive when the dimensionality increases. However, many integration methods are [embarrassingly parallel](https://en.wikipedia.org/wiki/Embarrassingly_parallel), which means they can strongly benefit from GPU parallelization. The curse of dimensionality still applies but the improved scaling alleviates the computational impact.
* **Enabling full differentiability**: In line with recent trends (e.g. [JAX](https://github.com/google/jax) torchquad builds on [PyTorch](https://pytorch.org/) in such a way that it is [fully differentiable](https://en.wikipedia.org/wiki/Differentiable_programming). This enables a broad range of optimization and machine learning applications.


<!-- GETTING STARTED -->
## Getting Started

This is a brief guide for how to set up torchquad.

### Prerequisites

We recommend using [conda](https://anaconda.org/conda-forge/torchquad), especially if you want to utilize the GPU. It will automatically set up CUDA and the cudatoolkit for you in that case.
Note that torchquad also works on the CPU; however, it is optimized for GPU usage. Currently torchquad only supports NVIDIA cards with CUDA. We are investigating future support for AMD cards through [ROCm](https://pytorch.org/blog/pytorch-for-amd-rocm-platform-now-available-as-python-package/).

For a detailed list of required packages, please refer to the [conda environment file](https://github.com/esa/torchquad/blob/main/environment.yml).

### Installation

The easiest way to install torchquad is simply to 

   ```sh
   conda install torchquad -c conda-forge -c pytorch
   ```

Note that since PyTorch is not yet on *conda-forge* for Windows, we have explicitly included it here using `-c pytorch`.  

Alternatively, it is also possible to use
   ```sh
   pip install torchquad
   ```

NB Note that *pip* will **not** set up PyTorch with CUDA and GPU support. Therefore, we recommend to use *conda*. 

### Test 

After installing `torchquad` through `conda` or `pip`, users can test its correct installation with:

```py
import torchquad
torchquad._deployment_test() 
```

After cloning the repository, developers can check the functionality of `torchquad` by running the following command in the `torchquad/tests` directory:

```sh
pytest
```

**GPU Utilization**

With *conda* you can install the GPU version of PyTorch with `conda install pytorch cudatoolkit -c pytorch`. 
For alternative installation procedures please refer to the [PyTorch Documentation](https://pytorch.org/get-started/locally/).


<!-- USAGE EXAMPLES -->
## Usage

This is a brief example how torchquad can be used to compute a simple integral. For a more thorough introduction please refer to the [tutorial](https://torchquad.readthedocs.io/en/main/tutorial.html) section in the documentation.

The full documentation can be found on [readthedocs](https://torchquad.readthedocs.io/en/main/).

```python
# To avoid copying things to GPU memory, 
# ideally allocate everything in torch on the GPU
# and avoid non-torch function calls
import torch 
from torchquad import MonteCarlo, enable_cuda

# Enable GPU support if available
enable_cuda() 

# The function we want to integrate, in this example f(x0,x1) = sin(x0) + e^x1 for x0=[0,1] and x1=[-1,1]
# Note that the function needs to support multiple evaluations at once (first dimension of x here)
# Expected result here is ~3.2698
def some_function(x):
    return torch.sin(x[:,0]) + torch.exp(x[:,1]) 

# Declare an integrator, here we use the simple, stochastic Monte Carlo integration method
mc = MonteCarlo()

# Compute the function integral by sampling 10000 points over domain 
integral_value = mc.integrate(some_function,dim=2,N=10000,integration_domain = [[0,1],[-1,1]])
```

You can find all available integrators [here](https://torchquad.readthedocs.io/en/main/integration_methods.html).

<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/esa/torchquad/issues) for a list of proposed features (and known issues).


<!-- PERFORMANCE -->
## Performance

Using GPUs torchquad scales particularly well with integration methods that offer easy parallelization. For example, below you see error and runtime results for integrating the function `f(x,y,z) = sin(x * (y+1)²) * (z+1)` on a consumer-grade desktop PC.

![](https://github.com/esa/torchquad/blob/main/resources/torchquad_runtime.png?raw=true)
*Runtime results of the integration. Note the far superior scaling on the GPU (solid line) in comparison to the CPU (dashed and dotted) for both methods.*

![](https://github.com/esa/torchquad/blob/main/resources/torchquad_convergence.png?raw=true)
*Convergence results of the integration. Note that Simpson quickly reaches floating point precision. Monte Carlo is not competitive here given the low dimensionality of the problem.*

<!-- CONTRIBUTING -->
## Contributing

The project is open to community contributions. Feel free to open an [issue](https://github.com/esa/torchquad/issues) or write us an email if you would like to discuss a problem or idea first.

If you want to contribute, please 

1. Fork the project on [GitHub](https://github.com/esa/torchquad). 
2. Get the most up-to-date code by following this quick guide for installing torchquad from source:
     1. Get [miniconda](https://docs.conda.io/en/latest/miniconda.html) or similar
     2. Clone the repo
      ```sh
      git clone https://github.com/esa/torchquad.git
      ```
     3. Setup the environment. This will create a conda environment called `torchquad`
      ```sh
      conda env create -f environment.yml
      conda activate torchquad
      ```

Once the installation is done, then you are ready to contribute. 
Please note that PRs should be created from and into the `develop` branch. For each release the develop branch is merged into main.

3. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
4. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
5. Push to the Branch (`git push origin feature/AmazingFeature`)
6. Open a Pull Request on the `develop` branch, *not* `main` (NB: We autoformat every PR with black. Our GitHub actions may create additional commits on your PR for that reason.)

and we will have a look at your contribution as soon as we can. 

Furthermore, please make sure that your PR passes all automated tests. Review will only happen after that.
Only PRs created on the `develop` branch with all tests passing will be considered. The only exception to this rule is if you want to update the documentation in relation to the current release on conda / pip. In that case you may ask to merge directly into `main`.

<!-- LICENSE -->
## License

Distributed under the GPL-3.0 License. See [LICENSE](https://github.com/esa/torchquad/blob/main/LICENSE) for more information.


<!-- FAQ -->
## FAQ 

  1. Q: `Error enabling CUDA. cuda.is_available() returned False. CPU will be used.`  <br/>A: This error indicates that no CUDA-compatible GPU could be found. Either you have no compatible GPU or the necessary CUDA requirements are missing. Using `conda`, you can install them with `conda install cudatoolkit`. For more detailed installation instructions, please refer to the [PyTorch documentation](https://pytorch.org/get-started/locally/).




<!-- CONTACT -->
## Contact 

Created by ESA's [Advanced Concepts Team](https://www.esa.int/gsp/ACT/index.html)

- Pablo Gómez - `pablo.gomez at esa.int`
- Gabriele Meoni - `gabriele.meoni at esa.int`
- Håvard Hem Toftevaag - `havard.hem.toftevaag at esa.int`

Project Link: [https://github.com/esa/torchquad](https://github.com/esa/torchquad)



<!-- ACKNOWLEDGEMENTS 
This README was based on https://github.com/othneildrew/Best-README-Template
-->
