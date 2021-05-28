# torchquad
<!--
*** Based on https://github.com/othneildrew/Best-README-Template
-->

[![Documentation Status](https://readthedocs.org/projects/torchquad/badge/?version=latest)](https://torchquad.readthedocs.io/en/latest/?badge=latest)



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

* [PyTorch](https://pytorch.org/)
* [conda](https://docs.conda.io/en/latest/)


<!-- GOALS -->
## Goals

* **Progressing science**:  Multidimensional integration is needed in many fields of physics (from particle physics to astrophysics), in applied finance, in medical statistics, and so on. With torchquad, we wish to reach research groups in such fields, as well as the general machine learning community.
* **Withstanding the curse of dimensionality**: The [curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality) makes deterministic methods in particular, but also stochastic ones, extremely slow when the dimensionality increases. This gives the researcher a choice between computationally heavy and time-consuming simulations on the one hand and inaccurate evaluations on the other. Luckily, many integration methods are [embarrassingly parallel](https://en.wikipedia.org/wiki/Embarrassingly_parallel), which means they can strongly benefit from GPU parallelization. The curse of dimensionality still applies, but GPUs can handle the problem much better than CPUs can.
* **Delivering a convenient and functional tool**: torchquad is built with [PyTorch](https://pytorch.org/), which means it is [fully differentiable](https://en.wikipedia.org/wiki/Differentiable_programming). Furthermore, the library of available and upcoming methods in torchquad offers high-effeciency integration for any need. 


<!-- GETTING STARTED -->
## Getting Started

This is a brief guide for how to set up torchquad.

### Prerequisites

We recommend using [conda](https://docs.conda.io/en/latest/), especially if you want to utilize the GPU. It will automatically set up CUDA and the cudatoolkit for you in that case.
Note that torchquad also works on the CPU. However, it is optimized for GPU usage.

* [conda](https://docs.conda.io/en/latest/), which will take care of all requirements for you. For a detailed list of required packages, please refer to the [conda environment file](https://github.com/esa/torchquad/blob/main/environment.yml).

### Installation

1. Get [miniconda](https://docs.conda.io/en/latest/miniconda.html) or similar
2. Clone the repo
   ```sh
   git clone https://github.com/esa/torchquad.git
   ```
3. Setup the environment. This will create a conda environment called `torchquad`
   ```sh
   conda env create -f environment.yml
   ```

Alternatively you can use
   ```sh
   pip install torchquad
   ```

NB Note that `pip` will not set up PyTorch with CUDA and GPU support. Therefore, we recommend to use `conda`.

**GPU Utilization**

With `conda` you can install the GPU version of PyTorch with `conda install pytorch cudatoolkit -c pytorch`. 
For alternative installation procedures please refer to the [PyTorch Documentation](https://pytorch.org/get-started/locally/).


<!-- USAGE EXAMPLES -->
## Usage

This is a brief example how torchquad can be used to compute a simple integral. For a more thorough introduction please refer to the [example notebook](https://github.com/esa/torchquad/blob/main/notebooks/Example_notebook.ipynb).

The full documentation can be found on [readthedocs](https://torchquad.readthedocs.io/en/latest/).

```python
# To avoid copying things to GPU memory, 
# ideally allocate everything in torch on the GPU
# and avoid non-torch function calls
import torch 
from torchquad import MonteCarlo

# The function we want to integrate, in this example f(x,y) = sin(x) + e^y
def some_function(x):
    return torch.sin(x[0]) + torch.exp(x[1])

# Declare an integrator, here we use the simple, stochastic Monte Carlo integration method
mc = MonteCarlo()

# Compute the function integral by sampling 10000 points over domain 
integral_value = mc.integrate(some_function,dim=2,N=10000,integration_domain = [[0,1],[-1,1]])
```

You can find all available integrators [here](https://torchquad.readthedocs.io/en/latest/integration_methods.html).

<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/esa/torchquad/issues) for a list of proposed features (and known issues).


<!-- CONTRIBUTING -->
## Contributing

The project is open to community contributions. Feel free to open an [issue](https://github.com/esa/torchquad/issues) or write us an email if you would like to discuss a problem or idea first.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



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
