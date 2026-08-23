# torchquad
<!--
*** Based on https://github.com/othneildrew/Best-README-Template
-->

![Read the Docs (version)](https://img.shields.io/readthedocs/torchquad/main?style=flat-square) [![Tests](https://github.com/esa/torchquad/actions/workflows/run_tests.yml/badge.svg)](https://github.com/esa/torchquad/actions/workflows/run_tests.yml) ![GitHub last commit](https://img.shields.io/github/last-commit/esa/torchquad?style=flat-square)
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
    High-performance numerical integration on the GPU with PyTorch, JAX and Tensorflow
    <br />
    <a href="https://torchquad.readthedocs.io"><strong>Explore the docs »</strong></a>
    <br />
    <br />
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

The torchquad module allows utilizing GPUs for efficient numerical integration with [PyTorch](https://pytorch.org/) and other numerical Python3 modules.
The software is free to use and is designed for the machine learning community and research groups focusing on topics requiring high-dimensional integration.

### Built With

This project is built with the following packages:

* [autoray](https://github.com/jcmgray/autoray), which means the implemented quadrature supports [NumPy](https://numpy.org/) and can be used for machine learning with modules such as [PyTorch](https://pytorch.org/), [JAX](https://github.com/google/jax/) and [Tensorflow](https://www.tensorflow.org/), where it is fully differentiable
* [uv](https://docs.astral.sh/uv/) or [conda](https://docs.conda.io/en/latest/), either of which can set up the required environment for you


If torchquad proves useful to you, please consider citing the [accompanying paper](https://joss.theoj.org/papers/10.21105/joss.03439).

<!-- GOALS -->
## Goals


* **Supporting science**:  Multidimensional numerical integration is needed in many fields, such as physics (from particle physics to astrophysics), in applied finance, in medical statistics, and others. torchquad aims to assist research groups in such fields, as well as the general machine learning community.
* **Withstanding the curse of dimensionality**: The [curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality) makes deterministic methods in particular, but also stochastic ones, computationally expensive when the dimensionality increases. However, many integration methods are [embarrassingly parallel](https://en.wikipedia.org/wiki/Embarrassingly_parallel), which means they can strongly benefit from GPU parallelization. The curse of dimensionality still applies but the improved scaling alleviates the computational impact.
* **Delivering a convenient and functional tool**: torchquad is built with autoray, which means it is [fully differentiable](https://en.wikipedia.org/wiki/Differentiable_programming) if the user chooses, for example, PyTorch as the numerical backend. Furthermore, the library of available and upcoming methods in torchquad offers high-effeciency integration for any need.


<!-- GETTING STARTED -->
## Getting Started

This is a brief guide for how to set up torchquad.

### Prerequisites

torchquad has no backend pinned as a hard dependency — install the numerical backend(s) you want (PyTorch, JAX, TensorFlow) alongside it. It also runs on NumPy alone. Any of pip, [uv](https://docs.astral.sh/uv/), or conda works; pick whichever fits your workflow.

Note that torchquad also works on the CPU; however, it is optimized for GPU usage. GPU support is tested only on NVIDIA cards with CUDA. For GPU installs, follow each framework's own install guide — the CPU-only convenience extras below cannot select GPU wheels, and JAX/TensorFlow GPU builds are Linux/WSL2-only.

For a detailed list of required packages and packages for numerical backends,
please refer to the conda environment files [environment.yml](/environment.yml) and [environment_all_backends.yml](/environment_all_backends.yml).
torchquad requires Python 3.10 or newer. Its CI suite runs on Python 3.12 with JAX 0.4.35, NumPy 2.3, PyTorch 2.5 and TensorFlow 2.18 on Linux; other versions of the backends should work as well but some may require additional setup on other platforms such as Windows.


### Installation

Install torchquad from PyPI:
   ```sh
   pip install torchquad
   # or, with uv:
   uv pip install torchquad
   ```

It is also available on conda-forge:
   ```sh
   conda install torchquad -c conda-forge
   ```

**Adding a backend.** torchquad ships convenience extras that pull a backend
from the default package index:
   ```sh
   pip install "torchquad[torch]"        # PyTorch — CPU on macOS/Windows, CUDA on Linux
   pip install "torchquad[jax]"          # JAX (CPU)
   pip install "torchquad[tensorflow]"   # TensorFlow (CPU)
   pip install "torchquad[all]"          # all three
   ```
`jax` and `tensorflow` install CPU builds; `torch`, however, resolves to the
CUDA build on Linux, because that is what PyTorch publishes to PyPI. For a
guaranteed-CPU torch, install it from the PyTorch CPU index first.

**Adding a backend (GPU).** These extras cannot select GPU wheels (a Python
package cannot encode the CUDA-specific index URLs each framework needs), so for
GPU support install the backend from its own guide, then `pip install torchquad`:
   - PyTorch: <https://pytorch.org/get-started/locally/>
   - JAX (Linux/WSL2 only): <https://docs.jax.dev/en/latest/installation.html>
   - TensorFlow (Linux/WSL2 only): <https://www.tensorflow.org/install/gpu>

For a full multi-backend setup, the conda file
[environment_all_backends.yml](/environment_all_backends.yml) installs every
backend (CPU) in one step:
   ```sh
   conda env create -f environment_all_backends.yml
   conda activate torchquad
   ```


### Test

After installing `torchquad` and PyTorch through `conda` or `pip`,
users can test `torchquad`'s correct installation with:

```py
import torchquad
torchquad._deployment_test()
```

After cloning the repository, developers can check the functionality of `torchquad` by running

```sh
pip install -e .
pytest
```

<!-- USAGE EXAMPLES -->
## Usage

This is a brief example how torchquad can be used to compute a simple integral with PyTorch. For a more thorough introduction please refer to the [tutorial](https://torchquad.readthedocs.io/en/main/tutorial.html) section in the documentation.

The full documentation can be found on [readthedocs](https://torchquad.readthedocs.io/en/main/).

```Python3
# To avoid copying things to GPU memory,
# ideally allocate everything in torch on the GPU
# and avoid non-torch function calls
import torch
from torchquad import MonteCarlo, set_up_backend

# Enable GPU support if available and set the floating point precision
set_up_backend("torch", data_type="float32")

# The function we want to integrate, in this example
# f(x0,x1) = sin(x0) + e^x1 for x0=[0,1] and x1=[-1,1]
# Note that the function needs to support multiple evaluations at once (first
# dimension of x here)
# Expected result here is ~3.2698
def some_function(x):
    return torch.sin(x[:, 0]) + torch.exp(x[:, 1])

# Declare an integrator;
# here we use the simple, stochastic Monte Carlo integration method
mc = MonteCarlo()

# Compute the function integral by sampling 10000 points over domain
integral_value = mc.integrate(
    some_function,
    dim=2,
    N=10000,
    integration_domain=[[0, 1], [-1, 1]],
    backend="torch",
)
```
## Logging Configuration

By default, torchquad disables its internal logging when installed from PyPI to avoid interfering with other loggers in your application. To enable logging change `TORCHQUAD_DISABLE_LOGGING` in `__init__.py`:

1. **Set the log level**: Use the `TORCHQUAD_LOG_LEVEL` environment variable:
   ```bash
   export TORCHQUAD_LOG_LEVEL=DEBUG   # For detailed debugging
   export TORCHQUAD_LOG_LEVEL=INFO    # For general information  
   export TORCHQUAD_LOG_LEVEL=WARNING # For warnings only (default when enabled)
   ```

2. **Enable logging programmatically**:
   ```python
   import torchquad
   torchquad.set_log_level("DEBUG")  # This will enable and configure logging
   ```

## Multi-GPU Usage

torchquad supports multi-GPU systems through standard PyTorch practices. The recommended approach is to use the `CUDA_VISIBLE_DEVICES` environment variable to control GPU selection:

```bash
# Use specific GPU
export CUDA_VISIBLE_DEVICES=0  # Use GPU 0
python your_script.py

export CUDA_VISIBLE_DEVICES=1  # Use GPU 1  
python your_script.py

# Use multiple GPUs with separate processes
export CUDA_VISIBLE_DEVICES=0 && python integration_script.py &
export CUDA_VISIBLE_DEVICES=1 && python integration_script.py &
```

For parallel processing across multiple GPUs, we recommend spawning separate processes rather than trying to coordinate multiple GPUs within a single process. This approach:

- Provides clean separation between GPU processes
- Avoids complex device management
- Follows PyTorch best practices
- Enables easy load balancing and error handling

For detailed examples and advanced multi-GPU patterns, see the [Multi-GPU Usage section](https://torchquad.readthedocs.io/en/main/tutorial.html#multi-gpu-usage) in our documentation.

You can find all available integrators [here](https://torchquad.readthedocs.io/en/main/integration_methods.html).

<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/esa/torchquad/issues) for a list of proposed features (and known issues).


<!-- PERFORMANCE -->
## Performance

Using GPUs, torchquad scales particularly well with integration methods that offer easy parallelization. The benchmarks below demonstrate performance across challenging functions from 1D to 15D, comparing torchquad's GPU-accelerated methods against scipy's CPU implementations.

<!-- TODO Update plot links -->
### Convergence Analysis
![](https://github.com/esa/torchquad/blob/main/resources/torchquad_convergence.png?raw=true)
*Convergence comparison across challenging test functions from 1D to 15D. GPU-accelerated torchquad methods demonstrate great performance, particularly for high-dimensional integration where scipy's nquad becomes computationally infeasible. Beyond 1D, torchquad significantly outperforms scipy in efficiency.*

### Runtime vs Error Efficiency  
![](https://github.com/esa/torchquad/blob/main/resources/torchquad_runtime_vs_error.png?raw=true)
*Runtime-error trade-offs across dimensions. Lower-left positions indicate better performance. While scipy's traditional methods are competitive for simple 1D problems, torchquad's GPU acceleration provides orders of magnitude better performance for multi-dimensional integration, achieving both faster computation and lower errors.*

### Scaling Performance
![](https://github.com/esa/torchquad/blob/main/resources/torchquad_scaling_analysis.png?raw=true)
*Scaling investigation across problem sizes and dimensions of the different methods in torchquad.*

### Vectorized Integration Speedup
![](https://github.com/esa/torchquad/blob/main/resources/torchquad_vectorized_speedup.png?raw=true)
*Strong performance gains when evaluating multiple integrands simultaneously. The vectorized approach shows exponential speedup (up to 200x) compared to sequential evaluation, making torchquad ideal for parameter sweeps, uncertainty quantification, and machine learning applications requiring batch integration.*

### Framework Comparison  
![](https://github.com/esa/torchquad/blob/main/resources/torchquad_framework_comparison.png?raw=true)
*Cross-framework performance comparison for 1D integration using Monte Carlo and Simpson methods. Demonstrates torchquad's consistent API across PyTorch, TensorFlow, JAX, and NumPy backends, with GPU acceleration providing significant performance advantages for large number of function evaluations. All frameworks achieve similar accuracy while showcasing the computational benefits of GPU acceleration for parallel integration methods.*

### Running Benchmarks

To reproduce these benchmarks or test performance on your hardware:

```bash
# Run all benchmarks (convergence, framework comparison, scaling, vectorized)
python benchmarking/modular_benchmark.py --dimensions 1,3,7,15

# Run specific benchmark types
python benchmarking/modular_benchmark.py --convergence-only --dimensions 1,3,7,15
python benchmarking/modular_benchmark.py --scaling-only
python benchmarking/modular_benchmark.py --framework-only

# Generate all plots from results
python benchmarking/plot_results.py

# Configure benchmark parameters
# Edit benchmarking/benchmarking_cfg.toml to adjust:
# - Evaluation point ranges
# - Framework backends to test
# - Timeout limits  
# - Method selection
# - scipy integration tolerances
```

**New Features:**
- **Analytic Reference Values**: Uses SymPy for exact analytic solutions where possible, providing highly accurate reference values for error calculations
- **Enhanced Test Functions**: Analytically tractable but numerically challenging functions that better demonstrate convergence behavior
- **Framework Comparison**: Cross-backend performance benchmarking across PyTorch, TensorFlow, JAX, and NumPy with GPU/CPU device comparisons

**Hardware:** RTX 4060 Ti 16GB, i5-13400F, Precision: float32

<!-- CONTRIBUTING -->
## Contributing

The project is open to community contributions. Feel free to open an [issue](https://github.com/esa/torchquad/issues) or write us an email if you would like to discuss a problem or idea first.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide — how to set up a
development environment, the checks CI runs, and the review process. In short:
fork the repo, branch off `develop`, and open your pull request against
`develop` (not `main`). Documentation fixes for the *current release* are the
only exception and may target `main` directly.

<!-- LICENSE -->
## License

Distributed under the GPL-3.0 License. See [LICENSE](https://github.com/esa/torchquad/blob/main/LICENSE) for more information.


<!-- FAQ -->
## FAQ

  1. Q: `Error enabling CUDA. cuda.is_available() returned False. CPU will be used.`  <br/>A: This error indicates that PyTorch could not find a CUDA-compatible GPU. Either you have no compatible GPU or your PyTorch build has no CUDA support. Install a CUDA-enabled PyTorch build following the [PyTorch install guide](https://pytorch.org/get-started/locally/).




<!-- CONTACT -->
## Contact

Created by ESA's [Advanced Concepts Team](https://www.esa.int/gsp/ACT/index.html)

- Pablo Gómez - `pablo.gomez at esa.int`
- Gabriele Meoni - `gabriele.meoni at esa.int`
- Håvard Hem Toftevaag

Project Link: [https://github.com/esa/torchquad](https://github.com/esa/torchquad)



<!-- ACKNOWLEDGEMENTS
This README was based on https://github.com/othneildrew/Best-README-Template
-->
