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
    <img src="https://github.com/esa/torchquad/blob/main/logos/torchquad_white_background_PNG.png?raw=true" alt="Logo" width="280" height="120">
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

torchquad is silent by default so that importing it never interferes with the
logging of the application using it. Turn its log records on in either of two ways:

1. **Set the `TORCHQUAD_LOG_LEVEL` environment variable** before importing torchquad:
   ```bash
   export TORCHQUAD_LOG_LEVEL=DEBUG
   export TORCHQUAD_LOG_LEVEL=INFO
   export TORCHQUAD_LOG_LEVEL=WARNING
   ```
   Leaving it unset — or setting it to the empty string — keeps torchquad silent.
   An unrecognised level raises at import rather than being ignored.

2. **Enable logging programmatically**:
   ```python
   import torchquad
   torchquad.set_log_level("DEBUG")  # This will enable and configure logging
   ```

torchquad only ever adds its own handler, filtered to its own records, and never
removes one your application registered.

One consequence is worth knowing: the level applies to torchquad's own handler,
which is the only one it owns. Once records are enabled they also reach every
other sink [loguru](https://github.com/Delgan/loguru) has registered — including
loguru's default stderr handler, which is unfiltered and sits at `DEBUG`. So in a
plain interpreter you will see torchquad's records twice, and below the level you
asked for. That is loguru's global state, not something a library can change
without disturbing its host ([#184](https://github.com/esa/torchquad/issues/184)).
To control it, configure loguru yourself:

```python
from loguru import logger
logger.remove()                      # drop loguru's default handler
logger.add(sys.stderr, level="WARNING")

import torchquad
torchquad.set_log_level("WARNING")
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

All figures below were measured on an RTX 4060 Ti / i5-13400F. Accuracy is
measured against closed-form integrals from the
[Genz test-function family](benchmarking/genz_functions.py) in float64; runtimes
are float32 and synchronize the GPU before the clock stops.

### Convergence
![](https://github.com/esa/torchquad/blob/main/resources/torchquad_convergence_3d.png?raw=true)
*Relative error against the number of function evaluations, 3D, on the Genz oscillatory integrand. The higher-order Newton-Cotes rules separate cleanly — Trapezoid reaches 7.7e-04, Simpson 1.6e-07, Boole 2.4e-10 — and Gauss-Legendre hits double precision within a few hundred points. This is the plot to read first: it says what accuracy a given budget buys you.*

### Quasi-Monte Carlo vs Monte Carlo
![](https://github.com/esa/torchquad/blob/main/resources/torchquad_qmc_vs_mc_3d.png?raw=true)
*Passing `rng=Sobol(...)` to `MonteCarlo` replaces pseudo-random points with a low-discrepancy sequence. On a non-separable integrand this is worth five to six orders of magnitude — 2.8e-09 against 1.0e-03 at the same budget — and converges faster than O(N⁻¹) where plain Monte Carlo tracks O(N⁻¹ᐟ²). Errors are the median over five seeds.*

### Scaling with dimension
![](https://github.com/esa/torchquad/blob/main/resources/torchquad_dimension_scaling.png?raw=true)
*Error at a fixed budget of N=2¹⁶ as the dimension grows, with per-dimension difficulty held constant so the curve shows the cost of dimension rather than of a harder integrand. Plain Monte Carlo is famously dimension-insensitive and stays near 1e-03 throughout; the quasi-random advantage is largest in low dimensions and narrows as the dimension climbs, though at d=10 Sobol is still around 5e-05 against 3.6e-03.*

### Runtime: CPU vs GPU
![](https://github.com/esa/torchquad/blob/main/resources/torchquad_runtime_cpu_vs_gpu.png?raw=true)
*Wall-clock time per integration, measured in separate processes because the backend's default device is global state. At N=1e8 the GPU is 33x faster for Monte Carlo and 13x for Simpson. The crossover is shown honestly rather than hidden: for Simpson the CPU is the faster choice below roughly 1e6 evaluations, where the problem is too small to cover kernel-launch overhead.*

### Framework comparison
![](https://github.com/esa/torchquad/blob/main/resources/torchquad_framework_comparison.png?raw=true)
*The same 1D integration through each backend, run in isolated subprocesses. PyTorch and TensorFlow land within about 10% of each other on the GPU (66 ms and 73 ms at N=1e8 for Monte Carlo), and all backends reach comparable accuracy — which is the point of a single API across four numerical libraries.*

### Vectorized integration
![](https://github.com/esa/torchquad/blob/main/resources/torchquad_vectorized_speedup.png?raw=true)
*Integrating many integrands in one batched call against looping over them one at a time. The speedup grows roughly linearly with the number of integrands, because the loop pays a kernel launch per integrand while the batched call pays one: about 16x at 20 integrands and 110x at 200 on this machine. The exact figure at the top end is platform-dependent — the batched side is only 1-2 ms, close to the measurement floor — so treat the shape rather than the peak number as the result.*

### Comparison with SciPy

![](https://github.com/esa/torchquad/blob/main/resources/torchquad_vs_scipy_combined.png?raw=true)

Earlier versions of this section compared torchquad against `scipy.integrate.nquad`
and claimed a broad efficiency win. That was not a comparison we can stand behind,
so here is one we can — against SciPy's *best* configuration rather than its
weakest, on a deliberately hard integrand, measured both ways.

The test function sums three Genz integrands, each normalised to contribute
equally: one that oscillates, one whose mass concentrates in a corner, and one
that is continuous but **not differentiable**. Each defeats a different method, so
no single feature can flatter one library. Summing keeps the integral exact,
because integration is linear.

The top row of the figure is error against **function evaluations** — that
compares algorithms and is independent of hardware. The bottom row is error
against **runtime**, which is what you actually wait for but folds in the fact
that SciPy runs on the CPU while torchquad here runs on a GPU.

Both sides get the same 50-million-evaluation budget, so neither is being starved.

| | best torchquad | best SciPy |
|---|---|---|
| **d=3** | Boole **8.3e-12** @ 38M evals, **0.10 s** | `nquad` **4.4e-16** @ 250k · Genz-Malik 1.0e-12 @ 7.3M, 7.9 s |
| **d=6** | Boole **2.9e-05** @ 24M evals, **0.10 s** | Genz-Malik **1.8e-05** @ 1.7M evals, 0.41 s |
| **d=10** | VEGAS **4.8e-04** @ 11M evals, 0.60 s | *none completed* |

**SciPy's algorithms are more efficient per evaluation at low dimension.** At d=3
`nquad` reaches machine precision from a quarter of a million points; torchquad
needs 38 million to get to 8e-12 and never closes the last four orders. Adaptive
subdivision is simply the right approach for a low-dimensional integrand with a
localized feature, and torchquad does not implement it.

**GPU throughput can offset that, but only in wall-clock.** At d=3 Boole reaches
8.3e-12 in 0.10 s against Genz-Malik's 1.0e-12 in 7.9 s — comparable accuracy,
about 80x faster. By d=6 the two are near parity: SciPy is 1.6x more accurate on
14x fewer evaluations, torchquad is 4x faster in wall-clock. Which matters depends
on whether your integrand is cheap or expensive to evaluate.

**Past that, dimension decides it.** At d=10 every SciPy configuration here
fails: the default `gk21` rule is a *product* rule needing 21^d nodes, an
impossible 121 TiB allocation, and the other two exhaust the evaluation budget.
torchquad returns 4.8e-04 in 0.6 s.

Two things in this figure are worth knowing when picking a method:

- **The highest-order rule is not the best one here.** Gauss-Legendre applies a
  single global high-degree rule, which a kink defeats badly; Boole is composite,
  applying a lower-order rule piecewise, and beats it by six orders of magnitude
  at d=3 and nearly 300x at d=6. Reach for Boole on integrands that are not
  smooth everywhere.
- **VEGAS earns its keep as the dimension grows.** It is the *worst* torchquad
  method at d=3 (1.3e-04, behind everything) because it spends its early
  iterations learning the integrand, and the *best* at d=10, where adapting to
  where the mass actually lies beats sampling uniformly.

And the axis no SciPy comparison can capture at all: gradients through the
integral, GPU throughput at large N, and the same API across four numerical
backends.

### Running Benchmarks

To reproduce these benchmarks or test performance on your hardware:

```bash
# The accuracy figures: convergence, QMC vs MC, dimension scaling, CPU vs GPU
python benchmarking/release_plots.py --plots all
python benchmarking/release_plots.py --plots qmc,convergence   # or a subset

# The timing harness: scaling, framework comparison, vectorized
python benchmarking/modular_benchmark.py --dimensions 1,3,7,15
python benchmarking/modular_benchmark.py --scaling-only
python benchmarking/modular_benchmark.py --framework-only

# Redraw the harness figures from the results of the run above
python benchmarking/plot_results.py

# Configure benchmark parameters
# Edit benchmarking/benchmarking_cfg.toml to adjust:
# - Evaluation point ranges
# - Framework backends to test
# - Timeout limits
# - Method selection
# - scipy integration tolerances
```

Two notes if you re-measure:

- **Run the machine idle.** The harness synchronizes the device before stopping
  its clock, so the numbers are real wall-clock time and will pick up anything
  else competing for the GPU.
- **TensorFlow and PyTorch cannot share one environment on the GPU.** They pin
  conflicting versions of the bundled NVIDIA CUDA libraries, and whichever loses
  falls back to the CPU silently, turning a GPU comparison into a CPU one. Give
  each its own interpreter via the `[interpreters]` section of the config.

**Hardware for the figures above:** RTX 4060 Ti 16GB, i5-13400F. Accuracy in
float64, timings in float32.

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
