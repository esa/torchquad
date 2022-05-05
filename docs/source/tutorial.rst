.. _tutorial:

Tutorial
===============

*torchquad* is a dedicated module for numerical integration in arbitrary dimensions.
This tutorial gives a more detailed look at its functionality and explores some performance considerations.


Minimal working example
-----------------------

.. code:: ipython3

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

To set the default logger verbosity, change the ``TORCHQUAD_LOG_LEVEL``
environment variable; for example ``export TORCHQUAD_LOG_LEVEL=DEBUG``.
A :ref:`later section <tutorial_backend_selection>` in this tutorial shows how
to choose a different numerical backend.


Detailed Introduction
---------------------

The main problem with higher-dimensional numerical integration is that
the computation simply becomes too costly if the dimensionality, *n*, is large, as the number
of evaluation points increases exponentially - this problem is known as
the *curse of dimensionality*. This especially affects grid-based
methods, but is, to some degree, also present for Monte Carlo methods,
which also require larger numbers of points for convergence in higher
dimensions.

At the time, *torchquad* offers the following integration methods for
abritrary dimensionality.

+--------------+-------------------------------------------------+------------+
| Name         | How it works                                    | Spacing    |
|              |                                                 |            |
+==============+=================================================+============+
| Trapezoid    | Creates a linear interpolant between two |br|   | Equal      |
| rule         | neighbouring points                             |            |
+--------------+-------------------------------------------------+------------+
| Simpson's    | Creates a quadratic interpolant between |br|    | Equal      |
| rule         | three neighbouring point                        |            |
+--------------+-------------------------------------------------+------------+
| Boole's      | Creates a more complex interpolant between |br| | Equal      |
| rule         | five neighbouring points                        |            |
+--------------+-------------------------------------------------+------------+
| Monte Carlo  | Randomly chooses points at which the |br|       | Random     |
|              | integrand is evaluated                          |            |
+--------------+-------------------------------------------------+------------+
| VEGAS        | Adaptive multidimensional Monte Carlo |br|      | Stratified |
| Enhanced     | integration (VEGAS with adaptive stratified     | |br|       |
| |br| (VEGAS+)| |br| sampling)                                  | sampling   |
+--------------+-------------------------------------------------+------------+

.. |br| raw:: html

     <br>


Outline
-------

This tutorial is a guide for new users to *torchquad* and is structured in
the following way:

1.  Example integration in one dimension (1-D) with PyTorch
2.  Example integration in ten dimensions (10-D) with PyTorch
3.  Some accuracy / runtime comparisons with scipy
4.  Information on how to select a numerical backend
5.  Example showing how gradients can be obtained w.r.t. the integration domain with PyTorch
6.  Methods to speed up the integration

Feel free to test the code on your own computer as we go along.

Imports
-------

Now let's get started! First, the general imports:

.. code:: ipython3

    import scipy
    import numpy as np

    # For benchmarking
    import time
    from scipy.integrate import nquad

    # For plotting
    import matplotlib.pyplot as plt

    # To avoid copying things to GPU memory,
    # ideally allocate everything in torch on the GPU
    # and avoid non-torch function calls
    import torch
    torch.set_printoptions(precision=10) # Set displayed output precision to 10 digits

    from torchquad import set_up_backend  # Necessary to enable GPU support
    from torchquad import Trapezoid, Simpson, Boole, MonteCarlo, VEGAS # The available integrators
    from torchquad.utils.set_precision import set_precision
    import torchquad

.. code:: ipython3

    # Use this to enable GPU support and set the floating point precision
    set_up_backend("torch", data_type="float32")




One-dimensional integration
------------------------------------------------


To make it easier to understand the methods used in this notebook, we will start with an
example in one dimension. If you are new to these methods or simply want a clearer picture,
feel free to check out Patrick Walls'
`nice Python introduction <https://github.com/patrickwalls/mathematical-python/>`__
to the `Trapezoid rule <https://www.math.ubc.ca/~pwalls/math-python/integration/trapezoid-rule/>`__
and `Simpson's rule <https://www.math.ubc.ca/~pwalls/math-python/integration/simpsons-rule/>`__
in one dimension.
Similarly, `Tirthajyoti Sarkar <https://github.com/tirthajyoti>`__ has made a nice visual explanation of
`Monte Carlo integration in Python
<https://towardsdatascience.com/monte-carlo-integration-in-python-a71a209d277e>`__.

Let ``f(x)`` be the function :math:`f(x) = e^{x} \cdot x^{2}`. Over the domain
:math:`[0,2]`, the integral of ``f(x)`` is :math:`\int_{0}^{2} f(x) dx =
\int_{0}^{2} e^x \cdot x^2 dx = 2(e^{2} - 1) = 12.7781121978613004544...`

Let's declare the function and a simple function to print the absolute error,
as well as remember the correct result.

.. code:: ipython3

    def f(x):

        return torch.exp(x) * torch.pow(x, 2)

    def print_error(result, solution):
        print("Results:", result.item())
        print(f"Abs. Error: {(torch.abs(result - solution).item()):.8e}")
        print(f"Rel. Error: {(torch.abs((result - solution) / solution).item()):.8e}")

    solution = 2 * (torch.exp(torch.tensor([2.0])) - 1)

**Note that we are using the torch versions of functions like ``exp`` to ensure that all variables
are and stay on the GPU.** 
**Also, note:** the unit imaginary number :math:`i` is written as ``j`` in Python.

Let's plot the function briefly.

.. code:: ipython3

    points = torch.linspace(0, 2, 100)
    # Note that for plotting we have to move the values to the CPU first
    plt.plot(points.cpu(), f(points).cpu())
    plt.xlabel("$x$", fontsize=14)
    plt.ylabel("f($x$)", fontsize=14)


Let's define the integration domain, set the precision to double, and initialize the integrator - let's start with the trapezoid rule.

.. code:: ipython3

    # Integration domain is a list of lists to allow arbitrary dimensionality.
    integration_domain = [[0, 2]]
    # Initialize a trapezoid solver
    tp = Trapezoid()

Now we are all set to compute the integral. Let's try it with just 101 sample points for now.

.. code:: ipython3

    result = tp.integrate(f, dim=1, N=101, integration_domain=integration_domain)
    print_error(result, solution)


.. parsed-literal::

    **Output**: Results: 12.780082702636719
            Abs. Error: 1.97029114e-03
            Rel. Error: 1.54192661e-04

This is quite close already, as 1-D integrals are comparatively easy.
Let's see what type of value we get for different integrators.

.. code:: ipython3

    simp = Simpson()
    result = simp.integrate(f, dim=1, N=101, integration_domain=integration_domain)
    print_error(result, solution)


.. parsed-literal::

    **Output:** Results: 12.778112411499023
            Abs. Error: 0.00000000e+00
            Rel. Error: 0.00000000e+00


.. code:: ipython3

    mc = MonteCarlo()
    result = mc.integrate(f, dim=1, N=101, integration_domain=integration_domain)
    print_error(result, solution)


.. parsed-literal::

    **Output:** Results: 13.32831859588623
            Abs. Error: 5.50206184e-01
            Rel. Error: 4.30584885e-02


.. code:: ipython3

    vegas = VEGAS()
    result = vegas.integrate(f, dim=1, N=101, integration_domain=integration_domain)
    print_error(result, solution)


.. parsed-literal::

    **Output:** Results: 21.83991813659668
            Abs. Error: 9.06180573e+00
            Rel. Error: 7.09166229e-01


Notably, Simpson's method is already sufficient for a perfect solution here with 101 points.
Monte Carlo methods do not perform so well; they are more suited to higher-dimensional integrals.
VEGAS currently requires a larger number of samples to function correctly (as it performs several
iterations).

Let's step things up now and move to a ten-dimensional problem.

High-dimensional integration
----------------------------

Now, we will investigate the following ten-dimensional problem:

Let ``f_2`` be the function :math:`f_{2}(x) = \sum_{i=1}^{10} \sin(x_{i})`.

Over the domain :math:`[0,1]^{10}`, the integral of ``f_2`` is
:math:`\int_{0}^{1} \dotsc \int_{0}^{1} \sum_{i=1}^{10} \sin(x_{i}) = 20 \sin^{2}(1/2) = 4.59697694131860282599063392557 \dotsc`

Plotting this is tricky, so let's directly move to the integrals.

.. code:: ipython3

    def f_2(x):
        return torch.sum(torch.sin(x), dim=1)


    solution = 20 * (torch.sin(torch.tensor([0.5])) * torch.sin(torch.tensor([0.5])))

Let's start with just 3 points per dimension, i.e., :math:`3^{10}=59,049` sample points.

**Note**: *torchquad* currently only supports equal numbers of points per dimension.
We are working on giving the user more flexibility on this point.

.. code:: ipython3

    # Integration domain is a list of lists to allow arbitrary dimensionality
    integration_domain = [[0, 1]] * 10
    N = 3 ** 10

.. code:: ipython3

    tp = Trapezoid()  # Initialize a trapezoid solver
    result = tp.integrate(f_2, dim=10, N=N, integration_domain=integration_domain)
    print_error(result, solution)


.. parsed-literal::

    **Output:** Results: 4.500804901123047
            Abs. Error: 9.61723328e-02
            Rel. Error: 2.09207758e-02

.. code:: ipython3

    simp = Simpson()  # Initialize Simpson solver
    result = simp.integrate(f_2, dim=10, N=N, integration_domain=integration_domain)
    print_error(result, solution)


.. parsed-literal::

    **Output:** Results: 4.598623752593994
            Abs. Error: 1.64651871e-03
            Rel. Error: 3.58174206e-04

.. code:: ipython3

    boole = Boole()  # Initialize Boole solver
    result = boole.integrate(f_2, dim=10, N=N, integration_domain=integration_domain)
    print_error(result,solution)


.. parsed-literal::

    **Output:** Results: 4.596974849700928
            Abs. Error: 2.38418579e-06
            Rel. Error: 5.18642082e-07
            


.. code:: ipython3

    mc = MonteCarlo()
    result = mc.integrate(f_2, dim=10, N=N, integration_domain=integration_domain, seed=42)
    print_error(result, solution)


.. parsed-literal::

    **Output:** Results: 4.598303318023682
            Abs. Error: 1.32608414e-03
            Rel. Error: 2.88468727e-04

.. code:: ipython3

    vegas = VEGAS()
    result = vegas.integrate(f_2, dim=10, N=N, integration_domain=integration_domain)
    print_error(result, solution)


.. parsed-literal::

    **Output:** Results: 4.598696708679199
            Abs. Error: 1.71947479e-03
            Rel. Error: 3.74044670e-04

Note that the Monte Carlo methods are much more competitive in this case. 
The bad convergence properties of the trapezoid method are visible while Simpson's 
and Boole's rule are still OK given the comparatively smooth integrand. 

If you have been repeating the examples from this tutorial on your own computer, you 
might get ``RuntimeError: CUDA out of memory`` if you have a small GPU.
In that case, you could also try to reduce the number of sample points (e.g., 3 per dimension).
You can really see the curse of dimensionality fully at play here, since :math:`5^{10}=9,765,625`
but :math:`3^{10}=59,049`, reducing the number of sample points by a factor of 165.
Note, however, that Boole's method cannot work for only 3 points per dimension, so the number of sample points is therefore 
automatically increased to 5 per dimension for this method.

Comparison with scipy
---------------------

Let's explore how *torchquad*'s performance compares to scipy, the go-to
tool for numerical integration. A more detailed exploration of this
topic might be done as a side project at a later time. For simplicity,
we will stick to a 5-D version of the :math:`\sin(x)` of the previous
section. Let's declare it with numpy and torch. NumPy arrays will
remain on the CPU and torch tensor on the GPU.

.. code:: ipython3

    dimension = 5
    integration_domain = [[0, 1]] * dimension
    ground_truth = 2 * dimension * np.sin(0.5) * np.sin(0.5)

    def f_3(x):
        return torch.sum(torch.sin(x), dim=1)

    def f_3_np(*x):
        return np.sum(np.sin(x))

Now let's evaluate the integral using the scipy function ``nquad``.

.. code:: ipython3

    start = time.time()
    opts = {"limit": 10, "epsabs": 1, "epsrel": 1}
    result, _, details = nquad(f_3_np, integration_domain, opts=opts, full_output=True)
    end = time.time()
    print("Results:", result)
    print("Abs. Error:", np.abs(result - ground_truth))
    print(details)
    print(f"Took {(end - start) * 1000.0:.3f} ms")


.. parsed-literal::

    **Output:** Results: 2.2984884706593016
            Abs. Error: 0.0
            {'neval': 4084101}
            Took 33067.629 ms


Using scipy, we get the result in about 33 seconds on the authors'
machine (this might take shorter or longer on your machine). The integral was computed with
``nquad``, which on the inside uses the highly adaptive
`QUADPACK <https://en.wikipedia.org/wiki/QUADPACK>`__ algorithm.

In any event, *torchquad* can reach the same accuracy much, much quicker
by utilizing the GPU.

.. code:: ipython3

    N = 37 ** dimension
    simp = Simpson()  # Initialize Simpson solver
    start = time.time()
    result = simp.integrate(f_3, dim=dimension, N=N, integration_domain=integration_domain)
    end = time.time()
    print_error(result, ground_truth)
    print("neval=", N)
    print(f"Took {(end - start) * 1000.0:.3f} ms")


If you tried this yourself and ran out of CUDA memory, simply decrease :math:`N`
(this will, however, lead to a loss of accuracy).

Note that we use more evaluation points (:math:`37^{5}=69,343,957` for *torchquad* vs. :math:`4,084,101`
for scipy), given the comparatively simple algorithm.
Anyway, the decisive factor for this specific problem is runtime. A comparison with regard to
function evaluations is difficult, as ``nquad`` provides no support for a
fixed number of evaluations. This may follow in the future.

The results from using Simpson's rule in *torchquad* is: 

.. parsed-literal::

    **Output:** Results: 2.2984883785247803
            Abs. Error: 0.00000000e+00
            Rel. Error: 0.00000000e+00
            neval= 69343957
            Took 162.147 ms


In our case, *torchquad*  with Simpson's rule was more than 300 times faster than
``scipy.integrate.nquad``. We will add
more elaborate integration methods over time; however, this tutorial should
already showcase the advantages of numerical integration on the GPU.

Reasonably, one might prefer Monte Carlo integration methods for a 5-D
problem. We might add this comparison to the tutorial in the future.


.. _tutorial_backend_selection:

Using different backends with torchquad
---------------------------------------

This section shows how to select a different numerical backend for the quadrature.
Let's change the minimal working example so that it uses Tensorflow instead of PyTorch:

.. code:: ipython3

    import tensorflow as tf
    from torchquad import MonteCarlo, set_up_backend

    # Enable Tensorflow's NumPy behaviour and set the floating point precision
    set_up_backend("tensorflow", data_type="float32")

    # The integrand function rewritten for Tensorflow instead of PyTorch
    def some_function(x):
        return tf.sin(x[:, 0]) + tf.exp(x[:, 1])

    mc = MonteCarlo()
    # Set the backend argument to "tensorflow" instead of "torch"
    integral_value = mc.integrate(
        some_function,
        dim=2,
        N=10000,
        integration_domain=[[0, 1], [-1, 1]],
        backend="tensorflow",
    )

As the name suggests, the ``set_up_backend`` function configures a numerical
backend so that it works with torchquad and it optionally sets the floating
point precision.
For Tensorflow this means in our code it enables
`NumPy behaviour <https://www.tensorflow.org/guide/tf_numpy>`__
and configures torchquad so that it uses float32 precision when initialising
Tensors for Tensorflow.
More details about :func:`torchquad.set_up_backend` can be found in its
documentation.

To calculate an integral with Tensorflow we changed the ``backend`` argument of
the ``integrate`` method.
An alternative way to select Tensorflow as backend is to set the
``integration_domain`` argument to a ``tf.Tensor`` instead of a list.

The other code changes we did, for example rewriting the integrand,
are not directly related to torchquad.
To use NumPy or JAX we would analogously need to change the two backend
arguments to ``"numpy"`` resp. ``"jax"`` and rewrite the integrand function.


Computing gradients with respect to the integration domain
----------------------------------------------------------

*torchquad* allows fully automatic differentiation. In this tutorial, we will show how to extract the gradients with respect to the integration domain with the PyTorch backend.
We selected the composite Trapezoid rule and the Monte Carlo method to showcase that getting gradients is possible for both deterministic and stochastic methods.


.. code:: ipython3

    import torch
    from torchquad import MonteCarlo, Trapezoid, set_up_backend
    
    def test_function(x):
        """V shaped test function."""
        return 2 * torch.abs(x)

    set_up_backend("torch", data_type="float64")
    # Number of function evaluations
    N = 10000

    # Calculate a gradient with the MonteCarlo integrator
    # Define the integrator
    integrator_mc = MonteCarlo()
    # Integration domain
    domain = torch.tensor([[-1.0, 1.0]])
    # Enable the creation of a computational graph for gradient calculation.
    domain.requires_grad = True
    # Calculate the 1-D integral by using the previously defined test_function
    # with MonteCarlo; set a RNG seed to get reproducible results
    result_mc = integrator_mc.integrate(
        test_function, dim=1, N=N, integration_domain=domain, seed=0
    )
    # Compute the gradient with a backward pass
    result_mc.backward()
    gradient_mc = domain.grad

    # Calculate a gradient analogously with the composite Trapezoid integrator
    integrator_tp = Trapezoid()
    domain = torch.tensor([[-1.0, 1.0]])
    domain.requires_grad = True
    result_tp = integrator_tp.integrate(
        test_function, dim=1, N=N, integration_domain=domain
    )
    result_tp.backward()
    gradient_tp = domain.grad

    # Show the results
    print(f"Gradient result for MonteCarlo: {gradient_mc}")
    print(f"Gradient result for Trapezoid: {gradient_tp}")

The code above calculates the integral for a 1-D test-function ``test_function()`` in the [-1,1] domain and prints the gradients with respect to the integration domain.
The command ``domain.requires_grad = True`` enables the creation of a computational graph, and it shall be called before calling the ``integrate(...)`` method.
Gradients computation is, then, performed calling ``result.backward()``.
The output of the print statements is as follows:

.. parsed-literal::

    Gradient result for MonteCarlo: tensor([[-1.9828,  2.0196]])
    Gradient result for Trapezoid: tensor([[-2.0000,  2.0000]])


Speedups for repeated quadrature
--------------------------------

Compiling the integrate method
``````````````````````````````

To speed up the quadrature in situations where it is executed often with the
same number of points ``N`` and dimensionality ``dim``,
we can JIT-compile the performance-relevant parts of the integrate method:

.. code:: ipython3

    import time
    import torch
    from torchquad import Boole, set_up_backend


    def example_integrand(x):
        return torch.sum(torch.sin(x), dim=1)


    set_up_backend("torch", data_type="float32")
    N = 912673
    dim = 3
    integrator = Boole()
    domains = [torch.tensor([[-1.0, y]] * dim) for y in range(5)]

    # Integrate without compilation
    times_uncompiled = []
    for integration_domain in domains:
        t0 = time.perf_counter()
        integrator.integrate(example_integrand, dim, N, integration_domain)
        times_uncompiled.append(time.perf_counter() - t0)

    # Integrate with partial compilation
    integrate_jit_compiled_parts = integrator.get_jit_compiled_integrate(
        dim, N, backend="torch"
    )
    times_compiled_parts = []
    for integration_domain in domains:
        t0 = time.perf_counter()
        integrate_jit_compiled_parts(example_integrand, integration_domain)
        times_compiled_parts.append(time.perf_counter() - t0)

    # Integrate with everything compiled
    times_compiled_all = []
    integrate_compiled = None
    for integration_domain in domains:
        t0 = time.perf_counter()
        if integrate_compiled is None:
            integrate_compiled = torch.jit.trace(
                lambda dom: integrator.integrate(example_integrand, dim, N, dom),
                (integration_domain,),
            )
        integrate_compiled(integration_domain)
        times_compiled_all.append(time.perf_counter() - t0)

    print(f"Uncompiled times: {times_uncompiled}")
    print(f"Partly compiled times: {times_compiled_parts}")
    print(f"All compiled times: {times_compiled_all}")
    speedups = [
        (1.0, tu / tcp, tu / tca)
        for tu, tcp, tca in zip(times_uncompiled, times_compiled_parts, times_compiled_all)
    ]
    print(f"Speedup factors: {speedups}")

This code shows two ways of compiling the integration.
In the first case, we use ``integrator.get_jit_compiled_integrate``,
which internally uses ``torch.jit.trace`` to compile performance-relevant code
parts except the integrand evaluation.
In the second case we directly compile ``integrator.integrate``.
The function created in the first case may be a bit slower,
but it works even if the integrand cannot be compiled and we can re-use it
with other integrand functions.
The compilations happen in the first iteration of the for loops and in the
following iterations the previously compiled functions are re-used.

With JAX and Tensorflow it is also possible to compile the integration.
In comparison to compilation with PyTorch,
we would need to use ``jax.jit`` or ``tf.function`` instead of
``torch.jit.trace`` to compile the whole integrate method.
``get_jit_compiled_integrate`` automatically uses the compilation function
which fits to the numerical backend.
There is a special case with JAX and MonteCarlo:
If a function which executes the integrate method is compiled with ``jax.jit``,
the random number generator's current PRNGKey value needs to be an input and
output of this function so that MonteCarlo generates different random numbers
in each integration.
torchquad's RNG class has methods to set and get this PRNGKey value.

The disadvantage of compilation is the additional time required to compile or
re-compile the code,
so if the integrate method is executed only a few times or certain arguments,
e.g. ``N``, change often, the program may be slower overall.


Reusing sample points
`````````````````````

With the MonteCarlo and composite Newton Cotes integrators it is possible to
execute the methods for sample point calculation, integrand evaluation and
result calculation separately.
This can be helpful to obtain a speedup in situations where integration happens
very often with the same ``integration_domain`` and ``N`` arguments.
However, separate sample point calculation has some disadvantages:

- The code is more complex.
- The memory required for the grid points is not released after each integration.
- With MonteCarlo the same sample points would be used for each integration, which corresponds to a fixed seed.

Here is an example where we integrate two functions with Boole and use the same
sample points for both functions:

.. code:: ipython3

    import torch
    from torchquad import Boole

    def integrand1(x):
        return torch.sin(x[:, 0]) + torch.exp(x[:, 1])

    def integrand2(x):
        return torch.prod(torch.cos(x), dim=1)

    # The integration domain, dimensionality and number of evaluations
    # For the calculate_grid method we need a Tensor and not a list.
    integration_domain = torch.Tensor([[0.0, 1.0], [-1.0, 1.0]])
    dim = 2
    N = 9409

    # Initialize the integrator
    integrator = Boole()
    # Calculate sample points and grid information for the result calculation
    grid_points, hs, n_per_dim = integrator.calculate_grid(N, integration_domain)

    # Integrate the first integrand with the sample points
    function_values, _ = integrator.evaluate_integrand(integrand1, grid_points)
    integral1 = integrator.calculate_result(function_values, dim, n_per_dim, hs)

    # Integrate the second integrand with the same sample points
    function_values, _ = integrator.evaluate_integrand(integrand2, grid_points)
    integral2 = integrator.calculate_result(function_values, dim, n_per_dim, hs)

    print(f"Quadrature results: {integral1}, {integral2}")

