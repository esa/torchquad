.. _tutorial:

Tutorial
===============

*torchquad* is a dedicated module for numerical integration in arbitrary dimensions.
This tutorial gives a more detailed look at its functionality and explores some performance considerations.

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
| Simpson’s    | Creates a quadratic interpolant between |br|    | Equal      |
| rule         | three neighbouring point                        |            |
+--------------+-------------------------------------------------+------------+
| Boole’s      | Creates a more complex interpolant between |br| | Equal      |
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

This notebook is a guide for new users to *torchquad* and is structured in
the following way:

-  Example integration in one dimension (1-D)
-  Example integration in ten dimensions (10-D)
-  Some accuracy / runtime comparisons with scipy

Feel free to test the code on your own computer as we go along.

Imports
-------

Now let’s get started! First, the general imports:

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
    
    from torchquad import enable_cuda # Necessary to enable GPU support
    from torchquad import Trapezoid, Simpson, Boole, MonteCarlo, VEGAS # The available integrators
    import torchquad

.. code:: ipython3

    enable_cuda() # Use this to enable GPU support 


.. parsed-literal::

    **Output:** Setting default tensor type to cuda.Float32 (CUDA is initialized).




One-dimensional integration
----------------------------

To make it easier to understand the methods used in this notebook, we will start with an
example in one dimension. If you are new to these methods or simply want a clearer picture, 
feel free to check out Patrick Walls’ 
`nice Python introduction <https://github.com/patrickwalls/mathematical-python/>`__ 
to the `Trapezoid rule <https://www.math.ubc.ca/~pwalls/math-python/integration/trapezoid-rule/>`__
and `Simpson’s rule <https://www.math.ubc.ca/~pwalls/math-python/integration/simpsons-rule/>`__
in one dimension.
Similarly, `Tirthajyoti Sarkar <https://github.com/tirthajyoti>`__ has made a nice visual explanation of 
`Monte Carlo integration in Python 
<https://towardsdatascience.com/monte-carlo-integration-in-python-a71a209d277e>`__.

Let ``f(x)`` be the function :math:`f(x) = e^{x} \cdot x^{2}`. Over the domain 
:math:`[0,2]`, the integral of ``f(x)`` is :math:`\int_{0}^{2} f(x) dx = 
\int_{0}^{2} e^x \cdot x^2 dx = 2(e^{2} - 1) = 12.7781121978613004544...`

Let’s declare the function and a simple function to print the absolute error, 
as well as remember the correct result.

.. code:: ipython3

    def f(x):
        return torch.exp(x) * torch.pow(x,2)
    
    def print_error(result,solution):
        print("Results:",result.item())
        print(f"Abs. Error: {(torch.abs(result - solution).item()):.8e}")
        print(f"Rel. Error: {(torch.abs((result - solution) / solution).item()):.8e}")
    
    solution = 2*(torch.exp(torch.tensor([2.]))-1)

**Note that we are using the torch versions to ensure that all variables
are and stay on the GPU.**

Let’s plot the function briefly.

.. code:: ipython3

    points = torch.linspace(0,2,100)
    plt.plot(points.cpu(),f(points).cpu()) # Note that for plotting we have to move the values to the CPU first
    plt.xlabel("$x$",fontsize=14)
    plt.ylabel("f($x$)",fontsize=14)



.. image:: torchquad_tutorial_figure.png


Let’s define the integration domain now and initialize the integrator - let’s start with the trapezoid rule.

.. code:: ipython3

    integration_domain = [[0, 2]] # Integration domain is always a list of lists to allow arbitrary dimensionality.
    tp = Trapezoid()  # Initialize a trapezoid solver

Now we are all set to compute the integral. Let’s try it with just 101 sample points for now.

.. code:: ipython3

    result = tp.integrate(f, dim=1, N=101, integration_domain=integration_domain)
    print_error(result,solution)


.. parsed-literal::

    **Output**: Results: 12.780082702636719
            Abs. Error: 1.97029114e-03
            Rel. Error: 1.54192661e-04
    

This is quite close already, as 1-D integrals are comparatively easy.
Let’s see what type of value we get for different integrators.

.. code:: ipython3

    simp = Simpson()
    result = simp.integrate(f, dim=1, N=101, integration_domain=integration_domain)
    print_error(result,solution)


.. parsed-literal::

    **Output:** Results: 12.778112411499023
            Abs. Error: 0.00000000e+00
            Rel. Error: 0.00000000e+00
    

.. code:: ipython3

    mc = MonteCarlo()
    result = mc.integrate(f, dim=1, N=101, integration_domain=integration_domain)
    print_error(result,solution)


.. parsed-literal::

    **Output:** Results: 13.32831859588623
            Abs. Error: 5.50206184e-01
            Rel. Error: 4.30584885e-02
    

.. code:: ipython3

    vegas = VEGAS()
    result = vegas.integrate(f,dim=1,N=101,integration_domain=integration_domain)
    print_error(result,solution)


.. parsed-literal::

    **Output:** Results: 21.83991813659668
            Abs. Error: 9.06180573e+00
            Rel. Error: 7.09166229e-01
    

Notably, Simpson’s method is already sufficient for a perfect solution here with 101 points. 
Monte Carlo methods do not perform so well; they are more suited to higher-dimensional integrals. 
VEGAS currently requires a larger number of samples to function correctly (as it performs several
iterations). 

Let’s step things up now and move to a 10-dimensional problem.

High-dimensional integration
----------------------------

Now, we will investigate the following 10-dimensional problem:

Let ``f_2`` be the function :math:`f_{2}(x) = \sum_{i=1}^{10} \sin(x_{i})`.

Over the domain :math:`[0,1]^{10}`, the integral of ``f_2`` is
:math:`\int_{0}^{1} \dotsc \int_{0}^{1} \sum_{i=1}^{10} \sin(x_{i}) = 20 \sin^{2}(1/2) = 4.59697694131860282599063392557 \dotsc`

Plotting this is tricky, so let’s directly move to the integrals.

.. code:: ipython3

    def f_2(x):
        return torch.sum(torch.sin(x),dim=1)
    
    solution = 20*(torch.sin(torch.tensor([0.5]))*torch.sin(torch.tensor([0.5])))

Let’s start with just 3 points per dimension, i.e., :math:`3^{10}=59,049` sample points. 

**Note**: *torchquad* currently only supports equal numbers of points per dimension. 
We are working on giving the user more flexibility on this point.

.. code:: ipython3

    integration_domain = [[0, 1]]*10 # Integration domain always is a list of lists to allow arbitrary dimensionality
    N = 3**10 

.. code:: ipython3

    tp = Trapezoid()  # Initialize a trapezoid solver
    result = tp.integrate(f_2, dim=10, N=N, integration_domain=integration_domain)
    print_error(result,solution)


.. parsed-literal::

    **Output:** Results: 4.500804901123047
            Abs. Error: 9.61723328e-02
            Rel. Error: 2.09207758e-02
    

.. code:: ipython3

    simp = Simpson()  # Initialize Simpson solver
    result = simp.integrate(f_2, dim=10, N=N, integration_domain=integration_domain)
    print_error(result,solution)


.. parsed-literal::

    **Output:** Results: 4.598623752593994
            Abs. Error: 1.64651871e-03
            Rel. Error: 3.58174206e-04
    

.. code:: ipython3

    mc = MonteCarlo()
    result = mc.integrate(f_2, dim=10, N=N, integration_domain=integration_domain, seed=42)
    print_error(result,solution)


.. parsed-literal::

    **Output:** Results: 4.598303318023682
            Abs. Error: 1.32608414e-03
            Rel. Error: 2.88468727e-04
    

.. code:: ipython3

    vegas = VEGAS()
    result = vegas.integrate(f_2,dim=10,N=N,integration_domain=integration_domain)
    print_error(result,solution)


.. parsed-literal::

    **Output:** Results: 4.598696708679199
            Abs. Error: 1.71947479e-03
            Rel. Error: 3.74044670e-04
    

Note that the Monte Carlo methods are much more competitive for
this case. The bad convergence properties of the trapezoid method are
visible while Simpson’s rule is still OK given the comparatively smooth
integrand.

If you have been repeating the examples from this tutorial on your own computer, you could also try 
increasing N to :math:`5^{10}=9,765,625`.
You can see the curse of dimensionality fully at play here, and 
some users might even experience running out of memory at this point.

Comparison with scipy
---------------------

Let’s explore how *torchquad*’s performance compares to scipy, the go-to
tool for numerical integration. A more detailed exploration of this
topic might be done as a side project at a later time. For simplicity,
we will stick to a 5-D version of the :math:`\sin(x)` of the previous
section. Let’s declare it with numpy and torch. Numpy arrays will
remain on the CPU and torch tensor on the GPU.

.. code:: ipython3

    dimension = 5
    integration_domain = [[0, 1]]*dimension
    ground_truth = 2 * dimension * np.sin(0.5)*np.sin(0.5)
    
    def f_3(x):
        return torch.sum(torch.sin(x),dim=1)
    
    def f_3_np(*x):
        return np.sum(np.sin(x))

Now let’s evaluate the integral using the scipy function ``nquad``.

.. code:: ipython3

    start = time.time()
    opts={"limit": 10, "epsabs" : 1, "epsrel" : 1}
    result, _,details = nquad(f_3_np, integration_domain, opts=opts, full_output=True) 
    end = time.time()
    print("Results:",result)
    print("Abs. Error:",np.abs(result - ground_truth))
    print(details)
    print(f"Took {(end-start)* 1000.0:.3f} ms")


.. parsed-literal::

    **Output:** Results: 2.2984884706593016
            Abs. Error: 0.0
            {'neval': 4084101}
            Took 33067.629 ms
    

Using scipy, we get the result in about 33 seconds on the authors’
machine (this might take shorter or longer on your machine). The integral was computed with
``nquad``, which on the inside uses the highly adaptive
`QUADPACK <https://en.wikipedia.org/wiki/QUADPACK>`__ algorithm.

In any event, *torchquad* can reach the same accuracy much, much quicker
by utilizing the GPU. 

.. code:: ipython3

    N = 37**dimension 
    simp = Simpson()  # Initialize Simpson solver
    start = time.time()
    result = simp.integrate(f_3, dim=dimension, N=N, integration_domain=integration_domain)
    end = time.time()
    print_error(result,ground_truth)
    print('neval=',N)
    print(f"Took {(end-start)* 1000.0:.3f} ms")


If you tried this yourself and ran out of CUDA memory, simply decrease :math:`N` 
(this will, however, lead to a loss of accuracy). 

Note that we use more evaluation points (:math:`37^{5}=69,343,957` for *torchquad* vs. :math:`4,084,101` 
for scipy), given the comparatively simple algorithm. 
Anyway, the decisive factor for this specific problem is runtime. A comparison with regard to
function evaluations is difficult, as ``nquad`` provides no support for a
fixed number of evaluations. This may follow in the future.

The results from using Simpson’s rule in *torchquad* is: 

.. parsed-literal::

    **Output:** Results: 2.2984883785247803
            Abs. Error: 0.00000000e+00
            Rel. Error: 0.00000000e+00
            neval= 69343957
            Took 162.147 ms
    

In our case, *torchquad*  with Simpson’s rule was more than 300 times faster than
``scipy.integrate.nquad``. We will add
more elaborate integration methods over time; however, this tutorial should
already showcase the advantages of numerical integration on the GPU.

Reasonably, one might prefer Monte Carlo integration methods for a 5-D
problem. We will add this in the future.

Computing gradients with respect to the integration domain
--------------------------------------------------------

*torchquad* allows fully automatic differentiation. In this tutorial, we will show how to extract the gradients with respect to the integration domain.
We selected Trapezoid and MonteCarlo methods to showcase that getting the gradients it is possible for both deterministic and stochastic methods.


.. code:: ipython3

    import torch
    from torchquad.integration.monte_carlo import MonteCarlo
    from torchquad.integration.trapezoid import Trapezoid
    from torchquad.utils.enable_cuda import enable_cuda
    from torchquad.utils.set_precision import set_precision

    def test_function(x):
        """V shaped test function."""
        return 2 * torch.abs(x)

    enable_cuda()
    set_precision("double")
    N = 99997 # Number of iterations
    torch.manual_seed(0)  # We have to seed torch to get reproducible results
    integrators = [MonteCarlo(), Trapezoid()]   # Define integrators

    for integrator in integrators:

        domain = torch.tensor([[-1.0, 1.0]]) #Integration domains
        domain.requires_grad = True # It enables the creation of a computational graph for gradient calculation.
        result = integrator.integrate(
            test_function, dim=1, N=N, integration_domain=domain
        ) # We calculate the 1 D integral by using the previously defined test-fuction

        result.backward() #Gradients computation

        print("Method:", integrator, "Gradients:", domain.grad)

The code above calculates the integral for a 1-D test_fuction `test_function()` in the [-1,1] domain and prints the gradients with respect to the integration domain.
The command `domain.requires_grad = True` enables the creation a computational graph, and it shall be called before calling the `integrate(...)` method.
Gradients computation is, then, performed calling `result.backward()`. 
The output of the code is as follows:

.. parsed-literal::

    **Output:** Method: <torchquad.integration.monte_carlo.MonteCarlo object at 0x7f724735b6a0> Gradients: tensor([[-1.9872,  2.0150]])
        Method: <torchquad.integration.trapezoid.Trapezoid object at 0x7f724735b6d0> Gradients: tensor([[-2.0000,  2.0000]])
