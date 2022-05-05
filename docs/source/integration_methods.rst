Integration methods
======================================

This is the list of all available integration methods in *torchquad*.

We are continuously implementing new methods in our library.
For the code behind the integration methods, please see the `code page <https://torchquad.readthedocs.io/en/main/_modules/index.html>`_
or check out our full code and latest news at https://github.com/esa/torchquad.

.. contents::

Stochastic Methods
----------------------

Monte Carlo Integrator
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchquad.MonteCarlo
   :members: integrate
   :noindex:

VEGAS Enhanced
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchquad.VEGAS
   :members: integrate
   :noindex:

Deterministic Methods
----------------------

Boole's Rule
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchquad.Boole
   :members: integrate
   :noindex:


Simpson's Rule
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchquad.Simpson
   :members: integrate
   :noindex:


Trapezoid Rule
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchquad.Trapezoid
   :members: integrate
   :noindex:
