from .integration_test_functions import Polynomial, Exponential, Sinusoid

import numpy as np

# Here we define a bunch of function that shall be used for testing
TEST_FUNCTIONS_1D = [
    Polynomial(4, [2]),  # y = 2
    Polynomial(0, [1, 0]),  # y = x
    Polynomial(2 / 3, [0, 0, 2], domain=[0, 1]),  # y = 2x^2
    Polynomial(27.75, [3, -1, 2, -3], domain=[-2, 1]),  # y=-3x^3+2x^2-x+3
    # y=7x^4-3x^3+2x^2-x+3
    Polynomial(44648.0 / 15.0, [7, 3, -1, 2, -3], domain=[-4, 4]),
    # # y=-x^5+7x^4-3x^3+2x^2-x+3
    Polynomial(8939.0 / 60.0, [-1, 7, 3, -1, 2, -3], domain=[2, 3]),
    Exponential(np.exp(-2) + np.exp(1), domain=[-2, 1]),
    Exponential((np.exp(2) - 1.0) / np.exp(3), domain=[-3, -1]),
    Sinusoid(2 * np.sin(1) * np.sin(1), domain=[0, 2]),
]

# Compute convergence order

# Check if convergence order is met
