from autoray import numpy as anp
from autoray import get_dtype_name


class RNG:
    """
    A random number generator helper class for multiple numerical backends

    Notes:
        - The seed argument may behave differently in different versions of a
          numerical backend and when using GPU instead of CPU

            - https://pytorch.org/docs/stable/notes/randomness.html
            - https://numpy.org/doc/stable/reference/random/generator.html#numpy.random.Generator
            - https://www.tensorflow.org/api_docs/python/tf/random/Generator
              Only the Philox RNG guarantees consistent behaviour in Tensorflow.
        - Often uniform random numbers are generated in [0, 1) instead of [0, 1].

            - numpy: random() is in [0, 1) and uniform() in [0, 1]
            - JAX: uniform() is in [0, 1)
            - torch: rand() is in [0, 1)
            - tensorflow: uniform() is in [0, 1)
    """

    def __init__(self, backend, seed=None, torch_save_state=False):
        """Initialize a RNG which can be seeded.

        An initialized RNG maintains a local PRNG state with JAX, Tensorflow and NumPy, and PyTorch if torch_save_state is True.

        Args:
            backend (string): Numerical backend, e.g. "torch".
            seed (int or None, optional): Random number generation seed. If set to None, the RNG is seeded randomly. Defaults to None.
            torch_save_state (Bool, optional): If True, maintain a separate RNG state for PyTorch. This argument can be helpful to avoid problems with integrand functions which set PyTorch's RNG seed. Unused unless backend is "torch". Defaults to False.

        Returns:
            An object whose "uniform" method generates uniform random numbers for the given backend
        """
        if backend == "numpy":
            import numpy as np

            self._rng = np.random.default_rng(seed)
            self.uniform = lambda size, dtype: self._rng.random(size=size, dtype=dtype)
        elif backend == "torch":
            self._set_torch_uniform(seed, torch_save_state)
        elif backend == "jax":
            from jax.random import PRNGKey, split, uniform

            if seed is None:
                # Generate a random seed; copied from autoray:
                # https://github.com/jcmgray/autoray/blob/35677037863d7d0d25ff025998d9fda75dce3b44/autoray/autoray.py#L737
                from random import SystemRandom

                seed = SystemRandom().randint(-(2**63), 2**63 - 1)
            self._jax_key = PRNGKey(seed)

            def uniform_func(size, dtype):
                self._jax_key, subkey = split(self._jax_key)
                return uniform(subkey, shape=size, dtype=dtype)

            self.uniform = uniform_func
        elif backend == "tensorflow":
            import tensorflow as tf

            if seed is None:
                self._rng = tf.random.Generator.from_non_deterministic_state()
            else:
                self._rng = tf.random.Generator.from_seed(seed)
            self.uniform = lambda size, dtype: self._rng.uniform(
                shape=size, dtype=dtype
            )
        else:
            if seed is not None:
                anp.random.seed(seed, like=backend)
            self._backend = backend
            self.uniform = lambda size, dtype: anp.random.uniform(
                size=size, dtype=get_dtype_name(dtype), like=self._backend
            )

    def _set_torch_uniform(self, seed, save_state):
        """Set self.uniform to generate random numbers with PyTorch

        Args:
            seed (int or None): Random number generation seed. If set to None, the RNG is seeded randomly.
            save_state (Bool): If True, maintain a separate RNG state.
        """
        import torch

        if save_state:
            # Set and restore the global RNG state before and after
            # generating random numbers

            if torch.cuda.is_initialized():
                # RNG state functions for the current CUDA device
                get_state = torch.cuda.get_rng_state
                set_state = torch.cuda.set_rng_state
            else:
                # RNG state functions for the Host
                get_state = torch.get_rng_state
                set_state = torch.set_rng_state

            previous_rng_state = get_state()
            if seed is None:
                torch.random.seed()
            else:
                torch.random.manual_seed(seed)
            self._rng_state = get_state()
            set_state(previous_rng_state)

            def uniform_func(size, dtype):
                # Swap the state
                previous_rng_state = get_state()
                set_state(self._rng_state)
                # Generate numbers
                random_values = torch.rand(size=size, dtype=dtype)
                # Swap the state back
                self._rng_state = get_state()
                set_state(previous_rng_state)
                return random_values

            self.uniform = uniform_func
        else:
            # Use the global RNG state for random number generation
            if seed is None:
                torch.random.seed()
            else:
                torch.random.manual_seed(seed)
            self.uniform = lambda size, dtype: torch.rand(size=size, dtype=dtype)

    def uniform(self, size, dtype):
        """Generate uniform random numbers in [0, 1) for the given numerical backend.
        This function is backend-specific; its definitions are in the constructor.

        Args:
            size (list): The shape of the generated numbers tensor
            dtype (backend dtype): The dtype for the numbers, e.g. torch.float32

        Returns:
            backend tensor: A tensor with random values for the given numerical backend
        """
        pass

    def jax_get_key(self):
        """
        Get the current PRNGKey.
        This function is needed for non-determinism when JIT-compiling with JAX.
        """
        return self._jax_key

    def jax_set_key(self, key):
        """
        Set the PRNGKey.
        This function is needed for non-determinism when JIT-compiling with JAX.
        """
        self._jax_key = key
