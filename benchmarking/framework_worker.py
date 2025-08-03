#!/usr/bin/env python3
"""
Worker script for individual backend framework testing.
This runs in isolation to avoid TensorFlow device configuration conflicts.
"""

import sys
import json
import time
import gc
import warnings

warnings.filterwarnings("ignore")


def setup_backend(backend_name: str, device: str):
    """Setup backend with appropriate device configuration."""
    if backend_name == "torch":
        import os

        if device == "cpu":
            # Force CPU by hiding GPU from PyTorch BEFORE importing
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        import torch
        from torchquad import set_up_backend, enable_cuda

        if device == "gpu" and torch.cuda.is_available():
            set_up_backend("torch", data_type="float32")
            enable_cuda(data_type="float32")
        else:
            set_up_backend("torch", data_type="float32")
            torch.set_default_dtype(torch.float32)
            # Double-check CPU usage for PyTorch
            if device == "cpu":
                torch.set_default_device("cpu")

    elif backend_name == "tensorflow":
        import os

        if device == "cpu":
            # Force CPU by hiding GPU from TensorFlow BEFORE importing
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"

        import tensorflow as tf
        from torchquad import set_up_backend

        # Set device policy before torchquad setup
        if device == "cpu":
            # Ensure CPU-only execution
            tf.config.experimental.set_visible_devices([], "GPU")

        set_up_backend("tensorflow", data_type="float32")

    elif backend_name == "jax":
        import os

        if device == "cpu":
            os.environ["JAX_PLATFORM_NAME"] = "cpu"

        from torchquad import set_up_backend

        set_up_backend("jax", data_type="float32")

    elif backend_name == "numpy":
        from torchquad import set_up_backend

        set_up_backend("numpy", data_type="float32")
    else:
        raise ValueError(f"Unknown backend: {backend_name}")


def get_test_function(backend_name: str):
    """Get backend-specific test function."""
    if backend_name == "torch":
        import torch
        import numpy as np

        def torch_func(x):
            step_func = torch.where(x[:, 0] > 0.7, 1.0, 0.0)
            oscillatory = torch.sin(30 * torch.pi * x[:, 0]) * torch.exp(-10 * (x[:, 0] - 0.3) ** 2)
            rapid_osc = 0.5 * torch.cos(50 * torch.pi * x[:, 0])
            return oscillatory + rapid_osc + step_func + 0.1

        return torch_func

    elif backend_name == "tensorflow":
        import tensorflow as tf
        import numpy as np

        def tf_func(x):
            step_func = tf.where(x[:, 0] > 0.7, 1.0, 0.0)
            oscillatory = tf.sin(30 * tf.constant(np.pi) * x[:, 0]) * tf.exp(
                -10 * (x[:, 0] - 0.3) ** 2
            )
            rapid_osc = 0.5 * tf.cos(50 * tf.constant(np.pi) * x[:, 0])
            return oscillatory + rapid_osc + step_func + 0.1

        return tf_func

    elif backend_name == "jax":
        import jax.numpy as jnp

        def jax_func(x):
            step_func = jnp.where(x[:, 0] > 0.7, 1.0, 0.0)
            oscillatory = jnp.sin(30 * jnp.pi * x[:, 0]) * jnp.exp(-10 * (x[:, 0] - 0.3) ** 2)
            rapid_osc = 0.5 * jnp.cos(50 * jnp.pi * x[:, 0])
            return oscillatory + rapid_osc + step_func + 0.1

        return jax_func

    elif backend_name == "numpy":
        import numpy as np

        def numpy_func(x):
            step_func = np.where(x[:, 0] > 0.7, 1.0, 0.0)
            oscillatory = np.sin(30 * np.pi * x[:, 0]) * np.exp(-10 * (x[:, 0] - 0.3) ** 2)
            rapid_osc = 0.5 * np.cos(50 * np.pi * x[:, 0])
            return oscillatory + rapid_osc + step_func + 0.1

        return numpy_func
    else:
        raise ValueError(f"Unknown backend: {backend_name}")


def create_integrator(method_name: str):
    """Create integrator for the method."""
    if method_name == "monte_carlo":
        from torchquad import MonteCarlo

        return MonteCarlo()
    elif method_name == "simpson":
        from torchquad import Simpson

        return Simpson()
    else:
        raise ValueError(f"Unknown method: {method_name}")


def run_backend_benchmark(
    backend_name: str,
    device: str,
    method_name: str,
    eval_points: list,
    reference: float,
    num_runs: int,
    warmup_runs: int,
):
    """Run benchmark for a specific backend-method combination."""
    try:
        # Setup backend
        setup_backend(backend_name, device)

        # Get test function and integrator
        test_func = get_test_function(backend_name)
        integrator = create_integrator(method_name)

        domain = [[0, 1]]
        results = {"n_points": [], "errors": [], "times": [], "times_mean": [], "times_std": []}

        for n_points in eval_points:
            run_times = []
            run_errors = []

            # Perform multiple runs
            for run in range(warmup_runs + num_runs):
                is_warmup = run < warmup_runs

                # Clear caches
                gc.collect()

                start_time = time.perf_counter()

                # Run integration
                if method_name == "monte_carlo":
                    result = integrator.integrate(
                        test_func, dim=1, N=n_points, integration_domain=domain, seed=42 + run
                    )
                else:
                    result = integrator.integrate(
                        test_func, dim=1, N=n_points, integration_domain=domain
                    )

                end_time = time.perf_counter()
                elapsed = end_time - start_time

                # Extract result value (backend-agnostic)
                if hasattr(result, "item"):
                    result_value = result.item()
                elif hasattr(result, "numpy"):
                    result_value = float(result.numpy())
                else:
                    result_value = float(result)

                error = abs(result_value - reference)
                error = max(error, 1e-16)

                # Only record non-warmup runs
                if not is_warmup:
                    run_times.append(elapsed)
                    run_errors.append(error)

            if run_times:
                import statistics

                mean_time = statistics.mean(run_times)
                std_time = statistics.stdev(run_times) if len(run_times) > 1 else 0
                mean_error = statistics.mean(run_errors)

                results["n_points"].append(n_points)
                results["errors"].append(mean_error)
                results["times"].append(mean_time)
                results["times_mean"].append(mean_time)
                results["times_std"].append(std_time)

        return {"success": True, "results": results}

    except Exception as e:
        return {"success": False, "error": str(e)}


def main():
    """Main worker function."""
    if len(sys.argv) != 2:
        print("Usage: framework_worker.py <json_config>")
        sys.exit(1)

    config = json.loads(sys.argv[1])

    result = run_backend_benchmark(
        config["backend_name"],
        config["device"],
        config["method_name"],
        config["eval_points"],
        config["reference"],
        config["num_runs"],
        config["warmup_runs"],
    )

    print(json.dumps(result))


if __name__ == "__main__":
    main()
