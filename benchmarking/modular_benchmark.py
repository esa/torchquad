#!/usr/bin/env python3
"""
Modular torchquad benchmark with configuration support and incremental execution.

This benchmark is designed to:
1. Load configuration from benchmarking_cfg.toml
2. Execute benchmarks incrementally by dimension
3. Provide detailed logging and timing information
4. Allow interruption and resumption of benchmarks
5. Support adjustable parameters for different hardware

Usage:
    python benchmarking/modular_benchmark.py [--config benchmarking_cfg.toml] [--dim 1,3,7,15]
"""

import numpy as np
import torch
import time
import warnings
import logging
import argparse
import subprocess
import sys
from pathlib import Path
from scipy.integrate import quad, nquad

try:
    from scipy.integrate import trapezoid as trapz, simpson as simps
except ImportError:
    from scipy.integrate import trapz, simps
from typing import Dict, List, Optional
import gc
import json

try:
    import toml
except ImportError:
    # Fallback to basic TOML parsing if toml module not available
    toml = None

# torchquad imports
from torchquad import Simpson, GaussLegendre, MonteCarlo, VEGAS, enable_cuda, Boole, Trapezoid
from torchquad.utils.set_precision import set_precision


class ModularBenchmark:
    """Modular benchmarking suite with configuration support."""

    def __init__(self, config_path: str = "benchmarking/benchmarking_cfg.toml"):
        self.config = self.load_config(config_path)
        self.save_path = Path(self.config["general"]["save_path"])
        self.save_path.mkdir(exist_ok=True)
        self.setup_logging()
        self.setup_backend()

        # Results storage
        self.results = {}
        self.timing_info = {}

    def load_config(self, config_path: str) -> dict:
        """Load configuration from TOML file."""
        if toml is None:
            print("TOML module not available. Using default configuration.")
            return self.get_default_config()

        try:
            with open(config_path, "r") as f:
                config = toml.load(f)
            print(f"Loaded configuration from {config_path}")
            return config
        except FileNotFoundError:
            print(f"Configuration file {config_path} not found. Using defaults.")
            return self.get_default_config()
        except Exception as e:
            print(f"Error loading config: {e}. Using defaults.")
            return self.get_default_config()

    def get_default_config(self) -> dict:
        """Return default configuration if TOML file not found."""
        return {
            "general": {
                "device_info": "Unknown GPU",
                "precision": "float32",
                "save_path": "resources",
                "log_level": "INFO",
            },
            "convergence": {
                "enable_1d": True,
                "enable_3d": True,
                "enable_7d": True,
                "enable_15d": True,
                "reference_points_1d": 1000000,
                "reference_points_3d": 2000000,
                "reference_points_7d": 1000000,
                "reference_points_15d": 500000,
                "points_1d": {
                    "simpson": [10, 50, 100, 500, 1000, 5000],
                    "gauss_legendre": [10, 50, 100, 500, 1000, 5000],
                    "monte_carlo": [100, 1000, 10000, 50000, 100000],
                    "vegas": [100, 1000, 10000, 50000],
                    "scipy_grids": [51, 251, 1001],
                },
                "points_3d": {
                    "simpson": [27, 125, 512, 1000, 4096],
                    "gauss_legendre": [27, 125, 512, 1000, 4096],
                    "monte_carlo": [1000, 10000, 50000, 100000, 500000],
                    "vegas": [1000, 10000, 50000, 100000],
                },
                "points_7d": {
                    "simpson": [128, 512, 2187],  # Limited for grid methods
                    "gauss_legendre": [128, 512, 2187],  # Limited for grid methods
                    "monte_carlo": [1000, 10000, 50000, 100000, 500000],
                    "vegas": [1000, 10000, 50000, 100000],
                },
                "points_15d": {
                    "simpson": [],  # Skip for 15D - too expensive
                    "gauss_legendre": [],  # Skip for 15D - too expensive
                    "monte_carlo": [10000, 50000, 100000, 500000],
                    "vegas": [10000, 50000, 100000],
                },
            },
            "scipy": {
                "nquad_limit_1d": 200,
                "nquad_limit_3d": 10,  # Reduced from 50 - was too slow
                "nquad_limit_7d": 2,  # Very limited for 7D
                "nquad_limit_15d": 2,  # Very limited for 15D
                "nquad_epsabs_15d": 1e-2,  # Looser tolerance
                "nquad_epsrel_15d": 1e-2,  # Looser tolerance
            },
            "timeouts": {
                "max_time_per_method": 120,  # 2 minutes per method
                "max_time_total": 900,  # 15 minutes total
            },
        }

    def setup_logging(self):
        """Configure logging based on config."""
        log_level = getattr(logging, self.config["general"]["log_level"])

        # Configure logging
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.save_path / "benchmark.log"),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def setup_backend(self):
        """Configure backend with GPU acceleration."""
        precision = self.config["general"]["precision"]
        try:
            set_precision(precision, backend="torch")
            enable_cuda(data_type=precision)
            self.device = "CUDA" if torch.cuda.is_available() else "CPU"

            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name()
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                self.logger.info(f"GPU: {gpu_name}, Memory: {gpu_memory:.1f} GB")

            self.logger.info(f"Using device: {self.device}, Precision: {precision}")
        except Exception as e:
            self.logger.warning(f"GPU setup failed, using CPU: {e}")
            set_precision(precision, backend="torch")
            self.device = "CPU"

    # Test functions - challenging functions that show convergence differences
    @staticmethod
    def challenging_1d(x):
        """1D: sin(30πx)exp(-10(x-0.3)²) + 0.5cos(50πx) + discontinuous step"""
        step_func = torch.where(x[:, 0] > 0.7, 1.0, 0.0)
        oscillatory = torch.sin(30 * torch.pi * x[:, 0]) * torch.exp(-10 * (x[:, 0] - 0.3) ** 2)
        rapid_osc = 0.5 * torch.cos(50 * torch.pi * x[:, 0])
        return oscillatory + rapid_osc + step_func + 0.1

    @staticmethod
    def challenging_1d_np(x):
        """NumPy version for scipy."""
        step_func = np.where(x > 0.7, 1.0, 0.0)
        oscillatory = np.sin(30 * np.pi * x) * np.exp(-10 * (x - 0.3) ** 2)
        rapid_osc = 0.5 * np.cos(50 * np.pi * x)
        return oscillatory + rapid_osc + step_func + 0.1

    @staticmethod
    def challenging_3d(x):
        """3D: Multiple sharp peaks + oscillatory + sharp ridge"""
        peak1 = torch.exp(-20 * torch.sum((x - 0.2) ** 2, dim=1))
        peak2 = torch.exp(-20 * torch.sum((x - 0.8) ** 2, dim=1))
        peak3 = torch.exp(-20 * torch.sum((x - 0.5) ** 2, dim=1))
        oscillatory = 0.2 * torch.prod(torch.sin(15 * torch.pi * x), dim=1)
        ridge = torch.exp(-100 * (x[:, 0] - x[:, 1]) ** 2) * torch.exp(
            -100 * (x[:, 1] - x[:, 2]) ** 2
        )
        return peak1 + peak2 + peak3 + oscillatory + 0.5 * ridge + 0.1

    @staticmethod
    def challenging_3d_np(*x):
        """NumPy version for scipy."""
        x_arr = np.array(x)
        peak1 = np.exp(-20 * np.sum((x_arr - 0.2) ** 2))
        peak2 = np.exp(-20 * np.sum((x_arr - 0.8) ** 2))
        peak3 = np.exp(-20 * np.sum((x_arr - 0.5) ** 2))
        oscillatory = 0.2 * np.prod(np.sin(15 * np.pi * x_arr))
        ridge = np.exp(-100 * (x_arr[0] - x_arr[1]) ** 2) * np.exp(
            -100 * (x_arr[1] - x_arr[2]) ** 2
        )
        return peak1 + peak2 + peak3 + oscillatory + 0.5 * ridge + 0.1

    @staticmethod
    def challenging_7d(x):
        """7D: Rastrigin-like with Gaussian envelope + sharp peak"""
        rastrigin = torch.sum(x**2 - 0.3 * torch.cos(8 * torch.pi * x), dim=1)
        gaussian = 2.0 * torch.exp(-3 * torch.sum((x - 0.5) ** 2, dim=1))
        sharp_peak = 5.0 * torch.exp(-50 * torch.sum((x - 0.3) ** 2, dim=1))
        return 0.1 * rastrigin + gaussian + sharp_peak + 0.2

    @staticmethod
    def challenging_7d_np(*x):
        """NumPy version for scipy."""
        x_arr = np.array(x)
        rastrigin = np.sum(x_arr**2 - 0.3 * np.cos(8 * np.pi * x_arr))
        gaussian = 2.0 * np.exp(-3 * np.sum((x_arr - 0.5) ** 2))
        sharp_peak = 5.0 * np.exp(-50 * np.sum((x_arr - 0.3) ** 2))
        return 0.1 * rastrigin + gaussian + sharp_peak + 0.2

    @staticmethod
    def challenging_15d(x):
        """15D: High-dimensional with multiple scales"""
        linear_sum = torch.sum(torch.sin(torch.pi * x) * x, dim=1)
        gaussian = torch.exp(-torch.sum((x - 0.5) ** 2, dim=1))
        quadratic = 0.1 * torch.sum(x**2, dim=1)
        return linear_sum + 2.0 * gaussian + quadratic + 0.5

    @staticmethod
    def challenging_15d_np(*x):
        """NumPy version for scipy."""
        x_arr = np.array(x)
        linear_sum = np.sum(np.sin(np.pi * x_arr) * x_arr)
        gaussian = np.exp(-np.sum((x_arr - 0.5) ** 2))
        quadratic = 0.1 * np.sum(x_arr**2)
        return linear_sum + 2.0 * gaussian + quadratic + 0.5

    def get_reference_value(self, func, dim: int, domain: List[List[float]]) -> float:
        """Get analytical reference value computed using SymPy."""
        func_name = f"{dim}D"
        self.logger.info(f"Using analytical reference value for {func_name}...")

        # Analytical reference values computed using SymPy
        # These have been validated against high-precision VEGAS/Boole computations
        analytical_references = {
            1: 4.0422850545e-01,  # Computed analytically using SymPy
            3: 2.6605308056e-01,  # Validated against Boole's rule
            7: 8.4401047230e-01,  # Validated against VEGAS
            15: 6.3714799881e00,  # Validated against VEGAS
        }

        if dim in analytical_references:
            reference = analytical_references[dim]
            self.logger.info(f"Analytical reference for {dim}D: {reference:.8e}")
            return reference
        else:
            # Fallback to numerical computation for unsupported dimensions
            self.logger.warning(
                f"No analytical reference for {dim}D, using numerical computation..."
            )
            try:
                if dim <= 3:
                    ref = Boole()
                    ref_points = self.config["convergence"].get(f"reference_points_{dim}d", 1000000)
                    ref_result = ref.integrate(
                        func, dim=dim, N=ref_points, integration_domain=domain
                    )
                    self.logger.info(
                        f"Boole's rule reference ({ref_points} pts): {ref_result.item():.8e}"
                    )
                else:
                    vegas_ref = VEGAS()
                    ref_points_key = f"reference_points_{dim}d"
                    ref_points = self.config["convergence"].get(ref_points_key, 1000000)

                    ref_result = vegas_ref.integrate(
                        func, dim=dim, N=ref_points, integration_domain=domain, seed=12345
                    )
                    self.logger.info(f"VEGAS reference ({ref_points} pts): {ref_result.item():.8e}")
                return ref_result.item()

            except Exception as e:
                self.logger.error(f"Reference calculation failed: {e}")
                return 1.0  # Fallback value

    def benchmark_method(
        self,
        method_name: str,
        integrator,
        func,
        dim: int,
        domain: List[List[float]],
        n_points: List[int],
        reference: float,
        timeout: float = 300,
    ) -> Dict:
        """Benchmark a single integration method."""
        self.logger.info(f"Benchmarking {method_name} for {dim}D...")

        method_start = time.perf_counter()
        errors = []
        times = []
        actual_n = []

        for i, n in enumerate(n_points):
            if time.perf_counter() - method_start > timeout:
                self.logger.warning(f"{method_name} timeout reached after {timeout}s")
                break

            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()

                start_time = time.perf_counter()

                # Method-specific integration calls
                if method_name == "vegas":
                    result = integrator.integrate(
                        func,
                        dim=dim,
                        N=n,
                        integration_domain=domain,
                        max_iterations=5,
                        use_warmup=True,
                        seed=42,
                    )
                elif method_name == "monte_carlo":
                    result = integrator.integrate(
                        func, dim=dim, N=n, integration_domain=domain, seed=42
                    )
                else:
                    result = integrator.integrate(func, dim=dim, N=n, integration_domain=domain)

                end_time = time.perf_counter()

                error = abs(result.item() - reference)
                error = max(error, 1e-16)  # Minimum plottable error

                errors.append(error)
                times.append(end_time - start_time)
                actual_n.append(n)
                torch.cuda.empty_cache()

                # Log progress
                if i % 2 == 0 or n >= 100000:
                    self.logger.info(
                        f"  N={n:>8}: error={error:.2e}, time={end_time - start_time:.4f}s"
                    )

            except Exception as e:
                self.logger.warning(f"  N={n}: Failed - {str(e)[:50]}...")
                if len(actual_n) > 3:  # Continue if we have some results
                    break

        total_time = time.perf_counter() - method_start
        self.logger.info(f"{method_name} completed in {total_time:.2f}s")
        return {"n_points": actual_n, "errors": errors, "times": times, "total_time": total_time}

    def benchmark_scipy_methods(self, func_np, dim: int, reference: float) -> Dict:
        """Benchmark scipy integration methods."""
        self.logger.info(f"Benchmarking scipy methods for {dim}D...")

        scipy_results = {}

        # scipy.integrate.nquad
        try:
            start_time = time.perf_counter()

            if dim == 1:
                limit = self.config["scipy"]["nquad_limit_1d"]
                scipy_result, _ = quad(func_np, 0, 1, limit=limit)
            elif dim <= 7:
                limit_key = f"nquad_limit_{dim}d"
                limit = self.config["scipy"].get(limit_key, 20)
                epsabs = self.config["scipy"]["nquad_epsabs"]
                epsrel = self.config["scipy"]["nquad_epsrel"]
                scipy_result, _ = nquad(
                    func_np,
                    [(0, 1)] * dim,
                    opts={"limit": limit, "epsabs": epsabs, "epsrel": epsrel},
                )
            elif dim == 15:
                # Attempt 15D with loose tolerances
                epsabs = self.config["scipy"]["nquad_epsabs_15d"]
                epsrel = self.config["scipy"]["nquad_epsrel_15d"]
                limit = self.config["scipy"]["nquad_limit_15d"]

                scipy_result, _ = nquad(
                    func_np,
                    [(0, 1)] * dim,
                    opts={"limit": limit, "epsabs": epsabs, "epsrel": epsrel},
                )
                self.logger.info("15D nquad succeeded with loose tolerances")

            end_time = time.perf_counter()
            scipy_error = max(abs(scipy_result - reference), 1e-16)
            scipy_results["nquad"] = {
                "result": scipy_result,
                "error": scipy_error,
                "time": end_time - start_time,
            }
            self.logger.info(
                f"SciPy nquad: error={scipy_error:.2e}, time={end_time - start_time:.4f}s"
            )

        except Exception as e:
            self.logger.warning(f"SciPy nquad failed: {e}")

        # scipy trapz and simps for 1D only
        if dim == 1:
            grid_sizes = self.config["convergence"]["points_1d"].get("scipy_grids", [51, 251, 1001])

            for grid_n in grid_sizes:
                x_grid = np.linspace(0, 1, grid_n)
                y_values = func_np(x_grid)

                # Trapezoid
                try:
                    start_time = time.perf_counter()
                    result = trapz(y_values, x_grid)
                    end_time = time.perf_counter()

                    error = max(abs(result - reference), 1e-16)
                    scipy_results[f"trapz_{grid_n}"] = {
                        "result": result,
                        "error": error,
                        "time": end_time - start_time,
                        "n_points": grid_n,
                    }

                except Exception as e:
                    self.logger.warning(f"SciPy trapz (N={grid_n}): Failed - {e}")

                # Simpson
                if simps is not None:
                    try:
                        start_time = time.perf_counter()
                        result = simps(y_values, x=x_grid)
                        end_time = time.perf_counter()

                        error = max(abs(result - reference), 1e-16)
                        scipy_results[f"simps_{grid_n}"] = {
                            "result": result,
                            "error": error,
                            "time": end_time - start_time,
                            "n_points": grid_n,
                        }

                    except Exception as e:
                        self.logger.warning(f"SciPy simps (N={grid_n}): Failed - {e}")

        return scipy_results

    def benchmark_convergence_dimension(self, dim: int) -> Optional[Dict]:
        """Benchmark convergence for a specific dimension."""
        if not self.config["convergence"].get(f"enable_{dim}d", False):
            self.logger.info(f"Skipping {dim}D convergence (disabled in config)")
            return None

        self.logger.info("=" * 60)
        self.logger.info(f"BENCHMARKING {dim}D CONVERGENCE")
        self.logger.info("=" * 60)

        # Get function and configuration
        func_map = {
            1: (
                self.challenging_1d,
                self.challenging_1d_np,
                "Discontinuous oscillatory",
                "sin(30πx)exp(-10(x-0.3)²) + 0.5cos(50πx) + step",
            ),
            3: (
                self.challenging_3d,
                self.challenging_3d_np,
                "Multi-peak with ridge",
                "Multiple peaks + sin(15π∏xi) + sharp ridge",
            ),
            7: (
                self.challenging_7d,
                self.challenging_7d_np,
                "Rastrigin-Gaussian hybrid",
                "Rastrigin oscillatory + Gaussian envelope + sharp peak",
            ),
            15: (
                self.challenging_15d,
                self.challenging_15d_np,
                "High-dimensional mixed",
                "sin(πx)x + Gaussian + quadratic",
            ),
        }

        if dim not in func_map:
            self.logger.error(f"No function defined for {dim}D")
            return None

        func, func_np, name, description = func_map[dim]
        domain = [[0, 1]] * dim

        # Calculate reference value
        reference = self.get_reference_value(func, dim, domain)

        # Get evaluation points from config
        points_config = self.config["convergence"].get(f"points_{dim}d", {})

        # Benchmark torchquad methods
        integrators = {
            "simpson": Simpson(),
            "gauss_legendre": GaussLegendre(),
            "monte_carlo": MonteCarlo(),
            "vegas": VEGAS(),
        }

        results = {"dim": dim, "function": name, "description": description, "reference": reference}

        timeout = self.config.get("timeouts", {}).get("max_time_per_method", 300)

        for method_name, integrator in integrators.items():
            n_points = points_config.get(method_name, [])
            if not n_points:
                self.logger.info(f"No points configured for {method_name}, skipping")
                continue

            results[method_name] = self.benchmark_method(
                method_name, integrator, func, dim, domain, n_points, reference, timeout
            )

        # Benchmark scipy methods
        if func_np and dim < 7:
            results["scipy"] = self.benchmark_scipy_methods(func_np, dim, reference)

        # Store timing information
        self.timing_info[f"{dim}d"] = {
            method: results[method].get("total_time", 0)
            for method in ["simpson", "gauss_legendre", "monte_carlo", "vegas"]
            if method in results
        }

        return results

    def run_convergence_benchmarks(self, dimensions: List[int] = None) -> Dict:
        """Run convergence benchmarks for specified dimensions."""
        if dimensions is None:
            dimensions = [1, 3, 7, 15]

        self.logger.info(f"Starting convergence benchmarks for dimensions: {dimensions}")
        total_start = time.perf_counter()

        convergence_results = {}

        for dim in dimensions:
            dim_start = time.perf_counter()

            result = self.benchmark_convergence_dimension(dim)
            if result is not None:
                convergence_results[f"{dim}d"] = result

            dim_elapsed = time.perf_counter() - dim_start
            self.logger.info(f"{dim}D benchmark completed in {dim_elapsed:.2f}s")

            # Check total timeout
            total_elapsed = time.perf_counter() - total_start
            max_total = self.config.get("timeouts", {}).get("max_time_total", 1800)
            if total_elapsed > max_total:
                self.logger.warning(f"Total timeout ({max_total}s) reached")
                break

        total_elapsed = time.perf_counter() - total_start
        self.logger.info(f"All convergence benchmarks completed in {total_elapsed:.2f}s")

        # Save intermediate results
        self.save_results(convergence_results, "convergence_results.json")

        return convergence_results

    def save_results(self, results: Dict, filename: str):
        """Save results to JSON file."""
        try:
            # Convert numpy/torch values to native Python types for JSON serialization
            def convert_for_json(obj):
                if isinstance(obj, (np.ndarray, torch.Tensor)):
                    return obj.tolist()
                elif isinstance(obj, (np.integer, np.floating)):
                    return obj.item()
                elif isinstance(obj, dict):
                    return {k: convert_for_json(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_for_json(item) for item in obj]
                else:
                    return obj

            results_json = convert_for_json(results)

            with open(self.save_path / filename, "w") as f:
                json.dump(results_json, f, indent=2)

            self.logger.info(f"Results saved to {self.save_path / filename}")

        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")

    def load_results(self, filename: str) -> Dict:
        """Load results from JSON file."""
        try:
            with open(self.save_path / filename, "r") as f:
                results = json.load(f)
            self.logger.info(f"Results loaded from {self.save_path / filename}")
            return results
        except FileNotFoundError:
            self.logger.info(f"No existing results file found: {filename}")
            return {}
        except Exception as e:
            self.logger.error(f"Failed to load results: {e}")
            return {}

    def benchmark_framework_comparison(self) -> Dict:
        """Framework comparison benchmark for 1D Monte Carlo and Simpson methods."""
        framework_config = self.config.get("framework_comparison", {})

        if not framework_config.get("enable", False):
            self.logger.info("Framework comparison disabled in config")
            return {}

        self.logger.info("=" * 60)
        self.logger.info("FRAMEWORK COMPARISON BENCHMARK")
        self.logger.info("=" * 60)

        # Configuration
        methods = framework_config.get("methods", ["monte_carlo", "simpson"])
        backends = framework_config.get("backends", ["torch_gpu", "torch_cpu"])
        num_runs = framework_config.get("num_runs", 3)
        warmup_runs = framework_config.get("warmup_runs", 1)

        # Use the 1D test function
        # func = self.challenging_1d
        # domain = [[0, 1]]
        reference = 4.0422850545e-01  # 1D analytical reference

        results = {
            "methods": methods,
            "backends": backends,
            "reference": reference,
            "function": "Discontinuous oscillatory 1D",
            "results": {},
        }

        for method_name in methods:
            if method_name not in results["results"]:
                results["results"][method_name] = {}

            # Get evaluation points for this method
            points_key = f"points_{method_name}"
            eval_points = framework_config.get(points_key, [1000, 10000, 100000])

            self.logger.info(f"Benchmarking {method_name} across frameworks...")

            for backend_spec in backends:
                self.logger.info(f"  Testing {method_name} with {backend_spec}...")

                try:
                    # Parse backend specification
                    if "_" in backend_spec:
                        backend_name, device = backend_spec.split("_", 1)
                    else:
                        backend_name, device = backend_spec, "cpu"

                    # Skip unavailable backends gracefully
                    if not self._is_backend_available(backend_name):
                        self.logger.warning(f"  Backend {backend_name} not available, skipping...")
                        continue

                    # Benchmark this method-backend combination using subprocess isolation
                    backend_results = self._benchmark_method_backend_subprocess(
                        backend_name,
                        device,
                        method_name,
                        eval_points,
                        reference,
                        num_runs,
                        warmup_runs,
                    )

                    if backend_results:
                        results["results"][method_name][backend_spec] = backend_results

                except Exception as e:
                    self.logger.error(
                        f"  Failed to benchmark {method_name} with {backend_spec}: {e}"
                    )
                    continue

        return results

    def _benchmark_method_backend_subprocess(
        self,
        backend_name: str,
        device: str,
        method_name: str,
        eval_points: list,
        reference: float,
        num_runs: int,
        warmup_runs: int,
    ):
        """Benchmark method-backend combination using subprocess isolation."""
        self.logger.info(f"    Using subprocess isolation for {backend_name}_{device}")

        # Prepare configuration for worker process
        config = {
            "backend_name": backend_name,
            "device": device,
            "method_name": method_name,
            "eval_points": eval_points,
            "reference": reference,
            "num_runs": num_runs,
            "warmup_runs": warmup_runs,
        }

        # Path to worker script
        worker_script = Path(__file__).parent / "framework_worker.py"

        try:
            # Run worker in subprocess
            result = subprocess.run(
                [sys.executable, str(worker_script), json.dumps(config)],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout per backend
                cwd=Path(__file__).parent.parent,  # Run from torchquad root
            )

            if result.returncode != 0:
                self.logger.error(f"    Worker process failed: {result.stderr}")
                return None

            # Parse result
            worker_result = json.loads(result.stdout.strip())

            if worker_result.get("success"):
                return worker_result["results"]
            else:
                self.logger.error(
                    f"    Worker failed: {worker_result.get('error', 'Unknown error')}"
                )
                return None

        except subprocess.TimeoutExpired:
            self.logger.error(f"    Worker process timed out for {backend_name}_{device}")
            return None
        except Exception as e:
            self.logger.error(f"    Subprocess error: {e}")
            return None

    def _is_backend_available(self, backend_name: str) -> bool:
        """Check if a backend is available."""
        try:
            if backend_name == "torch":
                torch  # noqa: F401
                return True
            elif backend_name == "tensorflow":
                import tensorflow as tf  # noqa: F401

                return True
            elif backend_name == "jax":
                import jax  # noqa: F401

                return True
            elif backend_name == "numpy":
                import numpy  # noqa: F401

                return True
            else:
                return False
        except ImportError:
            return False

    def benchmark_scaling_analysis(self) -> Dict:
        """Runtime/feval scaling analysis from 10K to 100M function evaluations."""
        self.logger.info("Runtime/feval scaling analysis...")

        def test_integrand(x):
            """Simple quadratic function for scaling tests."""
            return torch.sum(x**2, dim=1)

        # Load configuration
        scaling_config = self.config.get("scaling", {})
        feval_counts = scaling_config.get(
            "feval_counts",
            [10000, 50000, 100000, 500000, 1000000, 5000000, 10000000, 50000000, 100000000],
        )

        # Gauss-Legendre specific fevals
        gauss_legendre_fevals_1d = scaling_config.get("gauss_legendre_fevals_1d", feval_counts)
        gauss_legendre_fevals_7d = scaling_config.get("gauss_legendre_fevals_7d", feval_counts)

        # Max fevals from config
        max_fevals_grid_1d = scaling_config.get("max_fevals_grid_1d", 10000000)
        max_fevals_grid_7d = scaling_config.get("max_fevals_grid_7d", 100000)
        max_fevals_mc = scaling_config.get("max_fevals_mc", 100000000)

        # Methods to test
        methods = {
            "trapezoid": Trapezoid(),
            "simpson": Simpson(),
            "boole": Boole(),
            "gauss_legendre": GaussLegendre(),
            "monte_carlo": MonteCarlo(),
            "vegas": VEGAS(),
        }

        # Test in 1D and 7D
        dimensions = [1, 7]
        num_runs = scaling_config.get("num_runs", 3)
        warmup_runs = scaling_config.get("warmup_runs", 1)

        results = {}

        for dim in dimensions:
            self.logger.info(f"\nScaling analysis for {dim}D:")
            results[f"{dim}d"] = {}
            domain = [[0, 1]] * dim

            for method_name, integrator in methods.items():
                self.logger.info(f"  Method: {method_name}")
                method_results = {
                    "fevals": [],
                    "times_mean": [],
                    "times_std": [],
                    "times_per_eval_mean": [],
                    "times_per_eval_std": [],
                }

                # Determine which feval counts to use for this method
                if method_name == "gauss_legendre":
                    # Use special Gauss-Legendre fevals
                    if dim == 1:
                        method_feval_counts = gauss_legendre_fevals_1d
                        max_fevals = max_fevals_grid_1d
                    else:
                        method_feval_counts = gauss_legendre_fevals_7d
                        max_fevals = max_fevals_grid_7d
                elif method_name in ["trapezoid", "simpson", "boole"]:
                    # Other grid methods use standard fevals with limits
                    method_feval_counts = feval_counts
                    if dim == 1:
                        max_fevals = max_fevals_grid_1d
                    else:
                        max_fevals = max_fevals_grid_7d
                else:
                    # Monte Carlo methods
                    method_feval_counts = feval_counts
                    max_fevals = max_fevals_mc

                for fevals in method_feval_counts:
                    if fevals > max_fevals:
                        continue

                    self.logger.info(f"    N={fevals}: ")

                    run_times = []

                    # Run warmup + actual runs
                    total_runs = warmup_runs + num_runs
                    for run in range(total_runs):
                        is_warmup = run < warmup_runs
                        run_type = "warmup" if is_warmup else f"run {run - warmup_runs + 1}"

                        try:
                            # Clear GPU cache if available
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                gc.collect()

                            start_time = time.perf_counter()

                            if method_name == "vegas":
                                integrator.integrate(
                                    test_integrand,
                                    dim=dim,
                                    N=fevals,
                                    integration_domain=domain,
                                    max_iterations=5,
                                    use_warmup=True,
                                    seed=42 + run,
                                )
                            elif method_name == "monte_carlo":
                                integrator.integrate(
                                    test_integrand,
                                    dim=dim,
                                    N=fevals,
                                    integration_domain=domain,
                                    seed=42 + run,
                                )
                            else:
                                integrator.integrate(
                                    test_integrand, dim=dim, N=fevals, integration_domain=domain
                                )

                            elapsed = time.perf_counter() - start_time

                            # Only record times after warmup runs
                            if run >= warmup_runs:
                                run_times.append(elapsed)
                                self.logger.debug(f"      {run_type}: {elapsed:.4f}s")
                            else:
                                self.logger.debug(f"      {run_type}: {elapsed:.4f}s (discarded)")

                        except Exception as e:
                            self.logger.warning(f"      {run_type} failed: {e}")
                            continue

                    if run_times:
                        import statistics

                        mean_time = statistics.mean(run_times)
                        std_time = statistics.stdev(run_times) if len(run_times) > 1 else 0
                        mean_time_per_eval = mean_time / fevals
                        std_time_per_eval = std_time / fevals

                        method_results["fevals"].append(fevals)
                        method_results["times_mean"].append(mean_time)
                        method_results["times_std"].append(std_time)
                        method_results["times_per_eval_mean"].append(mean_time_per_eval)
                        method_results["times_per_eval_std"].append(std_time_per_eval)

                        self.logger.info(
                            f"time={mean_time:.4f}±{std_time:.4f}s, "
                            f"time/eval={mean_time_per_eval:.2e}±{std_time_per_eval:.2e}s"
                        )
                    else:
                        self.logger.warning("All runs failed")
                        break

                results[f"{dim}d"][method_name] = method_results

        return results

    def benchmark_vectorized_analysis(self) -> Dict:
        """Vectorized integrand test with configurable scaling."""
        self.logger.info("Vectorized integrands analysis...")

        integrator = Simpson()
        domain = [[0, 1]]
        N = self.config.get("vectorized", {}).get("integration_points", 1001)

        grid_sizes = self.config.get("vectorized", {}).get("grid_sizes", [5, 20, 50, 100, 200])
        num_runs = self.config.get("vectorized", {}).get("num_runs", 2)
        results = {"grid_sizes": [], "loop_times": [], "vectorized_times": [], "speedups": []}

        for grid_size in grid_sizes:
            self.logger.info(f"  Grid size {grid_size}:")

            params = torch.linspace(1, 5, grid_size)

            try:
                # Method 1: Loop-based (multiple runs for stability)
                loop_times = []
                for run in range(num_runs):
                    start_time = time.perf_counter()
                    loop_results = []
                    for param in params:

                        def single_integrand(x):
                            return torch.sqrt(torch.cos(torch.sin(param * x[:, 0])))

                        result = integrator.integrate(
                            single_integrand, dim=1, N=N, integration_domain=domain
                        )
                        loop_results.append(result.item())
                    loop_times.append(time.perf_counter() - start_time)

                loop_time = sum(loop_times) / len(loop_times)

                # Method 2: Vectorized (multiple runs for stability)
                vectorized_times = []
                for run in range(num_runs):
                    start_time = time.perf_counter()

                    def vectorized_integrand(x):
                        x_vals = x[:, 0]
                        return torch.sqrt(torch.cos(torch.sin(torch.outer(x_vals, params))))

                    integrator.integrate(
                        vectorized_integrand, dim=1, N=N, integration_domain=domain
                    )
                    vectorized_times.append(time.perf_counter() - start_time)

                vectorized_time = sum(vectorized_times) / len(vectorized_times)
                speedup = loop_time / vectorized_time

                results["grid_sizes"].append(grid_size)
                results["loop_times"].append(loop_time)
                results["vectorized_times"].append(vectorized_time)
                results["speedups"].append(speedup)

                self.logger.info(
                    f"    Loop: {loop_time:.4f}s, Vectorized: {vectorized_time:.4f}s, Speedup: {speedup:.2f}x"
                )

            except Exception as e:
                self.logger.warning(f"    Failed: {e}")
                break

        return results


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Run modular torchquad benchmarks")
    parser.add_argument(
        "--config", default="benchmarking/benchmarking_cfg.toml", help="Path to configuration file"
    )
    parser.add_argument(
        "--dimensions", default="1,3,7,15", help="Comma-separated list of dimensions to benchmark"
    )
    parser.add_argument(
        "--convergence-only", action="store_true", help="Run only convergence benchmarks"
    )
    parser.add_argument("--scaling-only", action="store_true", help="Run only scaling benchmarks")
    parser.add_argument(
        "--framework-only", action="store_true", help="Run only framework comparison"
    )

    args = parser.parse_args()

    # Parse dimensions
    try:
        dimensions = [int(d.strip()) for d in args.dimensions.split(",")]
    except ValueError:
        print("Invalid dimensions format. Use comma-separated integers like '1,3,7'")
        return

    # Initialize benchmark
    warnings.filterwarnings("ignore")
    benchmark = ModularBenchmark(args.config)

    scaling_results = None
    vectorized_results = None
    framework_results = None

    # Handle mutually exclusive flags
    exclusive_flags = [args.convergence_only, args.scaling_only, args.framework_only]
    if sum(exclusive_flags) > 1:
        print("Error: Cannot use multiple exclusive flags together")
        return

    if args.framework_only:
        # Run only framework comparison
        benchmark.logger.info("Running framework comparison only...")
        framework_results = benchmark.benchmark_framework_comparison()
        benchmark.save_results(framework_results, "framework_results.json")
    elif args.scaling_only:
        # Run only scaling benchmarks
        benchmark.logger.info("Running scaling analysis only...")
        scaling_results = benchmark.benchmark_scaling_analysis()
        benchmark.save_results(scaling_results, "scaling_results.json")
    elif args.convergence_only:
        # Run only convergence benchmarks
        benchmark.run_convergence_benchmarks(dimensions)
    else:
        # Run all benchmarks
        # Run convergence benchmarks
        benchmark.run_convergence_benchmarks(dimensions)

        # Run scaling benchmarks
        benchmark.logger.info("Running scaling analysis...")
        scaling_results = benchmark.benchmark_scaling_analysis()
        benchmark.save_results(scaling_results, "scaling_results.json")

        # Run vectorized benchmarks
        benchmark.logger.info("Running vectorized analysis...")
        vectorized_results = benchmark.benchmark_vectorized_analysis()
        benchmark.save_results(vectorized_results, "vectorized_results.json")

        # Run framework comparison
        benchmark.logger.info("Running framework comparison...")
        framework_results = benchmark.benchmark_framework_comparison()
        benchmark.save_results(framework_results, "framework_results.json")

    benchmark.logger.info("Benchmark session completed successfully!")


if __name__ == "__main__":
    main()
