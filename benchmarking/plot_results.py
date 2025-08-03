#!/usr/bin/env python3
"""
Plotting module for torchquad benchmark results.

This module creates the enhanced plots addressing all identified issues:
1. Convergence plots with challenging functions and complete scipy coverage
2. Runtime vs error plots with all methods visible
3. Scaling analysis with error bars (when scaling data available)
4. Vectorized speedup plots with log-scale axes
"""

import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Dict


class ResultsPlotter:
    """Create comprehensive plots from benchmark results."""

    def __init__(self, results_dir: str = "resources"):
        self.results_dir = Path(results_dir)

        # Enhanced color palette and markers
        self.colors = {
            "simpson": "#0066CC",
            "gauss_legendre": "#00AA00",
            "monte_carlo": "#FF3333",
            "vegas": "#FF8C00",
            "scipy_nquad": "#000000",
            "scipy_trapz": "#808080",
            "scipy_simps": "#FF1493",
        }

        self.markers = {
            "simpson": "o",
            "gauss_legendre": "s",
            "monte_carlo": "D",
            "vegas": "^",
            "scipy_nquad": "X",
            "scipy_trapz": "+",
            "scipy_simps": "*",
        }

    def load_results(self, filename: str) -> Dict:
        """Load results from JSON file."""
        try:
            with open(self.results_dir / filename, "r") as f:
                results = json.load(f)
            print(f"Loaded results from {self.results_dir / filename}")
            return results
        except FileNotFoundError:
            print(f"Results file not found: {filename}")
            return {}
        except Exception as e:
            print(f"Error loading results: {e}")
            return {}

    def create_convergence_plots(self, convergence_results: Dict, device_info: str = "Unknown GPU"):
        """Create enhanced convergence analysis plots."""
        print("Creating convergence plots...")

        # Determine subplot layout based on available dimensions
        available_dims = [key for key in convergence_results.keys() if key.endswith("d")]
        n_plots = len(available_dims)

        if n_plots == 0:
            print("No convergence results to plot")
            return

        # Create subplot layout
        if n_plots == 1:
            fig, axes = plt.subplots(1, 1, figsize=(12, 8))
            axes = [axes]
        elif n_plots == 2:
            fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        elif n_plots <= 4:
            fig, axes = plt.subplots(2, 2, figsize=(18, 14))
            axes = axes.flatten()
        else:
            # More than 4 dimensions - use larger grid
            n_rows = (n_plots + 2) // 3
            fig, axes = plt.subplots(n_rows, 3, figsize=(24, 6 * n_rows))
            axes = axes.flatten()

        plot_idx = 0
        for case_key in sorted(available_dims, key=lambda x: int(x[:-1])):  # Sort by dimension
            case_data = convergence_results[case_key]

            if plot_idx >= len(axes):
                break

            ax = axes[plot_idx]
            dim = case_data["dim"]
            func_name = case_data["function"]
            func_desc = case_data["description"]

            # Plot torchquad methods
            for method in ["simpson", "gauss_legendre", "monte_carlo", "vegas"]:
                if method in case_data and case_data[method]["errors"]:
                    n_pts = case_data[method]["n_points"]
                    errors = case_data[method]["errors"]

                    valid_data = [(n, e) for n, e in zip(n_pts, errors) if e > 0]

                    if valid_data:
                        valid_n, valid_errors = zip(*valid_data)

                        label = f"{method.replace('_', ' ').title()} (CUDA)"
                        ax.loglog(
                            valid_n,
                            valid_errors,
                            color=self.colors[method],
                            marker=self.markers[method],
                            linewidth=2.5,
                            markersize=8,
                            label=label,
                            alpha=0.85,
                        )

            # Plot scipy methods
            if "scipy" in case_data:
                scipy_data = case_data["scipy"]

                # Plot nquad
                if "nquad" in scipy_data and scipy_data["nquad"]["error"] > 0:
                    representative_n = 10**6 if dim <= 3 else 10**5 if dim <= 7 else 10**4
                    ax.loglog(
                        [representative_n],
                        [scipy_data["nquad"]["error"]],
                        color=self.colors["scipy_nquad"],
                        marker=self.markers["scipy_nquad"],
                        markersize=14,
                        label="SciPy nquad (CPU)",
                        linestyle="none",
                        alpha=0.9,
                    )

                # Plot trapz and simps (1D only)
                if dim == 1:
                    # Trapz
                    trapz_points = []
                    for key, data in scipy_data.items():
                        if key.startswith("trapz_") and data["error"] > 0:
                            trapz_points.append((data["n_points"], data["error"]))
                    if trapz_points:
                        trapz_n, trapz_errors = zip(*sorted(trapz_points))
                        ax.loglog(
                            trapz_n,
                            trapz_errors,
                            color=self.colors["scipy_trapz"],
                            marker=self.markers["scipy_trapz"],
                            linewidth=2.5,
                            markersize=10,
                            label="SciPy trapz (CPU)",
                            alpha=0.8,
                        )

                    # Simps
                    simps_points = []
                    for key, data in scipy_data.items():
                        if key.startswith("simps_") and data["error"] > 0:
                            simps_points.append((data["n_points"], data["error"]))
                    if simps_points:
                        simps_n, simps_errors = zip(*sorted(simps_points))
                        ax.loglog(
                            simps_n,
                            simps_errors,
                            color=self.colors["scipy_simps"],
                            marker=self.markers["scipy_simps"],
                            linewidth=2.5,
                            markersize=10,
                            label="SciPy simps (CPU)",
                            alpha=0.8,
                        )

            ax.set_xlabel("Number of Function Evaluations", fontsize=13)
            ax.set_ylabel("Absolute Error", fontsize=13)
            ax.set_title(f"{dim}D: {func_name} \n Function: {func_desc}", fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10, loc="best")

            # Enhanced x-axis ticks for better low-count visibility
            if dim == 1:
                ax.set_xticks([10, 100, 1000, 10000, 100000, 1000000])
            elif dim == 3:
                ax.set_xticks([10, 100, 1000, 10000, 100000, 1000000])
            else:
                ax.set_xticks([1000, 10000, 100000, 1000000, 10000000])

            plot_idx += 1

        # Remove unused subplots
        for i in range(plot_idx, len(axes)):
            fig.delaxes(axes[i])

        plt.suptitle(
            f"Enhanced Convergence Analysis - Challenging Functions \n "
            f"Hardware: {device_info}, Precision: float32",
            fontsize=15,
        )
        plt.tight_layout()
        plt.savefig(self.results_dir / "torchquad_convergence.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Convergence plot saved to {self.results_dir / 'torchquad_convergence.png'}")

    def create_runtime_vs_error_plots(
        self, convergence_results: Dict, device_info: str = "Unknown GPU"
    ):
        """Create enhanced runtime vs error plots."""
        print("Creating runtime vs error plots...")

        # Determine subplot layout based on available dimensions
        available_dims = [key for key in convergence_results.keys() if key.endswith("d")]
        n_plots = len(available_dims)

        if n_plots == 0:
            print("No convergence results to plot")
            return

        # Create subplot layout
        if n_plots == 1:
            fig, axes = plt.subplots(1, 1, figsize=(12, 8))
            axes = [axes]
        elif n_plots == 2:
            fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        elif n_plots <= 4:
            fig, axes = plt.subplots(2, 2, figsize=(18, 14))
            axes = axes.flatten()
        else:
            # More than 4 dimensions - use larger grid
            n_rows = (n_plots + 2) // 3
            fig, axes = plt.subplots(n_rows, 3, figsize=(24, 6 * n_rows))
            axes = axes.flatten()

        plot_idx = 0
        for case_key in sorted(available_dims, key=lambda x: int(x[:-1])):
            case_data = convergence_results[case_key]

            if plot_idx >= len(axes):
                break

            ax = axes[plot_idx]
            dim = case_data["dim"]
            func_name = case_data["function"]
            func_desc = case_data["description"]

            # Plot torchquad methods
            for method in ["simpson", "gauss_legendre", "monte_carlo", "vegas"]:
                if method in case_data and case_data[method]["errors"]:
                    times = case_data[method]["times"]
                    errors = case_data[method]["errors"]

                    valid_data = [(t, e) for t, e in zip(times, errors) if e > 0 and t > 0]

                    if valid_data:
                        valid_times, valid_errors = zip(*valid_data)

                        label = f"{method.replace('_', ' ').title()} (CUDA)"
                        ax.loglog(
                            valid_times,
                            valid_errors,
                            color=self.colors[method],
                            marker=self.markers[method],
                            linewidth=2.5,
                            markersize=8,
                            label=label,
                            alpha=0.85,
                        )

            # Plot scipy methods
            if "scipy" in case_data:
                scipy_data = case_data["scipy"]

                # nquad
                if "nquad" in scipy_data:
                    data = scipy_data["nquad"]
                    if data["error"] > 0 and data["time"] > 0:
                        ax.loglog(
                            [data["time"]],
                            [data["error"]],
                            color=self.colors["scipy_nquad"],
                            marker=self.markers["scipy_nquad"],
                            markersize=14,
                            label="SciPy nquad (CPU)",
                            linestyle="none",
                            alpha=0.9,
                        )

                # 1D methods
                if dim == 1:
                    # Trapz
                    trapz_points = []
                    for key, data in scipy_data.items():
                        if key.startswith("trapz_") and data["error"] > 0 and data["time"] > 0:
                            trapz_points.append((data["time"], data["error"]))
                    if trapz_points:
                        trapz_t, trapz_e = zip(*trapz_points)
                        ax.loglog(
                            trapz_t,
                            trapz_e,
                            color=self.colors["scipy_trapz"],
                            marker=self.markers["scipy_trapz"],
                            markersize=10,
                            linestyle="none",
                            label="SciPy trapz (CPU)",
                            alpha=0.8,
                        )

                    # Simps
                    simps_points = []
                    for key, data in scipy_data.items():
                        if key.startswith("simps_") and data["error"] > 0 and data["time"] > 0:
                            simps_points.append((data["time"], data["error"]))
                    if simps_points:
                        simps_t, simps_e = zip(*simps_points)
                        ax.loglog(
                            simps_t,
                            simps_e,
                            color=self.colors["scipy_simps"],
                            marker=self.markers["scipy_simps"],
                            markersize=10,
                            linestyle="none",
                            label="SciPy simps (CPU)",
                            alpha=0.8,
                        )

            ax.set_xlabel("Runtime (seconds)", fontsize=13)
            ax.set_ylabel("Absolute Error", fontsize=13)
            ax.set_title(
                f"{dim}D: {func_name} \n Function: {func_desc} \n (Lower-left is better)",
                fontsize=12,
            )
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10, loc="best")

            plot_idx += 1

        # Remove unused subplots
        for i in range(plot_idx, len(axes)):
            fig.delaxes(axes[i])

        plt.suptitle(
            f"Runtime vs Error Analysis - Challenging Functions \n " f"Hardware: {device_info}",
            fontsize=15,
        )
        plt.tight_layout()
        plt.savefig(
            self.results_dir / "torchquad_runtime_vs_error.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
        print(
            f"Runtime vs error plot saved to {self.results_dir / 'torchquad_runtime_vs_error.png'}"
        )

    def create_scaling_plots(self, scaling_results: Dict, device_info: str = "Unknown GPU"):
        """Create runtime/feval scaling analysis plots for 1D and 7D."""
        print("Creating scaling plots...")

        import numpy as np

        if not scaling_results or ("1d" not in scaling_results and "7d" not in scaling_results):
            print("No scaling results to plot")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

        # Colors for different methods
        method_colors = {
            "trapezoid": "#8B4513",
            "simpson": "#0066CC",
            "boole": "#4B0082",
            "gauss_legendre": "#00AA00",
            "monte_carlo": "#FF3333",
            "vegas": "#FF8C00",
        }

        # Markers for different methods
        method_markers = {
            "trapezoid": "v",
            "simpson": "o",
            "boole": "p",
            "gauss_legendre": "s",
            "monte_carlo": "D",
            "vegas": "^",
        }

        # Plot 1D results
        if "1d" in scaling_results:
            ax = ax1
            dim_data = scaling_results["1d"]

            for method_name, method_data in dim_data.items():
                if method_data.get("fevals") and method_data.get("times_per_eval_mean"):
                    # Convert to fevals/second and scale to millions
                    fevals_per_sec = [
                        1.0 / time_per_eval for time_per_eval in method_data["times_per_eval_mean"]
                    ]
                    fevals_per_sec_std = [
                        std / (time_per_eval**2)
                        for time_per_eval, std in zip(
                            method_data["times_per_eval_mean"], method_data["times_per_eval_std"]
                        )
                    ]

                    # Scale to millions for better readability
                    fevals_per_sec_millions = [fps / 1e6 for fps in fevals_per_sec]
                    fevals_per_sec_std_millions = [std / 1e6 for std in fevals_per_sec_std]

                    # Plot with error bars
                    ax.errorbar(
                        method_data["fevals"],
                        fevals_per_sec_millions,
                        yerr=fevals_per_sec_std_millions,
                        color=method_colors.get(method_name, "black"),
                        marker=method_markers.get(method_name, "o"),
                        linewidth=2.5,
                        markersize=8,
                        capsize=5,
                        capthick=2,
                        label=method_name.replace("_", " ").title(),
                        alpha=0.85,
                    )

            # Add linear scaling reference line
            fevals_range = np.logspace(4, 8, 100)
            reference_throughput = 100  # 100M fevals/sec reference
            ax.plot(
                fevals_range,
                [reference_throughput] * len(fevals_range),
                "k--",
                alpha=0.5,
                label="Linear scaling",
            )

            ax.set_xlabel("Number of Function Evaluations", fontsize=13)
            ax.set_ylabel("Function Evaluations per Second (Millions)", fontsize=13)
            ax.set_title("1D Scaling Analysis \n Function: f(x) = Σx²", fontsize=14)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10, loc="best")

        # Plot 7D results
        if "7d" in scaling_results:
            ax = ax2
            dim_data = scaling_results["7d"]

            for method_name, method_data in dim_data.items():
                if method_data.get("fevals") and method_data.get("times_per_eval_mean"):
                    # Convert to fevals/second and scale to millions
                    fevals_per_sec = [
                        1.0 / time_per_eval for time_per_eval in method_data["times_per_eval_mean"]
                    ]
                    fevals_per_sec_std = [
                        std / (time_per_eval**2)
                        for time_per_eval, std in zip(
                            method_data["times_per_eval_mean"], method_data["times_per_eval_std"]
                        )
                    ]

                    # Scale to millions for better readability
                    fevals_per_sec_millions = [fps / 1e6 for fps in fevals_per_sec]
                    fevals_per_sec_std_millions = [std / 1e6 for std in fevals_per_sec_std]

                    # Plot with error bars
                    ax.errorbar(
                        method_data["fevals"],
                        fevals_per_sec_millions,
                        yerr=fevals_per_sec_std_millions,
                        color=method_colors.get(method_name, "black"),
                        marker=method_markers.get(method_name, "o"),
                        linewidth=2.5,
                        markersize=8,
                        capsize=5,
                        capthick=2,
                        label=method_name.replace("_", " ").title(),
                        alpha=0.85,
                    )

            # Add linear scaling reference line
            fevals_range = np.logspace(4, 8, 100)
            reference_throughput = 10  # 10M fevals/sec reference (lower for 7D)
            ax.plot(
                fevals_range,
                [reference_throughput] * len(fevals_range),
                "k--",
                alpha=0.5,
                label="Linear scaling",
            )

            ax.set_xlabel("Number of Function Evaluations", fontsize=13)
            ax.set_ylabel("Function Evaluations per Second (Millions)", fontsize=13)
            ax.set_title("7D Scaling Analysis \n Function: f(x) = Σx²", fontsize=14)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10, loc="best")

        plt.suptitle(f"Runtime/Evaluation Scaling Analysis\nHardware: {device_info}", fontsize=16)
        plt.tight_layout()
        plt.savefig(
            self.results_dir / "torchquad_scaling_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
        print(f"Scaling plot saved to {self.results_dir / 'torchquad_scaling_analysis.png'}")

    def create_vectorized_plots(
        self,
        vectorized_results: Dict,
        device_info: str = "Unknown GPU",
        x_log_scale: bool = True,
        y_log_scale: bool = True,
    ):
        """Create vectorized speedup plots with configurable log-scale axes."""
        print("Creating vectorized plots...")

        if not vectorized_results.get("grid_sizes"):
            print("No vectorized results to plot")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

        # Execution time comparison
        import numpy as np

        x_pos = np.arange(len(vectorized_results["grid_sizes"]))
        width = 0.35

        _ = ax1.bar(
            x_pos - width / 2,
            vectorized_results["loop_times"],
            width,
            label="Loop-based",
            alpha=0.8,
            color="lightcoral",
        )
        _ = ax1.bar(
            x_pos + width / 2,
            vectorized_results["vectorized_times"],
            width,
            label="Vectorized",
            alpha=0.8,
            color="lightblue",
        )

        ax1.set_xlabel("Parameter Grid Size", fontsize=13)
        ax1.set_ylabel("Execution Time (seconds)", fontsize=13)
        ax1.set_title(
            "Vectorized vs Loop-based Integration \n Function: sqrt(cos(sin(a*x)))", fontsize=13
        )
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(vectorized_results["grid_sizes"])

        if y_log_scale:
            ax1.set_yscale("log")

        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # Speedup factors
        if x_log_scale and y_log_scale:
            ax2.loglog(
                vectorized_results["grid_sizes"],
                vectorized_results["speedups"],
                marker="o",
                linewidth=3,
                markersize=10,
                color="green",
                label="Speedup",
            )
        elif x_log_scale:
            ax2.semilogx(
                vectorized_results["grid_sizes"],
                vectorized_results["speedups"],
                marker="o",
                linewidth=3,
                markersize=10,
                color="green",
                label="Speedup",
            )
        elif y_log_scale:
            ax2.semilogy(
                vectorized_results["grid_sizes"],
                vectorized_results["speedups"],
                marker="o",
                linewidth=3,
                markersize=10,
                color="green",
                label="Speedup",
            )
        else:
            ax2.plot(
                vectorized_results["grid_sizes"],
                vectorized_results["speedups"],
                marker="o",
                linewidth=3,
                markersize=10,
                color="green",
                label="Speedup",
            )

        ax2.axhline(y=1.0, color="red", linestyle="--", alpha=0.7, label="No speedup")

        ax2.set_xlabel("Parameter Grid Size", fontsize=13)
        ax2.set_ylabel("Speedup Factor", fontsize=13)

        scale_info = (
            f"({'Log' if x_log_scale else 'Linear'}-{'Log' if y_log_scale else 'Linear'} Scale)"
        )
        ax2.set_title(f"Vectorized Integration Speedup \n {scale_info}", fontsize=13)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=11)

        plt.suptitle(
            f"Enhanced Vectorized Integrand Performance Analysis \n " f"Hardware: {device_info}",
            fontsize=15,
        )
        plt.tight_layout()
        plt.savefig(
            self.results_dir / "torchquad_vectorized_speedup.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
        print(f"Vectorized plot saved to {self.results_dir / 'torchquad_vectorized_speedup.png'}")

    def create_framework_comparison_plots(
        self, framework_results: Dict, device_info: str = "Unknown GPU"
    ):
        """Create framework comparison plots for 1D Monte Carlo and Simpson methods."""
        print("Creating framework comparison plots...")

        if not framework_results or "results" not in framework_results:
            print("No framework results to plot")
            return

        # Colors and markers for different backends
        backend_colors = {
            "torch_gpu": "#FF3333",
            "torch_cpu": "#FF9999",
            "tensorflow_gpu": "#FF8C00",
            "tensorflow_cpu": "#FFB84D",
            "numpy_cpu": "#0066CC",
            "jax_cpu": "#00AA00",
        }

        backend_markers = {
            "torch_gpu": "o",
            "torch_cpu": "s",
            "tensorflow_gpu": "D",
            "tensorflow_cpu": "^",
            "numpy_cpu": "v",
            "jax_cpu": "p",
        }

        backend_labels = {
            "torch_gpu": "PyTorch GPU",
            "torch_cpu": "PyTorch CPU",
            "tensorflow_gpu": "TensorFlow GPU",
            "tensorflow_cpu": "TensorFlow CPU",
            "numpy_cpu": "NumPy CPU",
            "jax_cpu": "JAX CPU",
        }

        methods = framework_results.get("methods", ["monte_carlo", "simpson"])
        n_methods = len(methods)

        if n_methods == 0:
            print("No methods to plot")
            return

        # Create subplots - 2 plots per method (convergence + runtime vs error)
        fig, axes = plt.subplots(2, n_methods, figsize=(8 * n_methods, 12))

        if n_methods == 1:
            axes = axes.reshape(-1, 1)

        for method_idx, method_name in enumerate(methods):
            method_results = framework_results["results"].get(method_name, {})

            if not method_results:
                continue

            # Convergence plot (top row)
            ax_conv = axes[0, method_idx]

            # Runtime vs Error plot (bottom row)
            ax_runtime = axes[1, method_idx]

            for backend_spec, backend_data in method_results.items():
                if not backend_data.get("n_points") or not backend_data.get("errors"):
                    continue

                n_points = backend_data["n_points"]
                errors = backend_data["errors"]
                times = backend_data.get("times", [])

                # Skip invalid data
                valid_conv_data = [(n, e) for n, e in zip(n_points, errors) if e > 0]
                valid_runtime_data = [(t, e) for t, e in zip(times, errors) if e > 0 and t > 0]

                if not valid_conv_data:
                    continue

                label = backend_labels.get(backend_spec, backend_spec)
                color = backend_colors.get(backend_spec, "black")
                marker = backend_markers.get(backend_spec, "o")

                # Convergence plot
                conv_n, conv_errors = zip(*valid_conv_data)
                ax_conv.loglog(
                    conv_n,
                    conv_errors,
                    color=color,
                    marker=marker,
                    linewidth=2.5,
                    markersize=8,
                    label=label,
                    alpha=0.85,
                )

                # Runtime vs Error plot
                if valid_runtime_data:
                    runtime_times, runtime_errors = zip(*valid_runtime_data)
                    ax_runtime.loglog(
                        runtime_times,
                        runtime_errors,
                        color=color,
                        marker=marker,
                        linewidth=2.5,
                        markersize=8,
                        label=label,
                        alpha=0.85,
                    )

            # Format convergence plot
            ax_conv.set_xlabel("Number of Function Evaluations", fontsize=12)
            ax_conv.set_ylabel("Absolute Error", fontsize=12)
            ax_conv.set_title(f"{method_name.replace('_', ' ').title()} - Convergence", fontsize=13)
            ax_conv.grid(True, alpha=0.3)
            ax_conv.legend(fontsize=10, loc="best")

            # Format runtime vs error plot
            ax_runtime.set_xlabel("Runtime (seconds)", fontsize=12)
            ax_runtime.set_ylabel("Absolute Error", fontsize=12)
            ax_runtime.set_title(
                f"{method_name.replace('_', ' ').title()} - Runtime vs Error", fontsize=13
            )
            ax_runtime.grid(True, alpha=0.3)
            ax_runtime.legend(fontsize=10, loc="best")

        plt.suptitle(
            f"Framework Comparison - 1D Integration \n"
            f"Function: {framework_results.get('function', 'Unknown')} \n"
            f"Hardware: {device_info}",
            fontsize=16,
        )
        plt.tight_layout()
        plt.savefig(
            self.results_dir / "torchquad_framework_comparison.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
        print(
            f"Framework comparison plot saved to {self.results_dir / 'torchquad_framework_comparison.png'}"
        )


def main():
    """Create plots from existing results."""
    plotter = ResultsPlotter()
    device_info = "RTX 4060 Ti 16GB, i5-13400F"  # Could be loaded from config

    # Load convergence results
    convergence_results = plotter.load_results("convergence_results.json")
    if convergence_results:
        plotter.create_convergence_plots(convergence_results, device_info)
        plotter.create_runtime_vs_error_plots(convergence_results, device_info)

    # Load scaling results
    scaling_results = plotter.load_results("scaling_results.json")
    if scaling_results:
        plotter.create_scaling_plots(scaling_results, device_info)

    # Load vectorized results
    vectorized_results = plotter.load_results("vectorized_results.json")
    if vectorized_results:
        # Default to log-log scale, but could be made configurable
        plotter.create_vectorized_plots(
            vectorized_results, device_info, x_log_scale=True, y_log_scale=True
        )

    # Load framework comparison results
    framework_results = plotter.load_results("framework_results.json")
    if framework_results:
        plotter.create_framework_comparison_plots(framework_results, device_info)

    if convergence_results or scaling_results or vectorized_results or framework_results:
        print("All available plots created successfully!")
    else:
        print("No results available to plot. Run benchmark first.")


if __name__ == "__main__":
    main()
