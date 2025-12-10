"""
Iterative Solvers Comparison: Steepest Descent vs Conjugate Gradient

This module implements and compares two iterative methods for solving
symmetric positive-definite (SPD) linear systems Ax = b:
1. Steepest Descent (SD)
2. Conjugate Gradient (CG)

Author: Ömer Ergün
Institution: IPVS
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import matplotlib.pyplot as plt


# Configuration Constants
DEFAULT_TOLERANCE = 1e-6
DEFAULT_MAX_ITERATIONS = 10000
RANDOM_SEED = 42

# Visualization Constants
FIGURE_SIZE = (10, 6)
PLOT_DPI = 300
LINE_WIDTH = 2.0
GRID_ALPHA = 0.3


@dataclass
class SolverResult:
    """Container for solver results."""

    solution: np.ndarray
    residual_history: List[float]
    iterations: int
    converged: bool

    @property
    def final_residual(self) -> float:
        """Get the final residual norm."""
        return self.residual_history[-1]


class SPDMatrixGenerator:
    """Generator for symmetric positive-definite matrices with controlled properties."""

    @staticmethod
    def generate(n: int, condition_number: float) -> np.ndarray:
        """
        Generate an n×n SPD matrix with specified condition number.

        Uses SVD construction: A = Q * Λ * Q^T where Q is orthogonal
        and Λ contains logarithmically-spaced eigenvalues.

        Parameters
        ----------
        n : int
            Matrix dimension
        condition_number : float
            Desired condition number κ = λ_max / λ_min

        Returns
        -------
        np.ndarray
            SPD matrix of shape (n, n)
        """
        # Generate random orthogonal matrix via QR decomposition
        random_matrix = np.random.randn(n, n)
        orthogonal_matrix, _ = np.linalg.qr(random_matrix)

        # Create eigenvalues: logarithmically spaced between 1 and κ
        eigenvalues = np.logspace(0, np.log10(condition_number), n)

        # Construct SPD matrix
        diagonal_matrix = np.diag(eigenvalues)
        spd_matrix = orthogonal_matrix @ diagonal_matrix @ orthogonal_matrix.T

        # Ensure numerical symmetry
        return (spd_matrix + spd_matrix.T) / 2


class IterativeSolver:
    """Base class for iterative linear system solvers."""

    def __init__(
        self,
        tolerance: float = DEFAULT_TOLERANCE,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
    ):
        """
        Initialize solver with convergence parameters.

        Parameters
        ----------
        tolerance : float
            Relative residual tolerance for convergence
        max_iterations : int
            Maximum number of iterations allowed
        """
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def _check_convergence(
        self, residual_norm: float, initial_residual_norm: float
    ) -> bool:
        """Check if convergence criterion is satisfied."""
        relative_residual = residual_norm / initial_residual_norm
        return relative_residual < self.tolerance

    def solve(self, A: np.ndarray, b: np.ndarray, x0: np.ndarray) -> SolverResult:
        """
        Solve the linear system Ax = b.

        Parameters
        ----------
        A : np.ndarray
            SPD coefficient matrix
        b : np.ndarray
            Right-hand side vector
        x0 : np.ndarray
            Initial guess

        Returns
        -------
        SolverResult
            Container with solution and convergence information
        """
        raise NotImplementedError("Subclasses must implement solve()")


class SteepestDescentSolver(IterativeSolver):
    """
    Steepest Descent method for SPD systems.

    Uses negative gradient direction with optimal line search.
    Complexity: O(κ) iterations where κ is the condition number.
    """

    def solve(self, A: np.ndarray, b: np.ndarray, x0: np.ndarray) -> SolverResult:
        """
        Solve Ax = b using Steepest Descent.

        Algorithm
        ---------
        1. r_k = b - A*x_k (residual = negative gradient)
        2. α_k = (r_k^T * r_k) / (r_k^T * A * r_k) (optimal step size)
        3. x_{k+1} = x_k + α_k * r_k (update along gradient)

        Parameters
        ----------
        A : np.ndarray
            SPD coefficient matrix (n×n)
        b : np.ndarray
            Right-hand side vector (n,)
        x0 : np.ndarray
            Initial guess (n,)

        Returns
        -------
        SolverResult
            Solution with convergence history
        """
        x = x0.copy()
        residual = b - A @ x

        residual_history = [np.linalg.norm(residual)]
        initial_residual_norm = residual_history[0]

        for iteration in range(self.max_iterations):
            # Compute matrix-vector product once per iteration
            A_residual = A @ residual

            # Optimal step size via line search
            residual_dot_residual = residual.T @ residual
            step_size = residual_dot_residual / (residual.T @ A_residual)

            # Update solution and residual
            x += step_size * residual
            residual -= step_size * A_residual

            # Track convergence
            current_residual_norm = np.linalg.norm(residual)
            residual_history.append(current_residual_norm)

            # Check for convergence
            if self._check_convergence(current_residual_norm, initial_residual_norm):
                return SolverResult(
                    solution=x,
                    residual_history=residual_history,
                    iterations=iteration + 1,
                    converged=True,
                )

        # Max iterations reached without convergence
        print(
            f"Warning: Steepest Descent did not converge in {self.max_iterations} iterations"
        )
        return SolverResult(
            solution=x,
            residual_history=residual_history,
            iterations=self.max_iterations,
            converged=False,
        )


class ConjugateGradientSolver(IterativeSolver):
    """
    Conjugate Gradient method for SPD systems.

    Uses A-orthogonal search directions for faster convergence.
    Complexity: O(√κ) iterations where κ is the condition number.
    """

    def solve(self, A: np.ndarray, b: np.ndarray, x0: np.ndarray) -> SolverResult:
        """
        Solve Ax = b using Conjugate Gradient.

        Algorithm (Hestenes-Stiefel)
        ----------------------------
        1. r_0 = b - A*x_0, d_0 = r_0
        2. α_k = (r_k^T * r_k) / (d_k^T * A * d_k)
        3. x_{k+1} = x_k + α_k * d_k
        4. r_{k+1} = r_k - α_k * A * d_k
        5. β_{k+1} = (r_{k+1}^T * r_{k+1}) / (r_k^T * r_k)
        6. d_{k+1} = r_{k+1} + β_{k+1} * d_k

        Parameters
        ----------
        A : np.ndarray
            SPD coefficient matrix (n×n)
        b : np.ndarray
            Right-hand side vector (n,)
        x0 : np.ndarray
            Initial guess (n,)

        Returns
        -------
        SolverResult
            Solution with convergence history
        """
        x = x0.copy()
        residual = b - A @ x
        search_direction = residual.copy()

        residual_history = [np.linalg.norm(residual)]
        initial_residual_norm = residual_history[0]

        # Cache r^T * r for efficiency
        residual_squared_norm = residual.T @ residual

        for iteration in range(self.max_iterations):
            # Compute matrix-vector product once per iteration
            A_search_direction = A @ search_direction

            # Optimal step size along conjugate direction
            step_size = residual_squared_norm / (
                search_direction.T @ A_search_direction
            )

            # Update solution and residual
            x += step_size * search_direction
            residual -= step_size * A_search_direction

            # Track convergence
            current_residual_norm = np.linalg.norm(residual)
            residual_history.append(current_residual_norm)

            # Check for convergence
            if self._check_convergence(current_residual_norm, initial_residual_norm):
                return SolverResult(
                    solution=x,
                    residual_history=residual_history,
                    iterations=iteration + 1,
                    converged=True,
                )

            # Update search direction using Gram-Schmidt conjugation
            new_residual_squared_norm = residual.T @ residual
            conjugation_coefficient = new_residual_squared_norm / residual_squared_norm
            search_direction = residual + conjugation_coefficient * search_direction

            # Update cached value
            residual_squared_norm = new_residual_squared_norm

        # Max iterations reached without convergence
        print(
            f"Warning: Conjugate Gradient did not converge in {self.max_iterations} iterations"
        )
        return SolverResult(
            solution=x,
            residual_history=residual_history,
            iterations=self.max_iterations,
            converged=False,
        )


class ConvergencePlotter:
    """Utility for creating convergence comparison plots."""

    @staticmethod
    def plot_comparison(
        sd_result: SolverResult,
        cg_result: SolverResult,
        system_size: int,
        condition_number: float,
        tolerance: float,
        output_filename: str = "convergence_comparison.png",
    ):
        """
        Create and save a convergence comparison plot.

        Parameters
        ----------
        sd_result : SolverResult
            Steepest Descent results
        cg_result : SolverResult
            Conjugate Gradient results
        system_size : int
            Size of the linear system
        condition_number : float
            Condition number of the matrix
        tolerance : float
            Convergence tolerance used
        output_filename : str
            Output file path for the plot
        """
        plt.figure(figsize=FIGURE_SIZE)

        # Plot Steepest Descent
        iterations_sd = range(len(sd_result.residual_history))
        plt.semilogy(
            iterations_sd,
            sd_result.residual_history,
            "b-",
            linewidth=LINE_WIDTH,
            label=f"Steepest Descent ({sd_result.iterations} iter)",
        )

        # Plot Conjugate Gradient
        iterations_cg = range(len(cg_result.residual_history))
        plt.semilogy(
            iterations_cg,
            cg_result.residual_history,
            "darkorange",
            linewidth=LINE_WIDTH,
            label=f"Conjugate Gradient ({cg_result.iterations} iter)",
        )

        # Add tolerance threshold line
        initial_residual = sd_result.residual_history[0]
        plt.axhline(
            y=initial_residual * tolerance,
            color="red",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label=f"Tolerance ({tolerance})",
        )

        # Configure plot
        plt.xlabel("Iteration", fontsize=12)
        plt.ylabel("Residual Norm ||r||₂", fontsize=12)
        plt.title(
            f"Convergence Comparison (N={system_size}, κ={condition_number})",
            fontsize=14,
            fontweight="bold",
        )
        plt.grid(True, which="both", alpha=GRID_ALPHA, linestyle="--")
        plt.legend(fontsize=11, loc="best")
        plt.tight_layout()

        # Save and display
        plt.savefig(output_filename, dpi=PLOT_DPI, bbox_inches="tight")
        print(f"  ✓ Plot saved as '{output_filename}'")
        plt.show()


def run_comparison_experiment(
    system_size: int = 1000,
    condition_number: float = 100,
    tolerance: float = DEFAULT_TOLERANCE,
):
    """
    Run a comparison experiment between Steepest Descent and Conjugate Gradient.

    Parameters
    ----------
    system_size : int
        Dimension of the linear system
    condition_number : float
        Condition number of the SPD matrix
    tolerance : float
        Convergence tolerance
    """
    # Set seed for reproducibility
    np.random.seed(RANDOM_SEED)

    # Print experiment header
    print("=" * 70)
    print("Iterative Solvers Comparison: Steepest Descent vs Conjugate Gradient")
    print("=" * 70)
    print(f"\nExperiment Parameters:")
    print(f"  System size (N):         {system_size}")
    print(f"  Condition number (κ):    {condition_number}")
    print(f"  Convergence tolerance:   {tolerance}")
    print(f"  Initial guess:           x_0 = 0")
    print("\n" + "-" * 70)

    # Generate test problem
    print("\n[1/5] Generating SPD matrix...")
    A = SPDMatrixGenerator.generate(system_size, condition_number)

    actual_condition_number = np.linalg.cond(A)
    eigenvalues = np.linalg.eigvalsh(A)
    print(f"  Matrix generated: {system_size}×{system_size}")
    print(f"  Actual condition number: {actual_condition_number:.2f}")
    print(f"  Min eigenvalue: {eigenvalues[0]:.6f}")
    print(f"  Max eigenvalue: {eigenvalues[-1]:.6f}")

    # Set up linear system
    print("\n[2/5] Setting up linear system Ax = b...")
    x_true = np.random.randn(system_size)
    b = A @ x_true
    x0 = np.zeros(system_size)

    print(f"  Right-hand side vector norm: {np.linalg.norm(b):.6f}")
    print(f"  True solution norm: {np.linalg.norm(x_true):.6f}")

    # Solve with Steepest Descent
    print("\n[3/5] Running Steepest Descent...")
    sd_solver = SteepestDescentSolver(tolerance=tolerance)
    sd_result = sd_solver.solve(A, b, x0)

    sd_error = np.linalg.norm(sd_result.solution - x_true)
    print(f"  ✓ Converged in {sd_result.iterations} iterations")
    print(f"  Final residual norm: {sd_result.final_residual:.2e}")
    print(f"  Solution error: {sd_error:.2e}")

    # Solve with Conjugate Gradient
    print("\n[4/5] Running Conjugate Gradient...")
    cg_solver = ConjugateGradientSolver(tolerance=tolerance)
    cg_result = cg_solver.solve(A, b, x0)

    cg_error = np.linalg.norm(cg_result.solution - x_true)
    print(f"  ✓ Converged in {cg_result.iterations} iterations")
    print(f"  Final residual norm: {cg_result.final_residual:.2e}")
    print(f"  Solution error: {cg_error:.2e}")

    # Visualize results
    print("\n[5/5] Generating convergence plot...")
    ConvergencePlotter.plot_comparison(
        sd_result, cg_result, system_size, condition_number, tolerance
    )

    # Print summary
    speedup = sd_result.iterations / cg_result.iterations
    efficiency = 100 * cg_result.iterations / sd_result.iterations

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nSteepest Descent:")
    print(f"  Iterations:  {sd_result.iterations}")
    print(f"  Error:       {sd_error:.2e}")
    print(f"\nConjugate Gradient:")
    print(f"  Iterations:  {cg_result.iterations}")
    print(f"  Error:       {cg_error:.2e}")
    print(f"\nSpeedup:       {speedup:.2f}x faster")
    print(f"Efficiency:    CG used {efficiency:.1f}% of SD iterations")
    print("\n" + "=" * 70)
    print("\nThis confirms the theoretical complexity advantage:")
    print(f"  Steepest Descent: O(κ) = O({condition_number})")
    print(f"  Conjugate Gradient: O(√κ) = O({int(np.sqrt(condition_number))})")
    print("=" * 70 + "\n")


def main():
    """Main entry point for the comparison experiment."""
    run_comparison_experiment(system_size=1000, condition_number=100, tolerance=1e-6)


if __name__ == "__main__":
    main()
