# Iterative Solvers for Systems of Linear Equations

This repository contains the implementation code for comparing **Steepest Descent** and **Conjugate Gradient** methods for solving symmetric positive-definite (SPD) linear systems.

## Overview

The code implements and compares two fundamental iterative algorithms:

1. **Steepest Descent (SD)**: Takes the most direct path downhill but suffers from slow, zig-zagging convergence
2. **Conjugate Gradient (CG)**: Uses mathematically orthogonal search directions to eliminate errors systematically

## Repository Structure

```text
.
├── src/
│   ├── iterative_solvers.py          # Basic comparison (SD vs CG)
│   └── complete_experiments.py       # Full experimental suite
├── iterative_solvers_paper.tex       # LaTeX paper source
├── iterative_solvers_paper.pdf       # Compiled paper
└── requirements.txt                   # Python dependencies
```

## Features

- **SPD Matrix Generation**: Creates random n×n symmetric positive-definite matrices with controllable condition numbers
- **Two Solver Implementations**: Both Steepest Descent and Conjugate Gradient methods
- **Convergence Analysis**: Tracks and visualizes residual norms throughout iterations
- **Performance Comparison**: Generates plots comparing convergence behavior
- **Production-Quality Code**: Clean OOP design with type hints and comprehensive documentation

## Requirements

- Python 3.7+
- NumPy
- Matplotlib

## Installation

Clone or navigate to this repository:

```bash
cd Iterative-Solvers-for-Systems-of-linear-equations
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Alternatively, install dependencies manually:

```bash
pip install numpy matplotlib
```

## Usage

### Basic Comparison

Run the basic comparison script:

```bash
python src/iterative_solvers.py
```

This will generate Figure 1 for the paper (`convergence_comparison.png`).

### Complete Experimental Suite

Run all experiments for the paper:

```bash
python src/complete_experiments.py
```

This script runs four experiments:

1. **Basic Convergence Comparison** (N=1000, κ=100): SD vs CG
2. **Condition Number Sensitivity** (N=500, κ=10 to 1000): Scaling analysis
3. **Eigenvalue Distribution Impact** (N=1000, κ=100): Uniform vs clustered
4. **Preconditioning Effect** (N=1000, κ=1000): Standard CG vs Preconditioned CG

The script will:

- Run all four experiments with detailed progress output
- Generate four publication-quality plots (300 DPI)
- Create a summary file with all numerical results
- Display all statistics in the terminal

## Expected Output

### Terminal Output

```text
======================================================================
Iterative Solvers Comparison: Steepest Descent vs Conjugate Gradient
======================================================================

Experiment Parameters:
  System size (N):         1000
  Condition number (κ):    100
  Convergence tolerance:   1e-06
  Initial guess:           x_0 = 0

----------------------------------------------------------------------

[1/5] Generating SPD matrix...
  Matrix generated: 1000×1000
  Actual condition number: 100.00
  Min eigenvalue: 1.000000
  Max eigenvalue: 100.000000

[2/5] Setting up linear system Ax = b...
  Right-hand side vector norm: 31.622777
  True solution norm: 1.000000

[3/5] Running Steepest Descent...
  ✓ Converged in 600+ iterations
  Final residual norm: < 1e-06
  Solution error: < 1e-06

[4/5] Running Conjugate Gradient...
  ✓ Converged in ~60 iterations
  ✓ Final residual norm: < 1e-06
  Solution error: < 1e-06

[5/5] Generating convergence plot...
  ✓ Plot saved as 'convergence_comparison.png'

======================================================================
SUMMARY
======================================================================

Steepest Descent:
  Iterations:  600+
  Error:       ~1e-07

Conjugate Gradient:
  Iterations:  ~60
  Error:       ~1e-07

Speedup:       ~10x faster
Efficiency:    CG used ~10% of SD iterations

======================================================================

This confirms the theoretical complexity advantage:
  Steepest Descent: O(κ) = O(100)
  Conjugate Gradient: O(√κ) = O(10)
======================================================================
```

### Generated Files

The complete experimental suite generates:

1. **convergence_comparison.png** (Figure 1): Basic comparison showing SD vs CG convergence
   - SD: 423 iterations, CG: 61 iterations (6.93x speedup)

2. **condition_number_sensitivity.png** (Figure 2): Two-panel figure showing scaling with condition number
   - Left: Iteration count vs κ for both methods
   - Right: CG convergence curves for different κ values
   - Confirms O(κ) for SD and O(√κ) for CG

3. **eigenvalue_distribution.png** (Figure 3): Impact of eigenvalue clustering
   - Uniform distribution: 61 iterations
   - Clustered distribution: 22 iterations (63.9% faster)

4. **preconditioning_effect.png** (Figure 4): Jacobi preconditioning comparison
   - Standard CG: 162 iterations
   - Preconditioned CG: 163 iterations
   - Shows that simple preconditioning may not always help

5. **experimental_results_summary.txt**: Detailed numerical results for all experiments

## Code Structure

### Main Components

#### Classes

**`SPDMatrixGenerator`**

- Static methods for generating SPD matrices with controlled condition numbers

**`SolverResult` (dataclass)**

- Type-safe container for solver results
- Properties: `solution`, `residual_history`, `iterations`, `converged`, `final_residual`

**`IterativeSolver` (base class)**

- Shared convergence checking logic
- Common interface for all solvers

**`SteepestDescentSolver`**

- Implements steepest descent algorithm
- Inherits from `IterativeSolver`

**`ConjugateGradientSolver`**

- Implements conjugate gradient algorithm
- Inherits from `IterativeSolver`

**`ConvergencePlotter`**

- Static methods for creating publication-quality plots

**`ExperimentRunner`** (in complete experiments)

- Orchestrates all four experiments
- Manages experiment execution and data collection

**`ResultsVisualizer`** (in complete experiments)

- Creates all figures for the paper
- Handles multi-panel layouts

**`ResultsWriter`** (in complete experiments)

- Saves numerical results to text files

## Customization

You can modify the experiment parameters in `main()`:

```python
N = 1000          # System size (number of variables)
kappa = 100       # Condition number
tol = 1e-6        # Convergence tolerance
```

To test different scenarios:

```python
# Example: Test on a smaller, better-conditioned system
N = 100
kappa = 10
```

## Theory

### Steepest Descent

- **Search Direction**: Residual r = b - Ax (negative gradient)
- **Step Size**: α = (r^T r) / (r^T A r)
- **Convergence**: O(κ) iterations
- **Problem**: Zig-zagging in narrow valleys

### Conjugate Gradient

- **Search Direction**: A-orthogonal (conjugate) directions
- **Property**: d_i^T A d_j = 0 for i ≠ j
- **Convergence**: O(√κ) iterations (superlinear)
- **Advantage**: No error re-introduction in previous directions

## References

This implementation follows the algorithms described in:

- Shewchuk, J. R. (1994). "An Introduction to the Conjugate Gradient Method Without the Agonizing Pain"
- Hestenes, M. R., & Stiefel, E. (1952). "Methods of conjugate gradients for solving linear systems"

## Author

Ömer Ergün
IPVS

## License

This code is provided for academic and educational purposes.
