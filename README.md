# Strong Poisoning Attacks on Ridge Regression Models

This repository contains the implementation and results of an algorithm designed to generate strong poisoning attacks on ridge regression models. The method explicitly models and poisons categorical features by treating them as Special Ordered Sets of type 1 (SOS1). The project formulates this problem as a bilevel optimization problem and introduces a single-level reformulation based on the Karush-Kuhn-Tucker (KKT) conditions to improve computational efficiency.


## Overview

In this project, we propose an innovative approach to generating poisoning attacks for ridge regression models with both numerical and categorical features. The algorithm improves upon benchmarks in the literature by effectively modeling and poisoning categorical features, and it demonstrates significant improvements in the mean squared error (MSE) of the poisoned models across datasets.

Key highlights of this project include:

- Formulation of the poisoning attack as a bilevel optimization problem:
  - Nonconvex mixed-integer upper level.
  - Unconstrained convex quadratic lower level.
- Single-level reformulation using the KKT conditions.
- Improved solver performance through bounding the lower-level variables.
- A novel algorithm tailored to poisoning categorical features.
- Extensive numerical experiments demonstrating superior performance.


## Repository Structure

| File/Directory               | Description |
|------------------------------|-------------|
| `animated_plot_frames/`      | Frames for generating animated plots of poisoning results. |
| `categorical_weights_frames/`| Frames showing the impact of categorical weights on poisoning results. |
| `data/`                      | Contains datasets used for numerical experiments. |
| `numerical_weights_frames/`  | Frames showing the impact of numerical weights on poisoning results. |
| `percentage_ones_frames/`    | Frames visualizing the proportion of poisoned categorical features. |
| `poster/`                    | Presentation materials and visualizations. |
| `programs/`                  | Core implementations of the poisoning algorithms and benchmarks. |
| `results/`                   | Numerical results and analysis of the poisoning experiments. |
| `.gitignore`                 | Specifies files and directories to ignore in version control. |
| `CHANGELOG.txt`              | Change log tracking updates and fixes. |
| `animated_plot.html`         | HTML file showing animated poisoning results. |
| `categorical_weights.html`   | Visualization of the effects of categorical weights. |
| `numerical_weights.html`     | Visualization of the effects of numerical weights. |
| `percentage_ones.html`       | Visualization of the proportion of categorical poisoning. |
| `pyproject.toml`             | Python project configuration. |
| `setup_python_env.sh`        | Script for setting up the Python environment. |

### `programs/` Directory

| File/Directory               | Description |
|------------------------------|-------------|
| `benchmark/manip-ml-master` | Benchmark poisoning results for comparison. |
| `minlp/`                    | Mixed-Integer Nonlinear Programming implementations for poisoning attacks. |
| `polynomial_optimization/`  | Polynomial optimization approaches for PA relaxations. |

### `minlp/` Directory

| File/Directory               | Description |
|------------------------------|-------------|
| `attacks/`                   | CSV files containing datasets and results for various attack scenarios. |
| `old_implementation/`        | Previous versions of the poisoning algorithms with added regularization. |
| `results/`                   | Results of specific experiments, such as always flipping attacks. |
| `unused_algorithms/`         | Unused or deprecated algorithm implementations. |
| `unused_scripts/`            | Scripts for experimental or testing purposes. |
| `analyse_results.py`         | Script for analyzing results, including subset size evaluation. |
| `bounding_procedure.py`      | Implementation of the bounding procedure to enhance solver efficiency. |
| `computational_experiments.py` | Main script for running computational experiments on bilevel problems. |
| `config.yml`                 | Configuration file for experiments, including regularization settings. |
| `flipping_attack.py`         | Implementation of poisoning attacks targeting categorical features. |
| `instance_data_class.py`     | Data class for managing instances used in experiments. |
| `numerical_attack.py`        | Implementation of poisoning attacks for numerical features. |
| `plots.py`                   | Script for generating plots from experiment results. |
| `pyomo_model.py`             | Pyomo-based implementation of the bilevel optimization problem. |
| `random_projections.py`      | Random projection methods for dimensionality reduction. |
| `regularization_parameter.py` | Methods and analysis related to regularization in poisoning attacks. |
| `results_to_latex.py`        | Converts experimental results into LaTeX tables. |
| `ridge_regression.py`        | Ridge regression model implementation and integration with poisoning algorithms. |
| `testing.py`                 | Scripts for testing and debugging implementations. |

### `polynomial_optimization/` Directory

| File/Directory               | Description |
|------------------------------|-------------|
| `general_model.jl`           | Generalized model implementation for polynomial optimization problems. |
| `just_numerical_example.jl`  | Example implementation focusing on numerical data. |
| `nonlinear.jl`               | Nonlinear optimization problem implementation. |
| `polynomial.jl`              | Polynomial optimization model including the stable set problem. |
| `test.jl`                    | Script for building semi-algebraic sets iteratively. |
| `upload_data_test.jl`        | Script for uploading and testing data integration. |
| `with_categorical_example.jl`| Example implementation that includes categorical data in polynomial optimization. |


## Methodology

1. **Problem Formulation**:
   - The bilevel optimization problem is structured with a nonconvex mixed-integer upper level and a convex quadratic lower level.
   - Categorical features are modeled as SOS1 constraints to reflect their discrete nature.

2. **Single-Level Reformulation**:
   - Using the KKT conditions of the lower level, the bilevel problem is reformulated into a single-level optimization problem.
   - Bounds are derived for the lower-level variables to enhance computational efficiency.

3. **Algorithm Development**:
   - A novel poisoning algorithm explicitly targets categorical features.
   - Extensive numerical experiments validate the effectiveness of the approach.
     

## Installation and Usage

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook
- Required libraries (install using `pyproject.toml` or a `requirements.txt` file)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Set up the Python environment:
   ```bash
   bash setup_python_env.sh
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the experiments in the `programs/` directory:
   ```bash
   python programs/minlp/run_experiment.py
   ```

---

## Acknowledgments

This project was developed as part of research into robust machine learning and adversarial attack strategies. Special thanks to the open-source community and contributors for their tools and support.

