# FRACDemand
[![Documentation](https://img.shields.io/badge/docs-latest-blue)](https://jamesbrand.github.io/FRACDemand.jl/)

> **Note:** This documentation is new and has been prepared with a lot of help from LLMs, so apologies for errors and omissions. If you find any, please open an issue on GitHub or submit a pull request.

FRACDemand is a Julia package for estimating random-coefficients demand models (BLP-style) using a fixed-rank approximation of contraction mappings. It provides:

- Flexible specification of utility function in mixed logit models
- Fixed effects and robust standard errors  
- Bootstrap debiasing and Monte Carlo elasticity computations  
- Price elasticities and predicted market shares
- Easy exporting to PyBLP

This package is meant to make the estimation of mixed logit models (with market level data) and the resulting price elasticities trivially easy. The package estimates an approximation of these models which was developed by [Salanie and Wolak (2022)](https://economics.sas.upenn.edu/system/files/2022-04/Econometrics%2004112022.pdf).

In many cases, the approximation offered by the package is sufficient to estimate price elasticities reasonably accurately. In other cases, results from FRAC.jl may be used as a starting point for other packages, such as [PyBLP](https://pyblp.readthedocs.io/en/stable/). The estimation results are saved in a dictionary for ease of reading and saving.   

```julia
julia> problem.estimated_parameters
Dict{Any, Any} with 4 entries:
  :β_x       => 1.00785
  :σ2_x      => -0.0500151
  :β_prices  => -1.0184
  :σ2_prices => 0.0839573
```

## Quickstart

- Installation: `] add FRACDemand`  
- API Reference: `docs/src/api.md`  
- Usage Examples: `docs/src/usage.md` and `docs/src/examples.md`

## Installation

```julia
# From the REPL:
] add FRACDemand
```

## Quick Start

```julia
using FRACDemand, DataFrames, CSV

# 1. Define your problem
problem = define_problem(
  data          = your_df,               
  linear        = ["prices", "x"],
  nonlinear     = ["prices", "x"],
  fixed_effects = ["market_ids"],
  se_type       = "bootstrap"
)

# 2. Estimate parameters
estimate!(problem)

# 3. Compute price elasticities
price_elasticities!(problem; monte_carlo_draws=100)
own_df = own_elasticities(problem)
```

## Features

- Unconstrained IV‑style FRAC approximation  
- GMM‑based constrained estimation  
- Support for random coefficients & covariance terms  
- Flexible standard errors: simple, robust, cluster, bootstrap  
- Built‑in Monte Carlo price elasticities  
- Easy export to CSV / PyBLP workflows  

## Next Steps

- 📖 [Usage Example](usage.md)  
- ⚙️ [API Reference](api.md)  
- 🔧 [Examples Gallery](examples.md)  
- ❗ [Troubleshooting](troubleshooting.md)  
- 📚 [Theory](theory.md)   

## Passing results to PyBLP
Results can be saved to CSV and used in Python for PyBLP. Note that Python may not handle Unicode dictionary keys; you may want to rename entries. Example:

Save results in Julia:
```julia
frac_results = problem.estimated_parameters
# ... rename keys if needed ...
CSV.write("frac_results.csv", frac_results)
```

Load in Python:
```python
import pyblp 
import pandas as pd
import numpy as np 

frac_results = pd.read_csv("frac_results.csv", header = False)
frac_sigma = np.zeros((2,2))
frac_sigma[0,0] = np.sqrt(frac_results["sigmaSquared_x"])
frac_sigma[1,1] = np.sqrt(frac_results["sigmaSquared_prices"])

X1 = pyblp.Formulation("1 + x + prices")
X2 = pyblp.Formulation("0 + x + prices")

problem = pyblp.Problem((X1, X2), product_data)

problem.solve(sigma = frac_sigma)
```