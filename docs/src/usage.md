# Usage Example

This page shows a minimal end‑to‑end workflow. For extended demos, see `examples/`.

## 1. Simulate synthetic demand

```julia
using FRAC, Random, DataFrames, Plots

df = sim_logit_vary_J(
  2, 4,                 # min and max products per market
  500, 1,               # T markets, B time periods
  [-1.0, 1.5],          # true β_prices, β_x
  [0.5 0.3; 0.3 0.8],   # true covariance matrix
  0.3,                  # market‐level shock variance
  with_market_FEs = true
)
```

## 2. Define and estimate the model

```julia
problem = define_problem(
  data          = df,
  linear        = ["prices", "x"],
  nonlinear     = ["prices", "x"],
  cov           = [("prices","x")],
  fixed_effects = ["market_ids"],
  se_type       = "bootstrap"
)

estimate!(problem)
```

## 3. Bootstrap standard errors
The `bootstrap!` function will run a bootstrap procedure to debias the original parameter estimates. The `approximate` determines whether the residuals are constructed from the FRAC approximation (`true`) or the full BLP contraction mapping (`false`). The `nboot` argument specifies the number of bootstrap draws.
```julia
    bootstrap!(problem; nboot = 200, approximate=true)
    @show problem.bootstrap_debiased_parameters
```

## 4. Compute and extract elasticities

```julia
price_elasticities!(problem; monte_carlo_draws=100)
own_df = own_elasticities(problem)
```

## 5. Visualize results

```julia
using Plots
histogram(
  own_df.own_elasticities,
  bins = 30,
  title = "Own‑Price Elasticities",
  xlabel = "Elasticity",
  ylabel = "Frequency"
)
```

## 5. Full example scripts

```bash
julia examples/simulate_estimate_distribution.jl
julia examples/simulate_compare_elasticities.jl
```
