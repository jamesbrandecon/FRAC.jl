# FRAC.jl API Reference

## Problem Definition

- `define_problem(...)`  
  Construct a `FRACProblem` with your data, covariates, and settings.
- `FRACProblem`  
  Holds data, formulas, and estimation results.

## Estimation & Debiasing

- `estimate!(problem)`  
  Run IV or GMM estimation (unconstrained or constrained).
- `bootstrap!(problem; nboot, approximate)`  
  Perform bootstrap standard errors and optional debiasing.
- `get_FEs!(problem)`  
  Extract estimated fixed effects.

## Elasticity Calculations

- `price_elasticities!(problem; monte_carlo_draws, ...)`  
  Compute and store all own‑ and cross‑price elasticities.
- `inner_price_elasticities(...)`  
  Helper for one product’s partial elasticities.
- `all_elasticities(...)`  
  Internal: generate individual elasticity draws.
- `get_elasticities(problem, [i, j])`  
  Extract elasticity of product i w.r.t. price j.
- `own_elasticities(problem)`  
  Return a `DataFrame` of own‑price elasticities.
- `correlate_draws(...)`  
  Inject covariance among random‐coefficient draws.

```@autodocs
Modules = [FRAC]
Order = [
  estimate!,
  bootstrap!,
  price_elasticities!,
  own_elasticities, 
  FRAC.get_elasticities,
  FRAC.define_problem,
  FRAC.HaltonDraws!,
  FRAC.bootstrap!,
  FRAC.get_FEs!,
  FRAC.inner_price_elasticities,
  FRAC.pre_absorb,
  FRAC.all_elasticities,
  FRAC.correlate_draws,
  FRAC.price_elasticities!,
  FRAC.HaltonSeq!
]
```