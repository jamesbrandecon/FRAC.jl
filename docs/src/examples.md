# Examples Gallery

Below is a quick overview of the Julia scripts in `examples/`:

- [example1.jl](github.com/jamesbrandecon/FRACDemand.jl/tree/main/examples/example1.jl)  
  A basic end‑to‑end workflow: simulate data, estimate FRAC (constrained vs unconstrained), compute elasticities, compare to logit.  
- [simulate_estimate_distribution.jl](github.com/jamesbrandecon/FRACDemand.jl/tree/main/examples/simulate_estimate_distribution.jl)  
  Runs multiple sims to show distribution of parameter estimates versus true values.  
- [simulate_compare_elasticities.jl](github.com/jamesbrandecon/FRACDemand.jl/tree/main/examples/simulate_compare_elasticities.jl)  
  Compares MSE and MAPE of estimated price elasticities with and without a covariance IV.  
- [testing_debias.jl](github.com/jamesbrandecon/FRACDemand.jl/tree/main/examples/testing_debias.jl)  
  Demonstrates bootstrap standard errors and debiasing for the unconstrained estimator.  