# Troubleshooting

## 1. “Constraints not yet implemented with correlated random coefficients”  
Make sure `cov` is empty or remove covariance pairs when using `constrained = true`.

## 2. Cluster‐robust SE error  
If `se_type = "cluster"`, you must supply `cluster_var = "<column_name>"`.

## 3. Out‐of‐memory on large Monte‐Carlo draws  
Use `common_draws = true` in `price_elasticities!` or reduce `monte_carlo_draws`.  

For everything else open an issue:  
https://github.com/jamesbrand/FRAC.jl/issues
