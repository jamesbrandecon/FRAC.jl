# FRAC.jl
This package estimates mixed logit demand models using the fast, "robust", and approximately correct (FRAC) approach developed by Salanie and Wolak (2019). The current version of the code 

## Basic Usage
FRAC.jl requires a DataFrame as input which contains columns `shares`, `market_ids`, `product_ids`, `demand_instruments` (numbered from 0) and any relevant product characteristics. A minimal example, in which price is the only characteristic, would look like: 
```jl
julia> first(select(df, [:prices, :shares, :product_ids, :market_ids, :demand_instruments0, :demand_instruments1]))
DataFrameRow
 Row │ prices   shares    product_ids  market_ids  demand_instruments0  demand_instruments1 
     │ Float64  Float64   Int64        Int64       Float64              Float64             
─────┼──────────────────────────────────────────────────────────────────────────────────────
   1 │ 2.83905  0.426725            1           1             0.866898             0.751512
```

With this data, one can estimate FRAC and calculate own-price elasticities using 
```jl
results = estimateFRAC(data = df, linear= "prices", nonlinear = "prices", se_type = "robust")
own_price_elasticities = price_elasticities(frac_results = results, data = df, linear = "prices", nonlinear = "prices + x", which = "own") 
```
It is important to note that, currently, you have to restate some options in `price_elasticities` (i.e. `results` does not include information about the specification. 

## Additional options
I have added a few additional options in `estimateFRAC`: 
- `by_var`: Name (as a string) of a variable denoting different (e.g. geographic) regions in which FRAC should be estimated separately. 
- `constrained`: Boolean variable determining whether the to constrain (1) estimates of the mean preferences for price to be negative and (2) all estiamtes of the variance of random coefficients to be positive. 
- `fes`: String denoting up to two dimensions of fixed effects. E.g. to absorb `fe1` and `fe2`, set `fes = "fe1 + fe2"`.

## Helper functions
For situations in which one is estimating many `FRAC` models at once, two helpful functions (examples shown in `examples/example1.jl` are:
`plotFRACResults(df; frac_results = [], var = "prices", param = "mean", plot = "hist", by_var = "")`: This plots either a "hist" or a "scatter" plot of all model estimates. If `param = "mean"`, this returns estimates of the mean preferences for characteristic `var`. If `param` is set to anything else, it returns estiamtes of the variance of preferences for `var`.

`extractEstimates(df;frac_results = [], var = "prices", param = "mean", by_var = "")`: For more complicated plots or when the researcher wants to examine the results of `estimateFRAC`, this returns estimated coefficients directly as arrays. For now it only returns standard errors for the unconstrained case, though I may add GMM standard errors for the constrained cases soon. 


