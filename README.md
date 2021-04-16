# FRAC.jl
The package is brand new! Consider it a beta version, and let me know if you encounter issues. 

This package estimates mixed logit demand models using the fast, "robust", and approximately correct (FRAC) approach developed by Salanie and Wolak (2019). The current version of the code 

## Installation
I may continue to add things to the package and there may be some bugs remaining, so for now install directly from Github: 
```jl
using Pkg
Pkg.add(url = "https://github.com/jamesbrandecon/FRAC.jl")
```

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
own_price_elasticities = price_elasticities(frac_results = results, data = df, linear = "prices", nonlinear = "prices", which = "own") 
```
It is important to note that, (1) currently, you have to restate some options in `price_elasticities` (i.e. `results` does not include information about the specification) and (2) Although `estimateFRAC` is agnostic about the distribution of random coefficients, `price_elasticities` assumes that all random coefficients are normally distributed, as is standard. 

## Additional options
I have added a few additional options in `estimateFRAC`: 
- `by_var`: Name (as a string) of a variable denoting different (e.g. geographic) regions in which FRAC should be estimated separately. 
- `product`: When calculating cross-price elasticities, currently you have to set `which = "cross"` and set `product` equal to the `product_ids` value of the product of interest. `price_elasticities` will then return all cross-price elasticities of each observation with respect to `product` (or zero in markets which do not contain `product`). 
- `constrained`: Boolean variable determining whether the to constrain (1) estimates of the mean preferences for price to be negative and (2) all estiamtes of the variance of random coefficients to be positive. 
- `fes`: String denoting up to two dimensions of fixed effects. E.g. to absorb `fe1` and `fe2`, set `fes = "fe1 + fe2"`.
- `cluster_var`: when `se_type = "cluster"`, you must also set `cluster_var` equal to the variable on which you want to cluster standard errors. 
   
## Helper functions
For situations in which one is estimating many `FRAC` models at once, two helpful functions (examples shown in `examples/example1.jl` are:

`plotFRACResults(df; frac_results = [], var = "prices", param = "mean", plot = "hist", by_var = "")`: This plots either a "hist" or a "scatter" plot of all model estimates. If `param = "mean"`, this returns estimates of the mean preferences for characteristic `var`. If `param` is set to anything else, it returns estiamtes of the variance of preferences for `var`.

`extractEstimates(df;frac_results = [], var = "prices", param = "mean", by_var = "")`: For more complicated plots or when the researcher wants to examine the results of `estimateFRAC`, this returns estimated coefficients directly as arrays. For now it only returns standard errors for the unconstrained case, though I may add GMM standard errors for the constrained cases soon. 

# Timing an Example
Here is a snippet of the example contained in `examples/example1.jl`
First, I use `FRAC.simulate_logit` to simulate demand for 40000 markets with differing numbers of goods per market 
```jl
T = 20000;
s,p,z,x = simulate_logit(2,T,[-0.4 0.1], [0.5 0.5], 0.3);
s2,p2,z2,x2 = simulate_logit(4,T,[-0.4 0.1], [0.5 0.5], 0.3);
```
This simulates mixed logit (BLP) demand data in which there are two product characteristics `prices` and `x` for which consumers have heterogeneous preferences. I then convert this data to a DataFrame `df` (see example file), define a column `by_example` which randomly divides markets into one of 100 separate regions, and add a variable `dummyFE` which is a meaningess fixed effect simply for demonstration purposes. To estimate 100 separate FRAC models, we can run  

```jl 
julia> @time results = estimateFRAC(data = df, linear= "prices + x", nonlinear = "prices + x",
           by_var = "by_example", fes = "product_ids + dummy_FE",
           se_type = "robust", constrained = true)
Progress: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| Time: 0:00:18
 39.046212 seconds (115.15 M allocations: 11.935 GiB, 5.30% gc time, 0.01% compilation time)
```
Note that here `constrained = true`. So, 100 constrained estimates with two dimenions of FEs takes less than a minute! Note that the progress bar tells how long the command actually spent in estimation, which is only 18 seconds. Unconstrained results are even faster: 
```jl
@time results = estimateFRAC(data = df, linear= "prices + x", nonlinear = "prices + x",
           by_var = "by_example", fes = "product_ids + dummy_FE",
           se_type = "robust", constrained = false)
Progress: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| Time: 0:00:07
  7.643048 seconds (14.19 M allocations: 1.035 GiB, 2.75% gc time, 90.89% compilation time)
```
## To-do
- Add wild bootstrap-based de-biasing from Salanie and Wolak (2019) 
- Allow for log-normal distributions for random coefficients 
- Add options for (random coefficient) nested logit 
