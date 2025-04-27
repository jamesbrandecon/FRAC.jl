# FRACDemand.jl

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://jamesbrandecon.github.io/FRACDemand.jl)

This package estimates mixed logit demand models using the fast, "robust", and approximately correct (FRAC) approach developed by [Salanie and Wolak (2022)](https://economics.sas.upenn.edu/system/files/2022-04/Econometrics%2004112022.pdf). The current version of the code assumes that random coefficients on product characteristics are independent of each other and normally distributed. Each of these can be relaxed in the future, the former very easily.

## Installation
I hope to add things to the package and debug as needed, so install directly from Github: 
```jl
using Pkg
Pkg.add(url = "https://github.com/jamesbrandecon/FRACDemand.jl")
```

## Basic Usage
FRACDemand.jl requires a DataFrame as input which contains columns `shares`, `market_ids`, `product_ids`, `demand_instruments` (numbered from 0) and any relevant product characteristics. A minimal example, in which price is the only characteristic, would look like: 
```jl
julia> first(select(df, [:prices, :shares, :product_ids, :market_ids, :demand_instruments0, :demand_instruments1]))
DataFrameRow
 Row │ prices   shares    product_ids  market_ids  demand_instruments0  demand_instruments1 
     │ Float64  Float64   Int64        Int64       Float64              Float64             
─────┼──────────────────────────────────────────────────────────────────────────────────────
   1 │ 2.83905  0.426725            1           1             0.866898             0.751512
```

With this data, one can estimate FRAC and calculate all price elasticities using three simple functions:
```jl
problem = define_problem(data = df, 
            linear = ["prices", "x"], 
            nonlinear = ["prices", "x"],
            fixed_effects = ["market_ids"],
            se_type = "robust", 
            constrained = true);

estimate!(problem)

price_elasticities!(problem; monte_carlo_draws = 100);
```
It is important to note that, although `estimate!` is agnostic about the distribution of random coefficients, `price_elasticities` assumes that all random coefficients are normally distributed, as is standard. One could extend some inner functions called by `price_elasticities` pretty easily to allow for log-normal coefficients. It would just require an intermediate step that maps the estimated mean and standard deviations of the random coefficients to the appropriate parameters of a log-normal distribution.  


## Passing results to PyBLP
You should be able to save the results as a .csv file using the CSV package. The results can then be loaded in python and passed directly to PyBLP. The only caveat is that my understanding is that people don't like to use unicode characters in python as much as is done in Julia, so you may want to rename the results. An example workflow below:  
Save results in Julia:
```julia
frac_results = problem.estimated_parameters;
# .... rename dictionary entries as desired
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

## Notes
I have currently removed the feature of the package that allowed estimating many models at once. This is for two reasons: first, I'm hoping that it's essentially trivial, given the new package structure, to write a custom loop to do so yourself. Second, I found that it was making the code much less clean and more complicated to debug and develop. I can add it back with good reason. 