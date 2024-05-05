# FRAC.jl

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