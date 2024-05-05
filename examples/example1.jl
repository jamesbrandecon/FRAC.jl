using LinearAlgebra, Statistics
using BenchmarkTools
using Plots, DataFrames

using FRAC

Plots.theme(:ggplot2) # Change Plots.jl theme

# In this example, we simulate demand for `T` markets, and assume the researcher
# wants to allow for preferences to differ in `B` separate time periods/geographic
# regions.

B = 100; # number of separate models to estimate
T = 2000; # number of markets/2

J1 = 2; # number of products in half of markets
J2 = 4; # number of products in the other half of markets

df = FRAC.sim_logit_vary_J(J1, J2, T, B, [-1 1], [0.3 0.3], 0.3, with_market_FEs = true);

# Define and estimate two problems: one constrained, another unconstrained
# NOTE: I recommend using se_type = "bootstrap" for all problems, as other standard errors seem to be biased downward
    # see testing_debias.jl for usage of the boostrap function
    # once estimated, standard errors are saved as a dictionary in problem.se
problem = define_problem(data = df, 
            linear = ["prices", "x"], 
            nonlinear = ["prices", "x"],
            # by_var = "by_example", 
            fixed_effects = ["market_ids"],
            se_type = "bootstrap", 
            constrained = false);

estimate!(problem);

problemcon = define_problem(data = df, 
            linear = ["prices", "x"], 
            nonlinear = ["prices", "x"],
            fixed_effects = ["market_ids"],
            se_type = "bootstrap", 
            constrained = true);

estimate!(problemcon)

# Calculate all price elasticities 
    # results stored in problem.all_elasticities
price_elasticities!(problem; monte_carlo_draws = 100);
price_elasticities!(problemcon; monte_carlo_draws = 100);

# Extract all own-price elasticities in a convenient DataFrame
own_elasts = own_elasticities(problem) # implemented, returns dataframe of own-price elasticities paired with market_ids and product_ids
own_elastscon = own_elasticities(problemcon)

# Compare to logit own-price elasticities, which will be wrong when there are random coefficients but which in practice should 
# be highly correlated with the true elasticities
own_elasts[!,"logit_elasticity"] = -1 .* df.prices .* (1 .- df.shares)

scatter(own_elasts.logit_elasticity, own_elasts.own_elasticities, 
    xlabel = "Logit", ylabel = "FRAC", label = false)

# Use helper function to get elasticity of share 1 with respect to price 2
# Output is a dataframe with a market_ids field and a single column of elasticities
# Elasticities will be zero if and only if the two specified products are not both in the corresponding market
elast_1_2 = get_elasticities(problem, [1,2]); 

# Compare estimated fixed effects to the truth 
# NOTE: with only a few products (2-4 depending on the market) in this example, market fixed effects are not precisely estimated
scatter(problem.data.market_FEs, problem.data.estimatedFE_market_ids, 
    xlabel = "Truth", ylabel = "Estimate", label = "market_ids")

# Compare estimated fixed effects between constrained and unconstrained problem
scatter(problem.data.estimatedFE_market_ids, problemcon.data.estimatedFE_market_ids, 
    xlabel = "Unconstrained", ylabel = "Constrained", label = "market_ids")