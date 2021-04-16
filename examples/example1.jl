using LinearAlgebra, Statistics
using BenchmarkTools
using Plots, DataFrames

using Main.FRAC

Plots.theme(:ggplot2) # Change Plots.jl theme

# In this example, we simulate demand for `T` markets, and assume the researcher
# wants to allow for preferences to differ in `B` separate time periods/geographic
# regions.

# This file performs the following:
# -- Simulate demand data for many markets with varying numbers of products
# -- Calculate IVs
# -- Estimate FRAC models and calculate various price elasticities
# -- Extract and plot results of all models

B = 100; # number of separate models to estimate
T = 20000; # number of markets/2

J1 = 2; # number of products in half of markets
J2 = 4;

# Simulate demand with two characteristics, both with random coefficients
# Run benchmarks for multiple FRAC models
s,p,z,x = simulate_logit(J1,T,[-0.4 0.1], [0.5 0.5], 0.3);
s2,p2,z2,x2 = simulate_logit(J2,T,[-0.4 0.1], [0.5 0.5], 0.3);

# Reshape data into desired DataFrame, add necessary IVs
df = toDataFrame(s,p,z,x);
df = reshape_pyblp(df);
df2 = reshape_pyblp(toDataFrame(s2,p2,z2,x2));
df2[!,"product_ids"] = df2.product_ids .+ 2;
df2[!,"market_ids"] = df2.market_ids .+ T .+1;
df = [df;df2]
df[!,"by_example"] = mod.(1:size(df,1),B); # Generate variable indicating separate geographies
df[!,"demand_instruments1"] = df.demand_instruments0.^2;
df[!,"demand_instruments2"] = df.x .^2;

# Add simple differentiation-style IVs: difference from market-level sum
gdf = groupby(df, :market_ids);
cdf = combine(gdf, names(df) .=> sum);
cdf = select(cdf, Not(:market_ids_sum));
sums = innerjoin(select(df, :market_ids), cdf, on = :market_ids);
df[!,"demand_instruments3"] = (df.demand_instruments0 - sums.demand_instruments0_sum).^2;
df[!,"demand_instruments4"] = (df.x - sums.x_sum).^2;
df[!,"demand_instruments5"] = (df.demand_instruments1 - sums.demand_instruments1_sum).^2;
df[!,"demand_instruments6"] = (df.demand_instruments2 - sums.demand_instruments2_sum).^2;

df[!,"dummy_FE"] .= rand();
df[!,"dummy_FE"] = (df.dummy_FE .> 0.5);

# Estimate FRAC and calculate
#   (1) all own-price elasticities and
#   (2) all elasticities w.r.t. the good with product_ids == 1

# All models include two dimensions of random coefficients and two fixed effects
@time results = estimateFRAC(data = df, linear= "prices + x", nonlinear = "prices + x",
    by_var = "by_example", fes = "product_ids + dummy_FE",
    se_type = "robust", constrained = true)
@time own_elast = price_elasticities(frac_results = results, data = df,
    linear = "prices + x", nonlinear = "prices + x", which = "own",
    by_var="by_example")
@time cross_elast = price_elasticities(frac_results = results, data = df,
    linear = "prices + x", nonlinear = "prices + x", which = "cross",
    by_var="by_example", product=2)

# Extraneous helper functions can be used to plot FRAC results with standard
# errors (when unconstrained). Here we show, for all estimated models,
#   (1) Mean price coefficient
#   (2) Variance of price coefficient
#   (3) Variance of x coefficient
# (Horizontal lines indicate true value)

plot1 = plotFRACResults(df, frac_results = results, var = "prices", plot = "scatter", by_var = "by_example")
plot!(plot1, ylims = (-1, 0), leg=false)
plot!([-0.4], linetype = :hline, linewidth = 2)

plot2 = plotFRACResults(df, frac_results = results, var = "prices", param = "sd", plot = "scatter", by_var = "by_example")
plot!(plot2, ylims = (0, 0.5))
plot!([0.25], linetype = :hline, linewidth = 2)

plotFRACResults(df,frac_results = results, var = "x", param = "sd", plot = "scatter", by_var = "by_example")
plot!([0.25], linetype = :hline, linewidth = 2)

# Can also extract estimates, SEs, and groups (values of by_var) in the same order
alphas, alpha_se, byvals = extractEstimates(df, frac_results = results, param = "mean", by_var = "by_example")
sigmas, sigma_se, byvals = extractEstimates(df, frac_results = results, param = "sigmas", by_var = "by_example")
xsigmas, xsigma_se, byvals = extractEstimates(df, frac_results = results, var = "x", param = "sigmas", by_var = "by_example")
