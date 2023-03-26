using LinearAlgebra, Statistics
using BenchmarkTools
using Plots, DataFrames

using FRAC

Plots.theme(:ggplot2) # Change Plots.jl theme

# In this example, we simulate demand for `T` markets, and assume the researcher
# wants to allow for preferences to differ in `B` separate time periods/geographic
# regions.

B = 100; # number of separate models to estimate
T = 20000; # number of markets/2

J1 = 2; # number of products in half of markets
J2 = 4; # number of products in the other half of markets

df = FRAC.sim_logit_vary_J(J1, J2, T, B, [-0.4 1], [0.5 0.5], 0.3)

problem = define_problem(data = df, 
            linear = ["prices", "x"], 
            nonlinear = ["prices", "x"],
            by_var = "by_example", 
            fixed_effects = ["product_ids"],
            se_type = "robust", 
            constrained = true);

estimate!(problem)

price_elasticities!(problem)







price_elasticities(frac_results = problem.results, data = df,
    linear = "prices + x", nonlinear = "prices + x", which = "own",
    by_var="by_example", product=2)
# Estimate FRAC and calculate
#   (1) all own-price elasticities and
#   (2) all elasticities w.r.t. the good with product_ids == 1

# All models include two dimensions of random coefficients and two fixed effects
@time results = estimateFRAC(data = df, linear= "prices + x", nonlinear = "prices + x",
    by_var = "by_example", fes = "product_ids + dummy_FE",
    se_type = "robust", constrained = false)
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
