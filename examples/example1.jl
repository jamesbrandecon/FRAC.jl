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

df = FRAC.sim_logit_vary_J(J1, J2, T, B, [-1 1], [0.1 0.1], 0.3, with_market_FEs = true);

problem = define_problem(data = df, 
            linear = ["prices", "x"], 
            nonlinear = ["prices", "x"],
            by_var = "", 
            fixed_effects = ["market_ids"],
            se_type = "robust", 
            constrained = false);

estimate!(problem)

problemcon = define_problem(data = df, 
            linear = ["prices", "x"], 
            nonlinear = ["prices", "x"],
            by_var = "", 
            fixed_effects = ["market_ids"],
            se_type = "robust", 
            constrained = true);

estimate!(problemcon)

# Calculate all price elasticities 
    # results stored in problem.all_elasticities
price_elasticities!(problem; monte_carlo_draws = 100);
price_elasticities!(problemcon; monte_carlo_draws = 100);

# Extract all own-price elasticities in a convenient DataFrame
own_elasts = own_elasticities(problem) # implemented, returns dataframe of own-price elasticities paired with market_ids and product_ids
own_elastscon = own_elasticities(problemcon)

# Compare estimated fixed effects between constrained and unconstrained problem
scatter(problem.data.estimatedFE_market_ids, problemcon.data.estimatedFE_market_ids)
scatter(problem.data.xi, problemcon.data.xi, label = "xi")


# get_elasticities(problem, product = 1) # JMB need to implement

# a, b = debias!(problem; nboot = 20, approximate = true)

# copyproblem = deepcopy(problem);
# delta_next = zeros(size(problem.data,1));
# delta_old = delta_next .+ 1;
# I = 500
# raw_draws = FRAC.make_draws(problem.data, I)
# while maximum(abs.(delta_next .- delta_old)) > 1e-12
#     delta_old .= delta_next;
#     delta_next .= delta_old .+ log.(problem.data.shares) .- 
#             log.(FRAC.shares_from_deltas(delta_old, copyproblem, monte_carlo_draws=I, raw_draws = raw_draws));
#     @show maximum(abs.(log.(problem.data.shares) .- log.(FRAC.shares_from_deltas(delta_old, copyproblem))))
#     @show maximum(abs.(delta_next .- delta_old))
# end

# # Make real deltas and verify that calculated shares are very close to the truth 
# real_deltas = df.x .- df.prices .+ df.market_FEs .+ df.xi
# problem.estimated_parameters = Dict(
#     :β_prices => -1, 
#     :β_x => 1, 
#     :σ2_prices => 0.1, 
#     :σ2_x => 0.1)
# scatter(df.shares, FRAC.shares_from_deltas(real_deltas, problem, monte_carlo_draws=500, raw_draws = FRAC.make_draws(problem.data, 500, 2)))
# df.shares .= FRAC.shares_from_deltas(real_deltas, problem, monte_carlo_draws=500, raw_draws = FRAC.make_draws(problem.data, 500))
# raw_draws = FRAC.make_draws(problem.data, 2000)
# FRAC.shares_from_deltas(real_deltas, problem, monte_carlo_draws=500, raw_draws = raw_draws)
# df.shares
# histogram(df.shares.- FRAC.shares_from_deltas(real_deltas, problem, monte_carlo_draws=2000, raw_draws = raw_draws))
# scatter(df.shares, FRAC.shares_from_deltas(real_deltas, problem, monte_carlo_draws=I, raw_draws = raw_draws))

# # Check contraction mapping 
# delta_next = real_deltas; #zeros(size(real_deltas));
# delta_old = delta_next .+ 1;
# I = 100
# raw_draws = FRAC.make_draws(problem.data, I, 2, seed=1023)
# copyproblem = deepcopy(problem)
# while maximum(abs.(delta_next .- delta_old)) > 1e-12
#     delta_old = delta_next;
#     delta_next = delta_old .+ log.(df.shares) .- 
#             log.(FRAC.shares_from_deltas(delta_old, copyproblem, monte_carlo_draws=I, raw_draws = raw_draws));
#     # @show maximum(abs.(log.(df.shares) .- log.(FRAC.shares_from_deltas(delta_old, copyproblem,  monte_carlo_draws=I, raw_draws = raw_draws))))
#     # @show maximum(abs.(delta_next .- delta_old))
#     # @show mean(log.(problem.data.shares))
# end

# scatter(delta_next, real_deltas)

# # scatter(delta_next_start_real, delta_next)

# prod1 = getindex.(getindex.(problem.all_elasticities,2),6,6);
# prod1 = prod1[prod1.!=0]
# Plots.histogram(prod1)
# # Plots.histogram!(getindex.(problem.all_elasticities,2,2))

# df = deepcopy(problem.data)
# df.delta = exp.(-1 .* df.prices .+ df.x .+ df.estimatedFE_product_ids .+ df.estimatedFE_market_ids .+ df.xi);
# gdf = combine(groupby(df, :market_ids), :delta => sum => :sumdelta)
# leftjoin!(df, gdf, on = :market_ids)
# scatter(df.delta ./ ( 1 .+ df.sumdelta), df.shares)

# # for m ∈ unique(df.market_ids)

# # end



# price_elasticities(frac_results = problem.results, data = df,
#     linear = "prices + x", nonlinear = "prices + x", which = "own",
#     by_var="by_example", product=2)
# # Estimate FRAC and calculate
# #   (1) all own-price elasticities and
# #   (2) all elasticities w.r.t. the good with product_ids == 1

# # All models include two dimensions of random coefficients and two fixed effects
# @time results = estimateFRAC(data = df, linear= "prices + x", nonlinear = "prices + x",
#     by_var = "by_example", fes = "product_ids + dummy_FE",
#     se_type = "robust", constrained = false)
# @time own_elast = price_elasticities(frac_results = results, data = df,
#     linear = "prices + x", nonlinear = "prices + x", which = "own",
#     by_var="by_example")
# @time cross_elast = price_elasticities(frac_results = results, data = df,
#     linear = "prices + x", nonlinear = "prices + x", which = "cross",
#     by_var="by_example", product=2)

# # Extraneous helper functions can be used to plot FRAC results with standard
# # errors (when unconstrained). Here we show, for all estimated models,
# #   (1) Mean price coefficient
# #   (2) Variance of price coefficient
# #   (3) Variance of x coefficient
# # (Horizontal lines indicate true value)

# plot1 = plotFRACResults(df, frac_results = results, var = "prices", plot = "scatter", by_var = "by_example")
# plot!(plot1, ylims = (-1, 0), leg=false)
# plot!([-0.4], linetype = :hline, linewidth = 2)

# plot2 = plotFRACResults(df, frac_results = results, var = "prices", param = "sd", plot = "scatter", by_var = "by_example")
# plot!(plot2, ylims = (0, 0.5))
# plot!([0.25], linetype = :hline, linewidth = 2)

# plotFRACResults(df,frac_results = results, var = "x", param = "sd", plot = "scatter", by_var = "by_example")
# plot!([0.25], linetype = :hline, linewidth = 2)

# # Can also extract estimates, SEs, and groups (values of by_var) in the same order
# alphas, alpha_se, byvals = extractEstimates(df, frac_results = results, param = "mean", by_var = "by_example")
# sigmas, sigma_se, byvals = extractEstimates(df, frac_results = results, param = "sigmas", by_var = "by_example")
# xsigmas, xsigma_se, byvals = extractEstimates(df, frac_results = results, var = "x", param = "sigmas", by_var = "by_example")
