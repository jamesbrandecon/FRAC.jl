using FRACDemand, Random, DataFrames, Plots

# Simulation settings
nsim = 1000
J1, J2, T, B = 10, 10, 500, 1
β = [-2.0, 1.5]
Σ = [0.5 0.3;
     0.3 0.8]
ξ_var = 1.0  # nonzero market‐shock variance

# Prepare storage for each parameter
param_keys = [:β_prices, :β_x, :σ2_prices, :σ2_x, :σcov_prices_x]
estimates = Dict(k => Float64[] for k in param_keys)

for s in 1:nsim
    Random.seed!(s)
    # simulate data
    df = FRACDemand.sim_logit_vary_J(J1, J2, T, B, β, Σ, ξ_var)
    
    # build extra IV -- not a good idea in practice but fine in simulations
    df[!,"demand_instruments3"] .= df.demand_instruments0 .* df.demand_instruments1 .* df.demand_instruments2

    # define and run estimation
    problem = define_problem(
        data          = df,
        linear        = ["prices", "x"],
        nonlinear     = ["prices", "x"],
        cov           = [("prices","x")],
        fixed_effects = ["product_ids"],
        se_type       = "simple",
        constrained   = false
    )
    estimate!(problem)

    # collect estimates
    est = problem.estimated_parameters
    for k in param_keys
        push!(estimates[k], get(est, k, NaN))
    end
end

# plot distributions with true values
true_vals = Dict(
    :β_prices        => β[1],
    :β_x             => β[2],
    :σ2_prices       => Σ[1,1],
    :σ2_x            => Σ[2,2],
    :σcov_prices_x   => Σ[1,2]
)

plots = []
for k in param_keys
    temp = histogram(estimates[k];
        bins=30, title=string(k),
        xlabel=string(k), legend=false)
    vline!(temp, [true_vals[k]]; color=:red, linestyle=:dash, linewidth=2)
    push!(plots, temp)
end
plot(plots...; layout=(3,2), size=(800, 600))

# --------------------------------------------------------
# Miscellaneous plot examples to confirm estimation works well 
# --------------------------------------------------------
scatter(
    FRACDemand.shares_from_deltas(
        problem.data.delta, 
        problem.data, 
        results = problem.estimated_parameters), 
        problem.data.shares
        )

scatter(
    original_delta, problem.data.delta;
    xlabel="Original delta", ylabel="Estimated delta",
    title="Delta: Original vs Estimated", legend=false, alpha=0.2
)

scatter(
    original_xi, problem.data.xi_contraction;
    xlabel="Original xi", ylabel="Estimated xi",
    title="xi: Original vs Estimated", legend=false, alpha=0.2
)