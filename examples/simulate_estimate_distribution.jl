using FRACDemand, Random, DataFrames, Plots

# Simulation settings
nsim = 100
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
    df = sim_logit_vary_J(J1, J2, T, B, β, Σ, ξ_var)
    
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

for k in param_keys
    histogram(estimates[k];
        bins=30, title=string(k),
        xlabel=string(k), legend=false)
    vline!([true_vals[k]]; color=:red, linestyle=:dash, linewidth=2)
    savefig("examples/$(k)_distribution.png")
end
