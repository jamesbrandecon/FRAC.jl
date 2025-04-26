using FRAC, Random, DataFrames, Plots
using Statistics

# Simulation settings
nsim = 1000
J1, J2, T, B = 5, 5, 200, 1
β = [-2.0, 1.5]
Σ = [0.5 0.3; 0.3 0.8]
ξ_var = 0.01

# storage for all elasticity pairs
all_elast = DataFrame(mse=Float64[], mape=Float64[], 
    betax=Float64[], betap=Float64[], σ2x=Float64[], σ2p=Float64[], σcov=Float64[])

all_elast_nocov = DataFrame(mse=Float64[], mape=Float64[],
    betax=Float64[], betap=Float64[], σ2x=Float64[], σ2p=Float64[], σcov=Float64[])

for s in 1:nsim
    Random.seed!(s)
    # simulate and reshape
    df = FRAC.sim_logit_vary_J(J1, J2, T, B, β, Σ, ξ_var)
    df[!,"demand_instruments3"] .= df.demand_instruments0 .* df.demand_instruments1 .* df.demand_instruments2

    # set up IV and estimate
    problem = define_problem(
      data          = df,
      linear        = ["prices","x"],
      nonlinear     = ["prices","x"],
      cov           = [("prices","x")],
      fixed_effects = ["product_ids"],
      se_type       = "bootstrap",
      constrained   = false
    )
    estimate!(problem)

    # true elasticities
    truth = FRAC.sim_true_price_elasticities(df, β, Σ)

    # estimated elasticities
    est_elast = DataFrame(market_ids=Int[], product_i=Int[], product_j=Int[], elasticity=Float64[])
    for subdf in groupby(problem.data, :market_ids)
        p, x, xi = subdf.prices, subdf.x, subdf.xi
        fe = "market_FEs" in names(subdf) ? first(subdf.market_FEs) : 0
        βhat = [problem.estimated_parameters[:β_prices], problem.estimated_parameters[:β_x]]
        Ehat = FRAC.sim_price_elasticities(p, x, xi, βhat, Σ; market_FE=fe)
        mid, J = first(subdf.market_ids), length(p)
        for i in 1:J, j in 1:J
            push!(est_elast, (market_ids=mid, product_i=i, product_j=j, elasticity=Ehat[i,j]))
        end
    end

    # collect data for sim
    # DataFrame(truth=truth.elasticity, est=est_elast.elasticity)
    append!(all_elast, DataFrame(
        mse  = mean((truth.elasticity .- est_elast.elasticity).^2), 
        mape = mean(abs.(truth.elasticity .- est_elast.elasticity) ./ abs.(truth.elasticity)), 
        betax = problem.estimated_parameters[:β_x], 
        betap = problem.estimated_parameters[:β_prices],
        σ2x  = problem.estimated_parameters[:σ2_x],
        σ2p  = problem.estimated_parameters[:σ2_prices],
        σcov = problem.estimated_parameters[:σcov_prices_x]
    ))

    # set up IV and estimate
    problem_nocov = define_problem(
      data          = select(df, Not(:demand_instruments3)),
      linear        = ["prices","x"],
      nonlinear     = ["prices","x"],
    #   cov           = [("prices","x")],
      fixed_effects = ["product_ids"],
      se_type       = "bootstrap",
      constrained   = false
    )
    estimate!(problem_nocov)

    # estimated elasticities
    est_elast = DataFrame(market_ids=Int[], product_i=Int[], product_j=Int[], elasticity=Float64[])
    for subdf in groupby(problem_nocov.data, :market_ids)
        p, x, xi = subdf.prices, subdf.x, subdf.xi
        fe = "market_FEs" in names(subdf) ? first(subdf.market_FEs) : 0
        βhat = [problem_nocov.estimated_parameters[:β_prices], problem_nocov.estimated_parameters[:β_x]]
        Ehat = FRAC.sim_price_elasticities(p, x, xi, βhat, Σ; market_FE=fe)
        mid, J = first(subdf.market_ids), length(p)
        for i in 1:J, j in 1:J
            push!(est_elast, (market_ids=mid, product_i=i, product_j=j, elasticity=Ehat[i,j]))
        end
    end

    # collect data for sim
    append!(all_elast_nocov, DataFrame(
        mse  = mean((truth.elasticity .- est_elast.elasticity).^2), 
        mape = mean(abs.(truth.elasticity .- est_elast.elasticity) ./ abs.(truth.elasticity)), 
        betax = problem_nocov.estimated_parameters[:β_x], 
        betap = problem_nocov.estimated_parameters[:β_prices],
        σ2x  = problem_nocov.estimated_parameters[:σ2_x],
        σ2p  = problem_nocov.estimated_parameters[:σ2_prices],
        σcov = 0;#problem.estimated_parameters[:σcov_prices_x]
    ))
    print(".")
end

function cov_plots(result_obj)
        p1 = histogram(
            result_obj.σ2x,
        bins=30,
        label="Estimated Distribution",
        xlabel="σ²_x",
        ylabel="Frequency",
        title="Estimated σ²_x",
        color=:skyblue, # Assign a color for this histogram
        legend=:topright
    )
    vline!(p1, [Σ[2,2]], color=:red, linestyle=:dash, linewidth=2, label="True Value")

    # Plot for σ²_prices
    p2 = histogram(
        result_obj.σ2p,
        bins=30,
        label="Estimated Distribution",
        xlabel="σ²_prices",
        ylabel="Frequency",
        title="Estimated σ²_prices",
        color=:lightgreen, # Assign a different color
        legend=:topright
    )
    vline!(p2, [Σ[1,1]], color=:red, linestyle=:dash, linewidth=2, label="True Value")

    # Plot for σ_cov(prices, x)
    p3 = histogram(
        result_obj.σcov,
        bins=30,
        label="Estimated Distribution",
        xlabel="σ_cov(prices, x)",
        ylabel="Frequency",
        title="Estimated σ_cov(prices, x)",
        color=:lightcoral, # Assign another different color
        legend=:topright
    )
    vline!(p3, [Σ[1,2]], color=:red, linestyle=:dash, linewidth=2, label="True Value")

    # Combine plots into a single figure with a layout and an overall title
    p_out = plot(p1, p2, p3,
        layout = (3, 1),          # Arrange plots vertically
        size = (700, 900),        # Adjust figure size as needed
        plot_title = "Distribution of Estimated Var/Covar Params" # Overall title
        )
    return p_out
end

cov_plots(all_elast_nocov)
cov_plots(all_elast)


histogram(
    all_elast.mse;
    bins=30, xlabel="MSE", ylabel="Count",
    title="Mean Squared Error of Price Elasticities", legend=false
    )

histogram(
    all_elast.mape;
    bins=30, xlabel="MAPE", ylabel="Count",
    title="Mean Absolute Percentage Error of Price Elasticities", 
    label="With Cov Estiamtion", 
    normalize = :density, 
    alpha = 0.5,
    color = :lightgreen
    )
histogram!(
    all_elast_nocov.mape;
    bins=30, 
    xlabel="MAPE", ylabel="Count",
    label="Without Cov Estimation", 
    normalize = :density, 
    alpha = 0.5, 
    color = :skyblue
    )

histogram(
    all_elast.mse;
    bins=30, xlabel="MSE", ylabel="Count",
    title="Mean Absolute Percentage Error of Price Elasticities", 
    label="With Cov Estiamtion", 
    normalize = :density, 
    alpha = 0.5,
    color = :lightgreen
    )
histogram!(
    all_elast_nocov.mse;
    bins=30, 
    xlabel="MSE", ylabel="Count",
    label="Without Cov Estimation", 
    normalize = :density, 
    alpha = 0.5, 
    color = :skyblue
    )

scatter(
    all_elast_nocov.mape, all_elast.mape;
    xlabel="No cov", ylabel="w/ Cov",
    title="MAPE of Price Elasticities", legend=false, alpha=0.2
    )
plot!(
    all_elast_nocov.mape, all_elast_nocov.mape;
    color=:red, linestyle=:dash, linewidth=2,
    label="45 degree line"
    )
scatter(
    all_elast_nocov.mse, all_elast.mse;
    xlabel="No cov", ylabel="w/ Cov",
    title="MSE of Price Elasticities", legend=false, alpha=0.2
    )
plot!(
    all_elast_nocov.mse, all_elast_nocov.mse;
    color=:red, linestyle=:dash, linewidth=2,
    label="45 degree line"
    )

histogram(
    all_elast.betax;
    bins=30, xlabel="β_x", ylabel="Count",
    title="Estimated β_x", 
    label = "With Cov Estimation", alpha = 0.4
    )
histogram!(
    all_elast_nocov.betax;
    bins=30, xlabel="β_x", ylabel="Count",
    title="Estimated β_x", 
    label = "Without Cov Estimation", alpha = 0.4
    )
vline!([β[2]]; color=:red, linestyle=:dash, linewidth=2)

histogram(
    all_elast.betap;
    bins=30, xlabel="β_prices", ylabel="Count",
    title="Estimated β_prices", 
    label = "With Cov Estimation",alpha = 0.4
    )
histogram!(
    all_elast_nocov.betap;
    bins=30, xlabel="β_prices", ylabel="Count",
    title="Estimated β_prices", 
    label = "Without Cov Estimation", alpha = 0.4
    )
vline!([β[1]]; color=:red, linestyle=:dash, linewidth=2)

# Plot variance and covariance parameters

# Plot for σ²_x

# If you want to save the combined plot:
# savefig("examples/variance_covariance_distributions.png")




# scatter and error histogram
scatter(all_elast.truth, all_elast.est;
  xlabel="True elasticity", ylabel="Estimated elasticity",
  title="Estimated vs True Price Elasticities", legend=false, alpha=0.2)
savefig("examples/elasticity_scatter.png")

histogram((all_elast.est .- all_elast.truth).^2;
  bins=30, xlabel="Est – True", title="Elasticity Estimation Error", legend=false)
savefig("examples/elasticity_error_hist.png")
