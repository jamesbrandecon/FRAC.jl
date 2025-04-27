using LinearAlgebra, Statistics
using Plots, DataFrames

using FRACDemand

Plots.theme(:ggplot2) # Change Plots.jl theme

T = 1000; # number of markets/2
J1 = 2; # number of products in half of markets
J2 = 4; # number of products in the other half of markets

nboot = 5; # Number of bootstrap iterations. Use a larger number in actual usage
 
original_estimates = []
debias_estimates = []
for i = 1:25
    mean_utility_params = [-1 1];
    random_coefficient_sd = [0.3 0.3];
    df = FRACDemand.sim_logit_vary_J(J1, J2, T, 1, mean_utility_params, random_coefficient_sd, 0.3, with_market_FEs = true);

    problem = define_problem(data = df, 
                linear = ["prices", "x"], 
                nonlinear = ["prices", "x"],
                fixed_effects = ["market_ids"],
                se_type = "bootstrap", 
                constrained = true);

    estimate!(problem)
    push!(original_estimates, problem.estimated_parameters)
    bootstrap!(problem; nboot = nboot, approximate = false)
    push!(debias_estimates, problem.bootstrap_debiased_parameters)
    println("Done with iteration $i")
end

# Compare averages of estimates 
original_avg_dict = Dict()
debiased_avg_dict = Dict()
for i ∈ eachindex(original_estimates[1])
    push!(original_avg_dict, i => mean([original_estimates[j][i] for j ∈ 1:length(original_estimates)]))
    push!(debiased_avg_dict, i => mean([debias_estimates[j][i] for j ∈ 1:length(debias_estimates)]))
end
println("Average of Original Estimates")
display(original_avg_dict)
println("Average of Debiased Estimates")
display(debiased_avg_dict)

name = :β_prices
histogram([debias_estimates[j][name] for j ∈ 1:length(debias_estimates)],
    label = "debiased", normalize = :probability)
histogram!([original_estimates[j][name] for j ∈ 1:length(original_estimates)], 
    label = "original", normalize = :probability)
