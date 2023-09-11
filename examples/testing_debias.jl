using LinearAlgebra, Statistics
using Plots, DataFrames

using FRAC

Plots.theme(:ggplot2) # Change Plots.jl theme

T = 1000; # number of markets/2
J1 = 2; # number of products in half of markets
J2 = 4; # number of products in the other half of markets

original_estimates = []
debias_estimates = []
for i = 1:5
    df = FRAC.sim_logit_vary_J(J1, J2, T, 1, [-1 1], [0.3 0.3], 0.3, with_market_FEs = true);

    problem = define_problem(data = df, 
                linear = ["prices", "x"], 
                nonlinear = ["prices", "x"],
                by_var = "", 
                fixed_effects = ["market_ids"],
                se_type = "robust", 
                constrained = false);

    estimate!(problem)
    push!(original_estimates, problem.estimated_parameters)
    bootstrap!(problem; nboot = 25, approximate = false)
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
histogram([debias_estimates[j][name] for j ∈ 1:length(debias_estimates)],label = "debiased", normalize = :probability)
histogram!([original_estimates[j][name] for j ∈ 1:length(original_estimates)], label = "original", normalize = :probability)
# histogram!([a[j][name] for j ∈ 1:length(a)], label = "a", alpha = 0.3)

# histogram([a[j][name] for j ∈ 1:length(a)], label = "a")
# quantile([a[j][name] for j ∈ 1:length(a)], 0.975)
# quantile([a[j][name] for j ∈ 1:length(a)], 0.025)
# FRAC.replace_xi_contraction!(problem)
# RD = FRAC.make_draws(problem.data, I, length(problem.nonlinear))
# estimated_xi = problem.data.xi;
# resampled_xi = FRAC.resample(estimated_xi);
# display(mean(resampled_xi))
# bootstrapped_shares = FRAC.shares_for_bootstrap(problem, resampled_xi, raw_draws = RD)
# copydata = deepcopy(problem.data);
# copydata[!,"shares"] = bootstrapped_shares
# problem_boot = define_problem(data = select(copydata, Not(:xi)), 
#         linear = problem.linear, 
#         nonlinear = problem.nonlinear,
#         by_var = problem.by_var, 
#         fixed_effects = problem.fe_names,
#         se_type = problem.se_type, 
#         constrained = problem.constrained);
# estimate!(problem_boot)