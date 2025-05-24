# # # Get estimated \xi using contraction mapping (maybe adapt price elasticity func to do the same?)
#     # Two versions: one with contraction, the other using the approximate \xi from FRAC
# """
#     get_xi(FRACPRoblem; approximate = true)

# This takes the results of a FRAC problem and constructs the implied values of ξ (the aggregate demand shifters). 
# If `approximate` is set to `true`, ξ is constructed as the residual of the FRAC estimation problem itself. If `false`,
# ξ is constructed via the standard BLP inversion (contraction mapping).   
# """
# function get_xi(problem::FRACProblem; by_var = "", by_value = 0.0, approximate = true)
#     results = problem.results;
#     if approximate ==true
#         xi = get_xi_approx(problem, by_var, by_value)
#     else
#         xi = get_xi_contraction(problem)
#     end

#     return xi
# end    

function replace_xi_contraction!(problem::FRACProblem; precision = 1e-12, n_draws = 50)
    deltas = run_contraction_mapping(
        zeros(Float64, size(problem.data,1)), 
        problem, 
        precision = precision, n_draws = n_draws)
    
    # REPLACE deltas 
    problem.data[!,"delta"] .= deltas;

    xi = deltas;
    for l ∈ problem.linear
        xi = xi - problem.estimated_parameters[Symbol("β_$(l)")].* problem.data[!,l];
    end
    # for nl ∈ problem.nonlinear
    #     xi = xi - sqrt(max(0, problem.estimated_parameters[Symbol("σ2_$(nl)")])) .* problem.data[!,"K_$(nl)"];
    # end
    
    # ADD xi
    problem.data[!,"xi_contraction"] .= xi;
    
    # define fe_terms 
    fe_terms = nothing;
    if problem.fe_names!=[""]
        num_fes = maximum(size(problem.fe_names));
        for i=1:num_fes
            fe_terms += fe(Symbol(problem.fe_names[i]));
        end
    end

    # Calculate fixed effects
    results = reg(problem.data, term(:xi_contraction) ~ fe_terms, 
        save = :all, 
        double_precision = false, 
        drop_singletons = problem.drop_singletons);
    estimated_FEs = fe(results);
    
    for f ∈ problem.fe_names
        problem.data[:,"estimatedFE_$(f)"] .= estimated_FEs[!,findfirst(problem.fe_names .== f)];
        problem.data[!,"xi_contraction"] = problem.data[!,"xi_contraction"] .- problem.data[!,"estimatedFE_$(f)"];
    end   

    # REPLACE xi
    problem.data[!,"xi"] = residuals(results) #problem.data[!,"xi_contraction"];
end

#ORIGINAL
function shares_from_deltas(deltas, data::DataFrame; 
    seed = 10293, monte_carlo_draws = 50, 
    raw_draws = [], return_individual_shares = false,
    linear_vars = [], nonlinear_vars = [],
    results = [], by_var = "")

    I = monte_carlo_draws;
    n_obs = nrow(data)
    K = length(nonlinear_vars)

    # Simulate draws for random coefficients
    eta_i = zeros(Float64, size(data,1),1);
    u_i = zeros(Float64, size(data,1),1);
    alpha_i = zeros(Float64, size(data,1),I);
    
    # Currently a loop over number of simulated customers for a reason I don't remember -- old code, should update to matrix 
    # for i = 1:I
    #     eta_i = zeros(Float64, size(data,1),1);
    #     for nl ∈ nonlinear_vars
    #         sigma = sqrt(max(results[Symbol("σ2_$(nl)")], 0)); # Drop heterogeneity if estimates are negative
    #         k = findfirst(nonlinear_vars .== nl)
    #         scaled_draws = sigma .* raw_draws[k][:,i];
    #         eta_i = eta_i .+ data[!,nl] .* scaled_draws;
    #     end
    #     u_i = hcat(u_i, deltas + eta_i);
    # end
     # 1) Preallocate U = [const | δ + η] (n_obs × (I+1))
     U = zeros(Float64, n_obs, I+1)
     U[:, 2:end] .+= deltas  # put δ in every simulation column
 
     # 2) Add each random‐coef term: U[:,2:end] .+= σ_k * X_k .* raw_draws[k]
     @inbounds for (k, nl) in enumerate(nonlinear_vars)
         σk = sqrt(max(results[Symbol("σ2_$(nl)")], 0))
         # raw_draws[k] is n_obs×I, data[!,nl] is n_obs‐vector
         U[:, 2:end] .+= data[!, nl] .* (σk .* raw_draws[k])
     end
 
     # 3) exponentiate and drop the “const” column
     u_i = exp.(U[:, 2:end])  # now n_obs×I

    # Exponentiate individual utilities for market share calculation and add market_ids column
    # u_i = exp.(u_i[:,2:end]);
    u_i = DataFrame(u_i, :auto);
    u_i[!,"market_ids"] = data.market_ids;
    u_i[!,"product_ids"] = data.product_ids;

    # Calculate market-level sums of exp(u)
    gdf = groupby(u_i, :market_ids);
    cdf = combine(gdf, names(u_i) .=> sum);
    cdf = select(cdf, Not(:market_ids_sum));
    u_sums = innerjoin(
        select(u_i, :market_ids), 
        cdf, on = :market_ids);
    # u_i = transform(gdf,
    #     names(u_i, r"x") .=> (s -> sum(s)) .=> names(u_i, r"x")
    #     )
    # shares_i = Matrix(u_i[!, r"x"]) ./ (1 .+ Matrix(u_i[!, r"x"])) 
    
    sort!(u_i, [:market_ids, :product_ids])
    sort!(u_sums, [:market_ids])
    shares_i = Matrix(u_i[!,r"x"])./ Matrix(1 .+ u_sums[!,r"x"]);
    if return_individual_shares
        shares_i
        return shares_i
    else
        shares = mean(shares_i, dims=2);
        return shares
    end
end

function run_contraction_mapping(initial_deltas, 
    problem::FRACProblem; 
    precision = 1e-12, 
    n_draws = 50, 
    verbose = true)

    I = n_draws;
    copyproblem = deepcopy(problem);
    delta_next = initial_deltas;
    delta_old = initial_deltas .+ 1;

    raw_draws = make_draws(problem.data, I, length(problem.nonlinear), 
        method = :normal, common_draws = true, antithetic = true)
    if problem.cov != []
        raw_draws = correlate_draws(raw_draws, problem.data, problem.nonlinear, problem.cov, problem.estimated_parameters)
    end
    println("Running contraction mapping...")
    println("Target precision: $(round(precision, sigdigits = 2))")
    while maximum(abs.(delta_next .- delta_old)) > precision
        delta_old .= delta_next;
        delta_next .= delta_old .+ 
                log.(problem.data.shares) .- 
                log.(shares_from_deltas(delta_old, 
                    copyproblem.data, 
                    monte_carlo_draws=I, 
                    raw_draws = raw_draws, 
                    # linear_vars = problem.linear, 
                    nonlinear_vars = problem.nonlinear, 
                    results = problem.estimated_parameters
                    # by_var = problem.by_var
                    ));
                    # for i in 1:100
                    #     pct = round(i/100*100; digits=1)
                gap = round(max(maximum(abs.(delta_next .- delta_old))), sigdigits = 2)
                print("\rContraction Gap: $(gap)")
                flush(stdout)
    end
    println("")
    return delta_next
end


function shares_for_bootstrap(problem, xi_boot; I=50, save_mem = true, raw_draws = [])
    linear_vars = problem.linear;
    nonlinear_vars = problem.nonlinear;
    results = problem.estimated_parameters;

    # Check whether problem.data should be subsetted -- if so, define data appropriately
    if problem.by_var != ""
        data = problem.data[problem.data[!,problem.by_var] .== by_value,:];
    else
        data = problem.data;
    end

    # Generate mean utilities for each product, market pair 
        # Begins with Δξ, adds in linear terms due to product characteristics, and then adds effects
    # Begin
    deltas = xi_boot; # Note: xi is a misnomer, these are residuals after absorbing fixed effects 
    # # Add contribution from product characteristics 
    for l ∈ linear_vars
        deltas = deltas + data[!,l] .* results[Symbol("β_$(l)")];
    end
    # Add fixed effects
    for f ∈ problem.fe_names
        deltas = deltas .+ data[!,"estimatedFE_$(f)"];
    end
    
    shares = shares_from_deltas(
        deltas, 
        data, 
        monte_carlo_draws = size(raw_draws[1],2), 
        raw_draws = raw_draws, 
        linear_vars = linear_vars, 
        nonlinear_vars = nonlinear_vars, 
        results = results, 
        by_var = problem.by_var);
    
    return shares
end


# # Resample xi 
function resample(xi)
    # Sample with replacement
    L = maximum(size(xi));
    sample_inds = rand(1:L, L);
    return xi[sample_inds]
end

"""
    bootstrap!(problem::FRACProblem; nboot = 100, ndraws = 100, approximate = false)

This function bootstraps the parameters of a FRAC problem. It does so by resampling the residuals ξ and 
re-simulating market shares repeatedly. The function modifies the `problem` object in place, adding the 
bootstrapped parameters (a vector of dictionaries including all bootstrap) to the `bootstrapped\\_parameters\\_all` field and the debiased parameters to the 
`bootstrap\\_debiased\\_parameters` field. The latter is calculated via the following: 

`bootstrap\\_debiased\\_parameters` = 2 β - 1/B sum(β^b)

where β is the original estimate and β^b is the estimate from the bth bootstrap sample.
"""
function bootstrap!(problem::FRACProblem; 
    nboot = 100, 
    ndraws = 100, 
    approximate = false, 
    precision = 1e-12)

    if approximate == false 
        replace_xi_contraction!(problem, precision = precision, n_draws = ndraws)
    end

    # Get xi from original problem, to be resampled
    estimated_xi = problem.data.xi;

    # Make monte carlow draws -- note: this sets the seed
    raw_draws = make_draws(
        problem.data, 
        ndraws, 
        length(problem.nonlinear), 
        method = :normal, common_draws = true, antithetic = true)

    if problem.by_var == ""
        boot_results = [];
        for nb = 1:nboot
            resampled_xi = resample(estimated_xi);
            bootstrapped_shares = shares_for_bootstrap(problem, 
                                        resampled_xi, raw_draws = raw_draws)
            copydata = deepcopy(problem.data);
            copydata[!,"shares"] .= bootstrapped_shares
            problem_boot = define_problem(data = select(copydata, Not(:xi)), 
                    linear = problem.linear, 
                    nonlinear = problem.nonlinear,
                    fixed_effects = problem.fe_names,
                    se_type = problem.se_type, 
                    constrained = problem.constrained);
            estimate!(problem_boot)
            push!(boot_results, problem_boot.estimated_parameters)
            
            # Clean up memory
            problem_boot= nothing;
            copydata = nothing;
            GC.gc()
        end
        # Loop over each results and calcluate the average among bootstrapped samples
    else
        for b ∈ unique(problem.data[!,problem.by_var])
        end
    end

    # bootstrap adjusted params = 2 * \hat β - 1/B ∑_b β^b 
    adjusted_parameters = deepcopy(problem.estimated_parameters)
    for i ∈ eachindex(problem.estimated_parameters)
        sum_boot = sum([boot_results[j][i] for j = 1:nboot])
        adjusted_parameters[i] = 2 .* problem.estimated_parameters[i] .- (1/nboot .* sum_boot);
    end
    # return boot_results, adjusted_parameters
    problem.bootstrapped_parameters_all = boot_results;
    problem.bootstrap_debiased_parameters = adjusted_parameters;
    
    # Calculate standard deviation of each parameter in boot_results and assign to problem.se 
    se_dict = Dict()
    for i ∈ eachindex(problem.estimated_parameters)
        push!(se_dict, i => std([boot_results[j][i] for j = 1:nboot]))
    end
    problem.se = se_dict;

end

function make_draws(data::DataFrame, I::Int, K::Int; seed::Integer = 10293,
                     method::Symbol = :normal, antithetic::Bool = true,
                     common_draws::Bool = false, skip::Integer = 500)
    # Prepare RNG
    rs = MersenneTwister(seed)

    markets = sort(unique(data.market_ids))
    n_markets = length(markets)
    n_obs = size(data,1)
    store_draws = Vector{Matrix{Float64}}(undef, K)

    for k in 1:K
        # Generate base draws per market (or common) of size (n_markets x I)
        if common_draws
            # draw once and repeat
            if method == :normal
                # antithetic sampling if requested
                if antithetic
                    half = I ÷ 2
                    base = randn(rs, half)
                    base = hcat(base, -base)
                else
                    base = randn(rs, I)
                end
            elseif method == :halton
                 # 1) Generate one (I×K) Halton in [0,1]^K
                sampler = HaltonSample()
                H = Matrix(QuasiMonteCarlo.sample(I, K, sampler)')
                # 2) map each column to N(0,1)
                Z = [quantile.(Normal(), H[:, k]) for k in 1:K]  # Vector of length-K each size I
                # 3) build per-market draws
                for k in 1:K
                    base = Z[k]'                              # 1×I row vector
                    if common_draws
                        rand_draws = repeat(base, n_markets, 1) # same for all markets
                    else
                        # # different for each market? tile the same row for now
                        # rand_draws = repeat(base, n_markets, 1)
                        @warn "common_draws is false, but Halton draws are common"
                    end
                    # expand market→obs
                    M = zeros(n_obs, I)
                    for (mi, m) in enumerate(markets)
                        mask = data.market_ids .== m
                        M[mask, :] .= repeat(rand_draws[mi, :]', sum(mask), 1)
                    end
                    store_draws[k] = M
                end
            else
                error("Unknown draw method: $(method)")
            end
            # replicate for all markets
            rand_draws = repeat(reshape(base, 1, I), n_markets, 1)
        else
            # draw separately for each market
            if method == :normal
                if antithetic
                    half = I ÷ 2
                    part = randn(rs, n_markets, half)
                    rand_draws = hcat(part, -part)
                else
                    rand_draws = randn(rs, n_markets, I)
                end
            elseif method == :halton
                total = n_markets * I
                H = Vector{Float64}(undef, total)
                HaltonDraws!(H, prime(k+1); skip = skip, distr = Normal())
                rand_draws = reshape(H, n_markets, I)
            else
                error("Unknown draw method: $(method)")
            end
        end

        # Map draws to each observation
        draws_length = zeros(Float64, n_obs, I)
        for (idx, m) in enumerate(markets)
            mask = data.market_ids .== m
            draws_length[mask, :] .= reshape(rand_draws[idx, :], 1, I)
        end
        store_draws[k] = draws_length
    end
    return store_draws
end

"""
    correlate_draws(raw_draws, data, nonlinear, cov, parameters)

Given independent standard normal draws in `raw_draws` (Vector of matrices length K, size n_obs×I),
and a full estimated covariance structure in `parameters` (with keys :σ2_var and :σcov_var1_var2),
returns a new set of draws that exhibit the specified correlations across dimensions.
"""
function correlate_draws(raw_draws::Vector{Matrix{Float64}}, data::DataFrame,
                         nonlinear::Vector{String}, cov::Vector{Tuple{String,String}},
                         parameters::Dict)
    # Number of random coefficients
    K = length(nonlinear)
    # Build covariance matrix Σ (K×K)
    Σ = zeros(K, K)
    for (i, nl) in enumerate(nonlinear)
        # variance term σ²_nl
        v = get(parameters, Symbol("σ2_$(nl)"), 0.0)
        Σ[i, i] = max(v, 0.0)
    end
    # Off-diagonal covariances σcov_v1_v2
    for (v1, v2) in cov
        if v1 != "" && v2 != ""
            i = findfirst(nonlinear .== v1)
            j = findfirst(nonlinear .== v2)
            if i !== nothing && j !== nothing
                key = Symbol("σcov_$(v1)_$(v2)")
                if haskey(parameters, key)
                    c = parameters[key]
                    Σ[i, j] = c
                    Σ[j, i] = c
                end
            end
        end
    end
    # Cholesky factor (upper triangular U such that U' * U = Σ)
    U = cholesky(Σ).U
    # Prepare correlated draws container
    new_draws = [zeros(size(raw_draws[1])) for _ in 1:K]
    markets = sort(unique(data.market_ids))
    # Transform base draws into correlated draws by market and simulation
    for m in markets
        mask = data.market_ids .== m
        # representative index for this market
        idx = findfirst(mask)
        # number of simulation draws
        I_draws = size(raw_draws[1], 2)
        # Collect independent draws into matrix Z (I_draws × K)
        Z = hcat([raw_draws[k][idx, :] for k in 1:K]...)
        # Apply correlation: Z_corr[i, :] = Z[i, :] * U
        Z_corr = Z * U
        n_obs_m = sum(mask)
        for k in 1:K
            row_vec = Z_corr[:, k]'
            block = repeat(reshape(row_vec, 1, I_draws), n_obs_m, 1)
            new_draws[k][mask, :] = block
        end
    end
    return new_draws
end


"""
    `HaltonSeq!{T<:AbstractFloat}(H::Vector{T}, B::Int; skip::Int=500)`
Copied from an old version of Halton.jl. Replaces `H` with entries from Halton low discrepancy sequence with base `B`.
Elements in `H` take values in the interval (0, 1).
Keyword argument `skip` is the number initial "burn-in" elements to drop.
"""
function HaltonSeq!(H::AbstractArray{<:AbstractFloat}, B::Integer; skip::Integer=500)
  isprime(B) || error("base number not prime")
  H!(H, B, skip)
end

"""
    `HaltonDraws!{T<:AbstractFloat}(H::Vector{T}, B::Int, [skip::Int=500, distr=Normal()])`
Copied from an old version of Halton.jl. Replaces `H` with draws from a distribution `Distributions.dist()`.
Draws are generated by using Halton sequence with base `B` as the quantiles drawn.
Keyword argument `skip` is the number initial "burn-in" elements to drop.
"""
function HaltonDraws!(H::AbstractArray, B::Integer; skip::Integer=500, distr = Normal())
  HaltonSeq!(H, B, skip=skip)
  H .= Distributions.quantile.(distr, H)
end

####

## Algorithm for generating Halton sequences
function H!(H::AbstractArray{T}, b::IT, skip::Integer) where {T<:AbstractFloat, IT<:Integer}
  # Fill H with Halton Sequence based on b
  S = skip + length(H)
  # set D to get generated seq >= S
  D = ceil(IT, log(S) / log(b))
  # placeholders
  d = zeros(T, D+1)
  r = zeros(T, D+1)

  # based on algorithm found in https://www.researchgate.net/publication/229173824_Fast_portable_and_reliable_algorithm_for_the_calculation_of_Halton_numbers
  for nn in 1:S
    ii = 1
    while d[ii] == b-1
      d[ii] = zero(T)
      ii += 1
    end
    d[ii] += 1
    if ii>=2
      r[ii - 1] = (d[ii] + r[ii]) / b
    end
    if ii >= 3
      for jj in (ii-1) : -1 : 2
        r[jj-1] = r[jj] / b
      end
    end
    if nn>skip
      H[nn-skip] = (d[1] + r[1]) / b
    end
  end
  return H
end

# function contraction(delta::Vector{<:Real}, s::Vector{Float64}, nubeta::Matrix{<:Real}, 
#     df::DataFrame, param, inner_tol::Float64)
#     delta = reshape(delta, size(s))
#     S = size(nubeta,2);

#     delta_i = repeat(reshape(delta, size(s)), 1, S) + nubeta;

#     # Calculate predicted market shares
#     s_hat = reshape(s_hat_logitRC1(delta_i, eta_i1, df.market_ids), size(delta));
#     s_reshape = reshape(s, size(delta));

#     # Update delta
#     out = delta .+  log.(s_reshape ./ s_hat)
#     return out
# end

# cdf = combine(groupby(df, :market_ids), :shares => sum => :sumshare)
# df = leftjoin(df, cdf, on = :market_ids);
# logs0 = log.(1 .- df.sumshare);

# logit_inv= log.(df.shares) - logs0;
# for k = 1:1:K2
#     nubeta = nubeta + X_i[:,k,:] .* (draws[:,k,:] .* sigma[k])
# end

# delta_1 = squaremJL(x -> contraction(x, df[!,"shares"], nubeta, df, param,
#  inner_tol), logit_inv; tol = outer_tol);
# xi = delta_1 - Matrix(theta .*df[!,linear]);

# function get_xi_approx(problem::FRACProblem, by_var = "", by_value = 0.0)
#     i=1;

#     linear_vars = problem.linear;
#     nonlinear_vars = problem.linear;
#     data = problem.data;
#     data = data[data[!,by_var] .== by_value,:];

#     params = problem.results[i];
    
#     xi = data[!,"y"]; # initialize xi
    
#     num_linear = maximum(size(linear_vars));
#     for lin ∈ linear_vars
#         xi = xi - params[i] .* data[!,lin];
#     end

#     for nonlin ∈ nonlinear_vars
#         xi = xi - params[i+num_linear] .* data[!,string("K_",nonlin)];
#     end

#     return xi
# end
    # # Simulate draws for random coefficients
    # eta_i = zeros(size(data,1),1);
    # u_i = zeros(size(data,1),1);
    # alpha_i = zeros(size(data,1),I);

    # # Currently a loop over number of simulated customers for a reason I don't remember -- old code, should update to matrix 
    # for i = 1:I
    #     eta_i = zeros(size(data,1),1);
    #     for nl ∈ nonlinear_vars
    #         sigma = sqrt(max(results[Symbol("σ2_$(nl)")], 0)); # Drop heterogeneity if estimates are negative
    #         scaled_draws = sigma .* randn(size(data,1),1);
    #         if nl == "prices"
    #             alpha_i[:,i] .= scaled_draws;
    #         end
    #         eta_i = eta_i .+ data[!,nl] .* scaled_draws;
    #     end
    #     u_i = hcat(u_i, data.delta + eta_i);
    # end

    # # Exponentiate individual utilities for market share calculation and add market_ids column
    # u_i = exp.(u_i[:,2:end]);
    # u_i = DataFrame(u_i, :auto);
    # u_i[!,"market_ids"] = data.market_ids;

    # # Calculate market-level sums of exp(u)
    # gdf = groupby(u_i, :market_ids);
    # cdf = combine(gdf, names(u_i) .=> sum);
    # cdf = select(cdf, Not(:market_ids_sum));

    # # Make a DataFrame with only market-level sums of exp(u)
    # u_sums = innerjoin(select(u_i, :market_ids), cdf, on = :market_ids);

    # shares_i = Matrix(u_i[!,r"x"])./ Matrix(1 .+ u_sums[!,r"x"]);
    # shares = mean(shares_i, dims=2);
    # # for each sample, calculate shares using estimated parameters and sampled \xi 

# # re-estimate parameters via FRAC 
# function reestimate(problem; nboot = 100, approximate = true)
#     data_copy = deepcopy(problem.data);
#     xi = get_xi(problem; approximate = approximate);

#     for nb = 1:nboot 
#         xi_boot = resample(xi);
#         shares_for_bootstrap!(data_copy, xi_boot);
#     end
# end
