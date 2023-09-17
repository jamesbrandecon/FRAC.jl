function own_elasticities(problem::FRACProblem)
    if problem.all_elasticities == []
        error("Must estimate model before calculating elasticities")
    end
    df_out = DataFrame();
    df_out[!,"market_ids"] = problem.data.market_ids;
    df_out[!,"product_ids"] = problem.data.product_ids;
    df_out[!,"own_elasticities"] .= 0.0;
    elasts_no_index = getindex.(problem.all_elasticities,2);
    # For each market_id, for each product in that market
    for m ∈ unique(problem.data.market_ids)
        market_index = problem.data.market_ids .==m;
        for j ∈ unique(problem.data.product_ids)
            df_out[(market_index) .& (df_out.product_ids .==j), "own_elasticities"] .= diag(elasts_no_index[getindex.(problem.all_elasticities,1) .== m][1])[j]
        end
    end
    return df_out
end
"""
    get_elasticities(problem::FRACProblem, products)

Helper function to calculate specific pairs of price elasticities. `products` is either a Tuple or Vector of length 2, where the first element is the product for which the elasticity is calculated and the second element is the product with respect to which the elasticity is calculated.
"""
function get_elasticities(problem::FRACProblem, products)
    try 
        @assert ((typeof(products)<:Tuple) | (typeof(products)<:Vector)) & (length(products) ===2)
    catch 
        error("Argument `products` must be either a Tuple or Vector of length 2")
    end    
    prod1 = products[1];
    prod2 = products[2];

    df_out = DataFrame(market_ids = unique(problem.data.market_ids));
    df_out[!, "elast$(prod1)_$(prod2)"] .=0.0
    elasts = getindex.(getindex.(problem.all_elasticities,2), prod1,prod2);
    for m ∈ unique(problem.data.market_ids)
        market_index = (df_out.market_ids .==m);
        df_out[market_index, "elast$(prod1)_$(prod2)"] .= elasts[getindex.(problem.all_elasticities,1) .== m][1]
    end
    return df_out
end

""" 
    all_elasticities(problem::FRACProblem, 
        results::Dict{Any,Any}; 
        I = 50, 
        product = 1, 
        by_value = nothing)

This function takes a `FRACProblem` and a dictionary of results as arguments. The dictionary of results should be the output of the `estimate!` function. It is an internal function that is not intended to be called by the user directly.
"""
function all_elasticities(problem::FRACProblem, 
    results::Dict{Any,Any}; 
    I = 50, 
    product = 1, 
    save_mem = true, 
    by_value = nothing, 
    raw_draws = [])

    linear_vars = problem.linear;
    nonlinear_vars = problem.nonlinear;
    by_var = problem.by_var;

    if raw_draws == []
        error("Must provide monte carlo draws")
    end

    # Check whether problem.data should be subsetted -- if so, define data appropriately
    if problem.by_var != ""
        data = problem.data[problem.data[!,problem.by_var] .== by_value,:];
    else
        data = problem.data;
    end

    # Generate mean utilities for each product, market pair 
        # Begins with Δξ, adds in linear terms due to product characteristics, and then adds effects
    data[!,"delta"] .= data.xi; # Note: xi is a misnomer, these are residuals after absorbing fixed effects 
    # Add contribution from product characteristics 
    if by_var==""
        for l ∈ linear_vars
            data.delta = data.delta + data[!,l] .* results[Symbol("β_$(l)")];
        end 
    else
        for l ∈ linear_vars, b ∈ unique(data[!,by_var])
            data[data[!,by_var].==b,:delta] = data[data[!,by_var].==b, :delta] + data[data[!,by_var].==b,l] .* results[b][Symbol("β_$(l)")];
        end 
    end
    # Add fixed effects
    for f ∈ problem.fe_names
        data.delta = data.delta .+ data[!,"estimatedFE_$(f)"];
    end

    shares_i = zeros(Float64, size(data,1), I);
    alpha_i = zeros(Float64, size(raw_draws[1]));
    if by_var == ""
        shares_i = shares_from_deltas(data.delta, problem.data, monte_carlo_draws = I, raw_draws = 
            raw_draws, return_individual_shares = true, linear_vars = linear_vars, nonlinear_vars = nonlinear_vars, results = results, by_var = problem.by_var);
        alpha_i = results[Symbol("β_prices")] .+ raw_draws[findfirst(problem.nonlinear .== "prices")] .* sqrt(max(results[Symbol("σ2_prices")], 0))
    else 
        for b ∈ unique(data[!,by_var])
            draw_index = findall(data[!,by_var].==b);
            raw_draws_b = [raw_draws[i][draw_index,:] for i ∈ 1:length(raw_draws)];
            shares_i[data[!,by_var].==b,:] = shares_from_deltas(data[data[!,by_var].==b,:delta], data[data[!,by_var].==b,:], monte_carlo_draws = I, raw_draws = raw_draws_b, return_individual_shares = true, linear_vars = linear_vars, nonlinear_vars = nonlinear_vars, results = results[b], by_var = problem.by_var);
            alpha_i[data[!,by_var].==b,:] = results[b][Symbol("β_prices")] .+ raw_draws_b[findfirst(problem.nonlinear .== "prices")] .* sqrt(max(results[b][Symbol("σ2_prices")], 0))
        end
    end

    # own-derivatives are : alpha_i s_i (1-s_i)
    # cross-derivatives are: alpha_i s_i s_j
    # Want matrices such that J(k,j) = ∂s_k / ∂p_j
    # cross_elast = zeros(size(data,1));
    elast_vec = zeros(size(data,1));
    for m = unique(data.market_ids) # loop over markets
        df_m = data[data.market_ids .==m,:];
        prod_ind = findall(df_m[!,"product_ids"] .== product);

        # Find index of product of interest
        if prod_ind != []
            # Individual-level utilities and sums of exp(u) for market m
            # u_i_m = u_i[u_i.market_ids .==m,:]; 
            # u_sums_m = u_sums[u_sums.market_ids .==m,:];

            # Market-specific individual-level shares
            # shares_i = Matrix(u_i_m[!,r"x"])./ Matrix(1 .+ u_sums_m[!,r"x"]);
            shares_i_m = shares_i[data.market_ids .==m,:];
            alpha_i_m = Matrix(alpha_i[data.market_ids .==m, :])[1,:];
            alpha_i_m = reshape(alpha_i_m, (1, length(alpha_i_m)))

            # Now construct elasticities for market m
                # ∂s_k / ∂p_j, k==`product` (function input), j==all other products
                # @show size(alpha_i_m .* shares_i[prod_ind,:] .* (1 .- shares_i[prod_ind,:]));
            elast_m = df_m.prices ./ df_m[prod_ind,"shares"] .* mean(-1 .* alpha_i_m .* shares_i_m .* shares_i[prod_ind,:],dims=2);
            # @show df_m[prod_ind,"prices"] ./ df_m[prod_ind,"shares"] .* mean(alpha_i_m .* shares_i_m[prod_ind,:] .* (1 .- shares_i_m[prod_ind,:]),dims=2)
            elast_m[prod_ind] = df_m[prod_ind,"prices"] ./ df_m[prod_ind,"shares"] .* mean(alpha_i_m .* shares_i_m[prod_ind,:] .* (1 .- shares_i_m[prod_ind,:]),dims=2);
            elast_vec[data[!,"market_ids"].==m,:] = elast_m;
        end
    end
    return elast_vec;
end

"""
    price_elasticities!(problem::FRACProblem)

This function takes a `FRACProblem` as the sole required argument and modifies the `FRACProblem.all_elasticities` property inplace,
replacing the (usually empty) array there with a Vector. Each element of the Vector is a Tuple of two values. 
The first value of the Tuple is a `market_id`. The second is a matrix of price elasticities, where the (i,j)
element of each matrix represents the elasticity of demand for product i with respect to the price of product j 
where `market_ids`==m.
"""
function price_elasticities!(problem::FRACProblem; monte_carlo_draws = 50)
    J = length(unique(problem.data.product_ids));
    M = length(unique(problem.data.market_ids));

    # Initialize Array which will hold all elasticities
    elast_mat = [];
    for m ∈ unique(problem.data.market_ids)
        push!(elast_mat, [zeros(J,J), m])
    end

    # Make draws for random coefficients
    raw_draws = make_draws(problem.data, monte_carlo_draws, length(problem.nonlinear))

    elast_size_J = []
    # Loop over j=1:J to get all elasticities with respect to each good.  
    # Each j here corresponds to a row of the elasticity matrix
    for j = 1:J
        elast = inner_price_elasticities(;problem, 
            frac_results = problem.estimated_parameters, 
            linear = problem.linear, 
            nonlinear = problem.nonlinear, 
            by_var = problem.by_var, 
            monte_carlo_draws = monte_carlo_draws, 
            product = j,
            raw_draws = raw_draws)
        push!(elast_size_J, elast)
    end
    # # Loop over markets, stacking elasticities into one JxJ matrix per market
    all_elasticity_matrices = [];
    all_product_ids = unique(problem.data[!, :product_ids]);
    for m ∈ unique(problem.data.market_ids)
        # Get product_ids in this market 
        product_ids_m = unique(problem.data[problem.data.market_ids .== m, :product_ids]);
        J_m = length(product_ids_m);

        # Reshape elasticities from this market into a single jacobian matrix 
        elasticity_mat_m = zeros(J, J);
        for j ∈ product_ids_m
            indexes_in_market = [j ∈ product_ids_m for j in all_product_ids];
            elasticity_mat_m[j,indexes_in_market.==1] = elast_size_J[j][problem.data.market_ids .== m];
        end
        push!(all_elasticity_matrices, (m,elasticity_mat_m));
    end

    problem.all_elasticities = all_elasticity_matrices;
end

"""
    inner_price_elasticities(;problem, 
        frac_results = [], 
        linear::Vector{String}, 
        nonlinear::Vector{String}, 
        by_var::String = "", 
        monte_carlo_draws = 50, 
        product = 1)
        
This function is called by `price_elasticities!` and is not intended to be called directly by the user. It is an intermediate function that passes the relevant arguments into `all_elasticities` and returns the results to be processed and re-shaped.
"""
function inner_price_elasticities(;problem, 
    frac_results = [], 
    linear::Vector{String}, 
    nonlinear::Vector{String}, 
    by_var::String = "", 
    monte_carlo_draws = 50, product = 1, raw_draws = [])

    if by_var ==""
        elasts = all_elasticities(problem, 
                frac_results, 
                I = monte_carlo_draws, 
                product = product,
                raw_draws = raw_draws)
        output = elasts;
    else        
        # Calculate all price elasticities for each value of by_var 
        output = zeros(size(problem.data,1),1);
        by_var_values = unique(problem.data[!,by_var]);
        for b ∈ by_var_values
            # i = findfirst(by_var_values .==b);
            elasts = all_elasticities(problem, 
                frac_results, 
                I = monte_carlo_draws, 
                product = product, 
                by_value = b,
                raw_draws = raw_draws)
            # output[problem.data[!,by_var].==b,:] = elasts;
            output = elasts;
        end
        
        # end
    end

    return output
end
