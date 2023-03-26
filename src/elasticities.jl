function own_price_elasticities(data::DataFrame, linear_vars,
    nonlinear_vars, results::FixedEffectModel; I =50, save_mem = true)
    # Find linear variables and parameters
    linear_mean = zeros(size(data,1),1);
    for l = 1:maximum(size(linear_vars))
        l_index = findall(x-> x==linear_vars[l], coefnames(results));
        linear_mean = linear_mean + data[!,linear_vars[l]] .* coef(results)[l_index];
    end
    # Use results to calculate residuals. Estimate of delta = linear_mean + xi
    data[!,"xi"] = data[:,"y"] .- dropdims(linear_mean, dims=2);
    u_i = zeros(size(data,1),I);

    # u_i = data[!, linear_vars] * beta_i + eta_i
    eta_i = zeros(size(data,1),1);
    if save_mem == true
        for i = 1:I
            for nl = 1:maximum(size(nonlinear_vars))
                nl_index = findall(x-> x== string("K_",nonlinear_vars[nl]), coefnames(results));
                sigma = max(coef(results)[nl_index][1], 0); # Drop heterogeneity if estimates are negative
                eta_i = eta_i .+ data[!,nonlinear_vars[nl]] .* sigma .* randn(size(data,1),1);
            end
            u_i = dropdims(linear_mean,dims=2) + eta_i;
        end
    end
    u_i = exp.(u_i);
    u_i = DataFrame(u_i, :auto);
    u_i[!,"market_ids"] = data.market_ids;

    # Calculate market-level sums of exp(u)
    gdf = groupby(u_i, :market_ids);
    cdf = combine(gdf, names(u_i) .=> sum);
    cdf = select(cdf, Not(:market_ids_sum));

    # Make a DataFrame with only market-level sums
    u_sums = innerjoin(select(u_i, :market_ids), cdf, on = :market_ids);
    shares_i = Matrix(u_i[!,r"x"])./ Matrix(1 .+ u_sums[!,r"x"]);

    price_index = findall(x-> x=="prices", coefnames(results));
    price_var_index = findall(x-> x=="K_prices", coefnames(results));
    alpha = coef(results)[price_index];
    sigma = max(coef(results)[price_var_index][1],0); # Don't use negative values for sigma
    alpha_i = randn(1,I).*sigma .+ alpha;

    # Closely matches estimates from NPDemand, March 3, 2021
    own = data.prices ./ data.shares .* mean(repeat(alpha_i,size(data,1),1) .* shares_i .* (1 .- shares_i), dims = 2);
    return own;
end

function own_price_elasticities(data::DataFrame, linear_vars,
    nonlinear_vars, results::Array{Float64,1}; I =50, save_mem = true)

    # Find linear variables and parameters
    num_lin = length(linear_vars);
    linear_mean = zeros(size(data,1),1);
    for l = 1:maximum(size(linear_vars))
        linear_mean = linear_mean + data[!,linear_vars[l]] .* results[l];
    end
    # Use results to calculate residuals. Estimate of delta = linear_mean + xi
    data[!,"xi"] = data[:,"y"] .- dropdims(linear_mean, dims=2);

    u_i = zeros(size(data,1),I);
    eta_i = zeros(size(data,1),1);
    if save_mem == true
        for i = 1:I
            for nl = 1:maximum(size(nonlinear_vars))
                #nl_index = findall(x-> x== string("K_",nonlinear_vars[nl]), coefnames(results));
                sigma = results[nl + num_lin] # Drop heterogeneity if estimates are negative
                eta_i = eta_i .+ data[!,nonlinear_vars[nl]] .* sigma .* randn(size(data,1),1);
            end
            u_i = dropdims(linear_mean,dims=2) + eta_i;
        end
    end
    u_i = exp.(u_i);
    u_i = DataFrame(u_i, :auto);
    u_i[!,"market_ids"] = data.market_ids;

    # Calculate market-level sums of exp(u)
    gdf = groupby(u_i, :market_ids);
    cdf = combine(gdf, names(u_i) .=> sum);
    cdf = select(cdf, Not(:market_ids_sum));

    # Make a DataFrame with only market-level sums
    u_sums = innerjoin(select(u_i, :market_ids), cdf, on = :market_ids);
    shares_i = Matrix(u_i[!,r"x"])./ Matrix(1 .+ u_sums[!,r"x"]);

    price_index = findall(x-> x=="prices", linear_vars);
    price_var_index = findall(x-> x=="prices", nonlinear_vars)[1] + num_lin;
    alpha = results[price_index];
    sigma = results[price_var_index]
    alpha_i = randn(1,I).*sigma .+ alpha;

    own = data.prices ./ data.shares .* mean(repeat(alpha_i,size(data,1),1) .* shares_i .* (1 .- shares_i), dims = 2);
    #(JMB)-- matches unconstrained estimates almost exactly
    return own;
end

function cross_elasticities(data::DataFrame, linear_vars,
    nonlinear_vars, results::FixedEffectModel; I =50, product = 1, save_mem = true)
    # using HaltonSequences
    # Halton(2, length = I) or haltonvalue(1,2) (first draw from halton w/ base 2)

    linear_mean = zeros(size(data,1),1);
    for l = 1:maximum(size(linear_vars))
        l_index = findall(x-> x==linear_vars[l], coefnames(results));
        linear_mean = linear_mean + data[!,linear_vars[l]] .* coef(results)[l_index];
    end
    # Use calculate residuals. Estimate of delta = linear_mean + xi
    data[!,"xi"] = data[:,"y"] .- dropdims(linear_mean, dims=2);

    eta_i = zeros(size(data,1),1);
    if save_mem == true
        for i = 1:I
            for nl = 1:maximum(size(nonlinear_vars))
                nl_index = findall(x-> x== string("K_",nonlinear_vars[nl]), coefnames(results));
                sigma = max(coef(results)[nl_index][1], 0); # Drop heterogeneity if estimates are negative
                eta_i = eta_i .+ data[!,nonlinear_vars[nl]] .* sigma .* randn(size(data,1),1);
            end
            u_i = dropdims(linear_mean,dims=2) + eta_i;
        end
    end
    u_i = exp.(u_i);
    u_i = DataFrame(u_i, :auto);
    u_i[!,"market_ids"] = data.market_ids;

    # Calculate market-level sums of exp(u)
    gdf = groupby(u_i, :market_ids);
    cdf = combine(gdf, names(u_i) .=> sum);
    cdf = select(cdf, Not(:market_ids_sum));

    # Make a DataFrame with only market-level sums
    u_sums = innerjoin(select(u_i, :market_ids), cdf, on = :market_ids);
    #shares_i = convert(Array{Float64,2},u_i[!,r"x"])./ convert(Array{Float64,2},(1 .+ u_sums[!,r"x"]));

    price_index = findall(x-> x=="prices", coefnames(results));
    price_var_index = findall(x-> x=="K_prices", coefnames(results));
    alpha = coef(results)[price_index];
    sigma = max(coef(results)[price_var_index][1],0); # Don't use negative values for sigma
    alpha_i = randn(1,I).*sigma .+ alpha;

    # own-derivatives are : alpha_i s_i (1-s_i)
    # cross-derivatives are: alpha_i s_i s_j
    # Want matrices such that J(i,j) = \partial s_i / \partial p_j
    #cross_elast = zeros(size(data,1));
    cross_elast = zeros(size(data,1));
    for m = unique(data.market_ids) # loop over markets
        df_m = data[data.market_ids .==m,:];
        prod_ind = findall(x-> x== product, df_m[!,"product_ids"]);
        # find index of product of interest
        if prod_ind != []
            u_i_m = u_i[u_i.market_ids .==m,:]; # restrict data to market m
            u_sums_m = u_sums[u_sums.market_ids .==m,:];

            # Getting market-specific shares
            shares_i = Matrix(u_i_m[!,r"x"])./ Matrix(1 .+ u_sums_m[!,r"x"]);

            elast_m = df_m.prices ./ df_m[prod_ind,"shares"] .* mean(-1 .* alpha_i .* shares_i .* shares_i[prod_ind,:],dims=2);
            elast_m[prod_ind] = df_m[prod_ind,"prices"] ./ df_m[prod_ind,"shares"] .* mean(alpha_i .* shares_i[prod_ind,:] .* (1 .- shares_i[prod_ind,:]),dims=2);
            global cross_elast[data[!,"market_ids"].==m,:] = elast_m;
        end
    end
    return cross_elast;
end

function cross_elasticities(data::DataFrame, linear_vars,
    nonlinear_vars, results::Array{Float64,1}; I =50, product = 1, save_mem = true)

    num_lin = length(linear_vars);
    linear_mean = zeros(size(data,1),1);
    for l = 1:maximum(size(linear_vars))
        linear_mean = linear_mean + data[!,linear_vars[l]] .* results[l];
    end
    # Use calculate residuals. Estimate of delta = linear_mean + xi
    data[!,"xi"] = data[:,"y"] .- dropdims(linear_mean, dims=2);
    eta_i = zeros(size(data,1),1);
    if save_mem == true
        for i = 1:I
            for nl = 1:maximum(size(nonlinear_vars))
                sigma = results[nl+num_lin]; # Drop heterogeneity if estimates are negative
                eta_i = eta_i .+ data[!,nonlinear_vars[nl]] .* sigma .* randn(size(data,1),1);
            end
            u_i = dropdims(linear_mean,dims=2) + eta_i;
        end
    end
    u_i = exp.(u_i);
    u_i = DataFrame(u_i, :auto);
    u_i[!,"market_ids"] = data.market_ids;

    # Calculate market-level sums of exp(u)
    gdf = groupby(u_i, :market_ids);
    cdf = combine(gdf, names(u_i) .=> sum);
    cdf = select(cdf, Not(:market_ids_sum));

    # Make a DataFrame with only market-level sums
    u_sums = innerjoin(select(u_i, :market_ids), cdf, on = :market_ids);
    #shares_i = convert(Array{Float64,2},u_i[!,r"x"])./ convert(Array{Float64,2},(1 .+ u_sums[!,r"x"]));

    price_index = findall(x-> x=="prices", linear_vars);
    price_var_index = findall(x-> x=="prices", nonlinear_vars);
    alpha = results[price_index];
    sigma = results[price_var_index]; # Don't use negative values for sigma
    alpha_i = randn(1,I).*sigma .+ alpha;
    cross_elast = zeros(size(data,1));
    for m = unique(data.market_ids) # loop over markets
        df_m = data[data.market_ids .==m,:];
        prod_ind = findall(x-> x== product, df_m[!,"product_ids"]);
        # find index of product of interest
        if prod_ind != []
            u_i_m = u_i[u_i.market_ids .==m,:]; # restrict data to market m
            u_sums_m = u_sums[u_sums.market_ids .==m,:];

            # Getting market-specific shares
            shares_i = Matrix(u_i_m[!,r"x"])./ Matrix(1 .+ u_sums_m[!,r"x"]);

            elast_m = df_m.prices ./ df_m[prod_ind,"shares"] .* mean(-1 .* alpha_i .* shares_i .* shares_i[prod_ind,:],dims=2);
            elast_m[prod_ind] = df_m[prod_ind,"prices"] ./ df_m[prod_ind,"shares"] .* mean(alpha_i .* shares_i[prod_ind,:] .* (1 .- shares_i[prod_ind,:]),dims=2);
            global cross_elast[data[!,"market_ids"].==m,:] = elast_m;
        end
    end
#(JMB)-- matches unconstrained estimates almost exactly
return cross_elast
end

function price_elasticities(;frac_results = [], data::DataFrame,
        linear::String, nonlinear::String, by_var::String = "", which = "own",
        monte_carlo_draws = 50, product = 1)
    # This function takes in results of FRAC estimation
    # and produces estiamtes of price elasticities

    # Construct LHS variable
    gdf = groupby(data, :market_ids);
    cdf = combine(gdf, :shares => sum);
    data = innerjoin(data, cdf, on=:market_ids);
    data[!, "y"] = log.(data[!,"shares"] ./ (1 .- data[!,"shares_sum"]));

    # Extract variable names from strings
    temp_lin = split(linear, "+");
    linear_vars = replace.(temp_lin, " " => "");
    L = maximum(size(linear_vars));

    temp_nonlin = split(nonlinear, "+");
    nonlinear_vars = replace.(temp_nonlin, " " => "");
    NL = maximum(size(nonlinear_vars));

    if by_var ==""
        result_for_type_check = frac_results;
    else
        result_for_type_check = frac_results[1];
    end

    # Find which covariate corresponds to price
    if typeof(result_for_type_check) == FixedEffectModel
        price_index = findall(x-> x=="prices", coefnames(result_for_type_check));
        price_var_index = findall(x-> x=="K_prices", coefnames(result_for_type_check));
    else
        num_lin = length(linear_vars);
        price_index = findall(x-> x=="prices", linear_vars);
        price_var_index = findall(x-> x=="prices", nonlinear_vars)[1] + num_lin;
    end
    mixed_logit = price_var_index !=[];

    if by_var ==""
        if which == "own"
            own = own_price_elasticities(data, linear_vars,
                nonlinear_vars, frac_results, I = monte_carlo_draws)
            output = own;
        else
            cross_elast = cross_elasticities(data, linear_vars,
                nonlinear_vars, frac_results, I = monte_carlo_draws, product = product)
            output = cross_elast;
        end
    else
        if which == "own"
            output = zeros(size(data,1),1);
            by_var_values = unique(data[!,by_var]);
            for b = by_var_values
                i = findfirst(x->x==b, by_var_values);
                    own = own_price_elasticities(data[data[!,by_var].==b,:], linear_vars,
                        nonlinear_vars, frac_results[i], I = monte_carlo_draws)
                output[data[!,by_var].==b,:] = own;
            end
        else
            # Calculate cross-price elasticities
            # have to be very careful to call correct products
            output = zeros(size(data,1),1);
            by_var_values = unique(data[!,by_var]);
            for b = by_var_values
                i = findfirst(x->x==b, by_var_values);
                cross_elast = cross_elasticities(data[data[!,by_var].==b,:], linear_vars,
                    nonlinear_vars, frac_results[i], I = monte_carlo_draws, product = product)
                output[data[!,by_var].==b,:] = cross_elast;
            end
        end
    end

    return output
end
