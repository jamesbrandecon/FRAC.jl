module FRAC

using DataFrames, FixedEffectModels, LinearAlgebra
using Statistics, Plots, Optim, Missings
using ProgressMeter

# Add helpful extension of addition function
    # This allows us to drop groups of terms from regression when irrelevant
    # by setting them to `nothing`

    # I've tried to define the addition functions as specifically as necessary
    # but these may cause issues in ways I cannot predict.
import Base.+

function +(a::Nothing, b::FixedEffectModels.Term)
    return b;
end

function +(a::Nothing,b::FixedEffectModels.AbstractTerm)
    return b;
end
function +(a::FixedEffectModels.AbstractTerm,b::Nothing)
    return a;
end

function +(a::Tuple,b::Nothing)
    return a;
end
function +(a::Nothing,b::Tuple)
    return b;
end

function estimateFRAC(;linear::String="", nonlinear::String, data::DataFrame, fes::String = "",
    by_var::String="", se_type = "robust", cluster_var = "", constrained::Bool = false,
    drop_singletons::Bool = true, gmm_steps = 1)

    # Using :gpu in FixedEffectModels would make everything faster but
    method = :cpu;


    # To - do
    #   Correlation between random coefficients -- annoying because I have to keep e for both vars
    if linear== ""
        error("At least one covariate (price) must enter utility linearly")
    end
    if !(se_type in ["simple", "robust", "cluster"])
        error("se_type must be either simple, robust, or cluster")
    end
    if se_type == "cluster" && cluster_var == ""
        error("cluster_var must be specified if se_type == 'cluster'")
    end
    if se_type == "simple"
        se = Vcov.simple();
    elseif se_type == "robust"
        se = Vcov.robust();
    elseif se_type == "cluster"
        se = Vcov.cluster(Symbol(cluster_var));
    end

    # Extract variable names from strings
    temp_lin = split(linear, "+");
    linear_vars = replace.(temp_lin, " " => "");
    if linear == "prices"
        linear_exog_terms = nothing;
    else
        linear_exog_vars = linear_vars[2:end];
        linear_exog_terms = sum(term.(Symbol.(linear_exog_vars)));
    end
    L = maximum(size(linear_vars));

    temp_nonlin = split(nonlinear, "+");
    nonlinear_vars = replace.(temp_nonlin, " " => "");

    iv_names = names(data[!, r"demand_instruments"]);

    fe_names = "";
    num_fes = 0;

    # Raise error if model is under-identified
    if length(iv_names) < length(nonlinear_vars) + 1
        error("Not enough instruments! FRAC requires one IV for price and for every nonlinear parameter.")
    end

    # Assuming one or two dimensions of FEs
    fe_terms = nothing;
    if fes!=""
        temp_fes = split(fes, "+");
        fe_names = replace.(temp_fes," " => "");
        num_fes = maximum(size(fe_names));
        if num_fes == 2
            fe_1 = fe_names[1];
            fe_2 = fe_names[2];
            fe_terms = fe(Symbol(fe_1)) + fe(Symbol(fe_2));
        else
            fe_1 = fe_names[1];
            fe_terms = fe(Symbol(fe_1));
        end
    end

    # Construct LHS variable
    gdf = groupby(data, :market_ids);
    cdf = combine(gdf, :shares => sum);
    data = innerjoin(data, cdf, on=:market_ids);
    data[!, "y"] = log.(data[!,"shares"] ./ (1 .- data[!,"shares_sum"]));

    # Construct regressors for variance parameters
    endog_vars = [];
    for i=1:size(nonlinear_vars,1)
        data[!,"e"] = data[!,nonlinear_vars[i]] .* data[!,"shares"];
        gdf = groupby(data,:market_ids);
        cdf = combine(gdf, :e => sum);
        data = innerjoin(data, cdf, on=:market_ids);

        # Generate constructed regressor for variances
        data[!,string("K_", nonlinear_vars[i])] = (data[!,nonlinear_vars[i]] ./2 - data[!,"e_sum"]) .* data[!, nonlinear_vars[i]];
        push!(endog_vars, string("K_", nonlinear_vars[i]))
        data = select!(data, Not(:e_sum))
    end
    push!(endog_vars, "prices")

    # Check if box constraints desired, and either
    # (1) Run IV regressions with FEs
    # (2) Estimate via constrained GMM
    if constrained == false
        # Run IV regression(s)
        results = []
        if by_var ==""
            results = reg(data, term(:y) ~ (sum(term.(Symbol.(endog_vars))) ~ sum(term.(Symbol.(iv_names)))) + linear_exog_terms + fe_terms, se, method=method)
        else
            by_var_values = unique(data[!,by_var]);
            p = Progress(length(by_var_values), 1);
            for b = by_var_values
                results_b = reg(data[data[!,by_var] .== b,:], term(:y) ~ (sum(term.(Symbol.(endog_vars))) ~ sum(term.(Symbol.(iv_names)))) + linear_exog_terms + fe_terms, se, method=method)
                next!(p)
                push!(results, results_b)
            end
        end
    else
        results = FRAC_gmm(data = data, linear_vars = linear_vars,
            nonlinear_vars = nonlinear_vars, fe_names = fe_names,
            iv_names = iv_names, by_var = by_var, num_fes = num_fes, drop_singletons = drop_singletons,
            gmm_steps = gmm_steps)
    end

    #results = reg(df, @formula(y~ (prices + K_1 ~ demand_instruments0 + demand_instruments0^2)))
    return results
end

function gmm_obj(params; data::DataFrame, linear_vars = "", nonlinear_vars = "",
    iv_names = "", W = inv(Array(data[!,iv_names])'Array(data[!,iv_names])), step = 1)
    # xi =  y - beta * linear_vars - Sigma * K
    # moments = xi * iv_names; (n * k)
    # gmm_obj = moments' * moments;

    xi = data[!,"y"]; # initialize xi
    Z = Array(data[!,iv_names]);
    num_linear = maximum(size(linear_vars));
    for i = 1:length(linear_vars)
        xi = xi - params[i] .* data[!,linear_vars[i]];
    end

    for i = 1:length(nonlinear_vars)
        xi = xi - params[i+num_linear] .* data[!,string("K_",nonlinear_vars[i])];
    end
    moments = mean(xi .* Array(data[!,iv_names]), dims=1);
    #W = inv(Z'Z); # will add second step weight matrix at some point

    if (step in [1,2])
        obj = moments * W * moments';
    else
        obj = xi .* Array(data[!,iv_names]);
    end
    return obj[1];
end

function FRAC_gmm(;data::DataFrame, linear_vars = "", nonlinear_vars = "",
    fe_names = "", iv_names = "", by_var = "", num_fes = 0, drop_singletons = true,
    gmm_steps = 1)

    num_linear = maximum(size(linear_vars));
    num_nonlinear = maximum(size(nonlinear_vars));
    price_ind = 0;
    results = [];

    if by_var == ""
        global price_ind, results
        if num_fes > 0
            data = pre_absorb(data, linear_vars, nonlinear_vars, iv_names, fe_names, drop_singletons);
        end

        # When reg() drops singletons, some residuals are missing
        # so I need to drop any rows w/ missing values of anything residualized
        for residual_var = linear_vars
            data_b = data_b[.! isequal.(Array(data_b[!,residual_var]), missing),:];
        end
        for residual_var = nonlinear_vars
            data_b = data_b[.! isequal.(Array(data_b[!,string("K_",residual_var)]), missing),:];
        end
        for residual_var = iv_names
            data_b = data_b[.! isequal.(Array(data_b[!,residual_var]), missing),:];
        end

        # lower is -INF for all linear
            # is zero for all K_
        # upper is INF for all except prices
        lower = convert(AbstractArray,zeros(num_linear + num_nonlinear));
        upper = convert(AbstractArray,Inf .* ones(num_linear + num_nonlinear));
        for i = 1:num_linear
            lower[i] = -Inf;
            if linear_vars[i] == "prices"
                upper[i] = 0; # upper bound α at 0
                price_ind = i;
            end
        end
        for i = 1:num_nonlinear
            lower[i + num_linear] = 0; # lower bound all σ^2 at 0
        end

        initial_param = 0.1 .* ones(num_linear + num_nonlinear);
        initial_param[price_ind] = -1 .* initial_param[price_ind];

        results = optimize(x-> gmm_obj(x; data = data, linear_vars = linear_vars,
            nonlinear_vars = nonlinear_vars, iv_names = iv_names),
            lower, upper, initial_param, Fminbox(); autodiff = :finite);

        results = Optim.minimizer(results);

        if gmm_steps == 2
            initial_param = results;
            moments = gmm_obj(initial_param; data = data, linear_vars = linear_vars,
                nonlinear_vars = nonlinear_vars, iv_names = iv_names, step = 1.5);
            W = inv(moments'moments / size(moments,1));
            # Second GMM step
            results = optimize(x-> gmm_obj(x; data = data, linear_vars = linear_vars,
                nonlinear_vars = nonlinear_vars, iv_names = iv_names, W = W, step = 2),
                lower, upper, initial_param, Fminbox(); autodiff = :finite);
            results = Optim.minimizer(results);
        end

        varnames = linear_vars,nonlinear_vars
        results = [results, varnames]
    else
        by_var_values = unique(data[!,by_var]);
        p = Progress(length(by_var_values),1);
        for b = by_var_values
            global price_ind, results
            data_b = data[data[!,by_var] .== b,:];
            # Absorb fixed effects from covariates
            if num_fes > 0
                data_b = pre_absorb(data_b, linear_vars, nonlinear_vars, iv_names, fe_names, drop_singletons);
            end

            # When reg() drops singletons, some residuals are missing
            # so I need to drop any rows w/ missing values of anything residualized
            for residual_var = linear_vars
                data_b = data_b[.! isequal.(Array(data_b[!,residual_var]), missing),:];
            end
            for residual_var = nonlinear_vars
                data_b = data_b[.! isequal.(Array(data_b[!,string("K_",residual_var)]), missing),:];
            end
            for residual_var = iv_names
                data_b = data_b[.! isequal.(Array(data_b[!,residual_var]), missing),:];
            end

            # lower is -INF for all linear
                # is zero for all K_
            # upper is INF for all except prices
            lower = convert(AbstractArray,zeros(num_linear + num_nonlinear));
            upper = convert(AbstractArray,Inf .* ones(num_linear + num_nonlinear));
            for i = 1:num_linear
                lower[i] = -Inf;
                if linear_vars[i] == "prices"
                    upper[i] = 0; # upper bound α at 0
                    price_ind = i;
                end
            end
            for i = 1:num_nonlinear
                lower[i + num_linear] = 0; # lower bound all σ^2 at 0
            end

            initial_param = 0.1 .* ones(num_linear + num_nonlinear);
            initial_param[price_ind] = -0.1 .* initial_param[price_ind];

            results_b = optimize(x-> gmm_obj(x; data = data_b, linear_vars = linear_vars,
                nonlinear_vars = nonlinear_vars, iv_names = iv_names),
                lower, upper, initial_param, Fminbox(), autodiff = :forward);
            results_b = Optim.minimizer(results_b);

            push!(results,results_b)
            next!(p) # update progress meter
        end
        varnames = linear_vars,nonlinear_vars
        push!(results, varnames)
    end
    return results
end

function pre_absorb(df::DataFrame, linear_vars, nonlinear_vars, iv_names, fe_names, drop_singletons::Bool)
# ------------------------------------------------------
# This function is used to absorb fixed effects prior to
# GMM estimation of FRAC model
# ------------------------------------------------------

    # Construct FE terms for regressions
    num_fes = maximum(size(fe_names));
    if num_fes == 2
        fe_1 = fe_names[1];
        fe_2 = fe_names[2];
        fe_terms = fe(Symbol(fe_1)) + fe(Symbol(fe_2));
    else
        fe_1 = fe_names[1];
        fe_terms = fe(Symbol(fe_1));
    end

    # Run regs and save residuals
    reg_res= reg(df, term(:y) ~ fe_terms, save = :residuals, double_precision = false, drop_singletons = drop_singletons);
    df[!,"y"] = residuals(reg_res,df);
    for i= 1:length(linear_vars)
        v = linear_vars[i];
        df[!,"v"] = df[!,linear_vars[i]];
        reg_res= reg(df, term(:v) ~ fe_terms, save = :residuals, double_precision = false, drop_singletons = drop_singletons);
        df[!,v] = residuals(reg_res,df);
    end
    for i= 1:length(nonlinear_vars)
        v = nonlinear_vars[i];
        df[!,"v"] = df[!,nonlinear_vars[i]];
        reg_res= reg(df, term(:v)  ~ fe_terms, save = :residuals, double_precision = false, drop_singletons = drop_singletons);
        df[!,v] = residuals(reg_res,df);
    end
    for i= 1:length(iv_names)
        v = iv_names[i];
        df[!,"v"] = df[!,iv_names[i]];
        reg_res= reg(df, term(:v) ~ fe_terms, save = :residuals);
        df[!,v] = residuals(reg_res,df);
    end
    return df
end



function simulate_logit(J,T, beta, sd, v)
# --------------------------------------------------%
# Simulate mixed logit with outside option
# --------------------------------------------------%
# J = number of products
# T = number of markets
# beta = coefficients on price and x
# sd = standard deviations of preferences on price and x
# v = standard deviation of market-product demand shock

    if minimum(size(sd)) != 1
        sd = cholesky(sd).U;
    end

    zt = 0.9 .* rand(T,J) .+ 0.05;
    xit = randn(T,J).*v;
    pt = 2 .*(zt .+ rand(T,J)*0.1) .+ xit;
    xt = 2 .* rand(T,J);

    # Loop over markets
    s = zeros(T,J);
    for t = 1:1:T
        I = 1000;
        if minimum(size(sd)) != 1
            beta_i = randn(I,2) * sd;
        else
            beta_i = randn(I,2) .* sd;
        end
        beta_i = beta_i .+ beta;
        denominator = ones(I,1);
        for j = 1:1:J
            denominator = denominator .+ exp.(beta_i[:,1].*pt[t,j] + beta_i[:,2].*xt[t,j] .+ xit[t,j]);
        end
        for j = 1:1:J
            s[t,j] = mean(exp.(beta_i[:,1].* pt[t,j] + beta_i[:,2].*xt[t,j] .+ xit[t,j])./denominator);
        end
    end

    s, pt, zt, xt, xit
end

function toDataFrame(s::Matrix, p::Matrix, z::Matrix, x::Matrix = zeros(size(s)) )
    df = DataFrame();
    J = size(s,2);
    for j = 0:J-1
        df[!, "shares$j"] =  s[:,j+1];
        df[!, "prices$j"] =  p[:,j+1];
        df[!, "demand_instruments$j"] =  z[:,j+1];
        df[!, "x$j"] =  x[:,j+1];
    end
    return df;
end

function reshape_pyblp(df::DataFrame; random_constant = false)
    df.market_ids = 1:size(df,1);
    shares = Matrix(df[!,r"shares"]);

    market_ids = df[!, "market_ids"];
    market_ids = repeat(market_ids, size(shares,2));

    product_ids = repeat((1:size(shares,2))', size(df,1),1);
    product_ids = dropdims(reshape(product_ids, size(df,1)*size(shares,2),1), dims=2);

    shares = dropdims(reshape(shares, size(df,1)*size(shares,2),1), dims=2);

    prices = Matrix(df[!,r"prices"]);
    prices = dropdims(reshape(prices, size(df,1)*size(prices,2),1), dims=2);

    xs = Matrix(df[!,r"x"]);
    xs = dropdims(reshape(xs, size(df,1)*size(xs,2),1), dims=2);

    demand_instruments0 = Matrix(df[!,r"demand_instruments"]);

    if random_constant ==true
        demand_instruments1 = dropdims(repeat(sum(demand_instruments0, dims=2), size(demand_instruments0,2) ,1), dims = 2);
    end

    demand_instruments0 = dropdims(reshape(demand_instruments0, size(df,1)*size(demand_instruments0,2),1), dims=2);

    new_df = DataFrame();
    new_df[!,"shares"] = shares;
    new_df[!,"prices"] = prices;
    new_df[!,"product_ids"] = product_ids;
    new_df[!,"x"] = xs;
    new_df[!,"demand_instruments0"] = demand_instruments0;
    if random_constant ==true
        new_df[!,"demand_instruments1"] = demand_instruments1;
    end
    new_df[!,"market_ids"] = market_ids;
    return new_df
end

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
    shares_i = convert(Array{Float64,2},u_i[!,r"x"])./ convert(Array{Float64,2},(1 .+ u_sums[!,r"x"]));

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
    shares_i = convert(Array{Float64,2},u_i[!,r"x"])./ convert(Array{Float64,2},(1 .+ u_sums[!,r"x"]));

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
            shares_i = convert(Array{Float64,2},u_i_m[!,r"x"])./ convert(Array{Float64,2},(1 .+ u_sums_m[!,r"x"]));

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
            shares_i = convert(Array{Float64,2},u_i_m[!,r"x"])./ convert(Array{Float64,2},(1 .+ u_sums_m[!,r"x"]));

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

    # Find which covariate corresponds to price
    if typeof(frac_results[1]) == FixedEffectModel
        price_index = findall(x-> x=="prices", coefnames(frac_results[1]));
        price_var_index = findall(x-> x=="K_prices", coefnames(frac_results[1]));
    else
        num_lin = length(linear_vars);
        price_index = findall(x-> x=="prices", linear_vars);
        price_var_index = findall(x-> x=="prices", nonlinear_vars)[1] + num_lin;
    end
    mixed_logit = price_var_index !=[];

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
    return output
end




function extractEstimates(df;frac_results = [], var = "prices", param = "mean", by_var = "")
if typeof(frac_results[1]) == FixedEffectModel
    B = maximum(size(unique(df[:,"$by_var"])));
    param_vec = zeros(B);
    se_vec = zeros(B);
    by_vec = zeros(B);
    for i = 1:B
        b = unique(df[:,"$by_var"])[i];
        if param == "mean"
            ind = findall(x-> x == string(var), coefnames(frac_results[i]));
        else
            ind = findall(x-> x == string("K_$var"), coefnames(frac_results[i]));
        end
        param_vec[i] = coef(frac_results[i])[ind][1]
        se_vec[i] = vcov(frac_results[i])[ind][1]
        by_vec[i] = unique(df[:,"$by_var"])[i];
    end
elseif typeof(frac_results[end][1]) <: Array{String}
    # When running constrained GMM, will return string array of
    # variable names as the last element
    B = maximum(size(unique(df[:,"$by_var"])));
    param_vec = zeros(B);
    se_vec = [];
    by_vec = zeros(B);
    num_lin = length(frac_results[end][1]);
    for i = 1:B
        if param == "mean"
            ind = findall(x-> x== string(var), frac_results[end][1]); # search for linear vars
            param_vec[i] = frac_results[i][ind][1];
        else
            num_lin = length(frac_results[end][1]);
            ind = findall(x-> x== string(var), frac_results[end][2])[1] + num_lin;
            param_vec[i] = frac_results[i][ind];
        end
        by_vec[i] = unique(df[:,"$by_var"])[i];
    end
else
    ArgumentError("Provided FRAC results are not an array of FixedEffectModels nor a string array of variable names")
end
return param_vec, se_vec, by_vec
end

function plotFRACResults(df;frac_results = [], var = "prices", param = "mean", plot = "hist", by_var = "")

    # Extract parameter estimates from frac_results
    param_vec, se_vec, by_vals = extractEstimates(df, frac_results = frac_results,
            param = param, by_var = by_var, var = var);

    if plot =="hist"
        return histogram(param_vec, leg=false, xaxis = "Estimates", yaxis = "Frequency", alpha = 0.4)
    elseif plot=="scatter"
        if se_vec ==[]
            return scatter(by_vals, param_vec, leg=false, xaxis = "Models", yaxis = "Estimates")
        else
            return scatter(by_vals, param_vec, leg=false, xaxis = "Models", yaxis = "Estimates", yerror = 1.96 .*se_vec)
        end
    end
end

export estimateFRAC, plotFRACResults, extractEstimates, price_elasticities, simulate_logit, toDataFrame, reshape_pyblp
end
