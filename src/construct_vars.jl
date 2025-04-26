function compute_combined(gdf, shares_col, value_col)
    combine(gdf, [shares_col, value_col] => ((x, y) -> sum(x .* y)) => Symbol("e_", value_col))
end

function make_vars(problem,
                   linear::Vector{String},
                   nonlinear::Vector{String},
                   cov::Vector{Tuple{String,String}},
                   fe_names::Vector{String},
                   data::DataFrame)
    
    # cov is a vector of pairs for covariance parameters; no symmetry check here

    linear_vars = linear;
    linear_exog_vars = nothing;
    linear_exog_terms = nothing;
    if linear != ["prices"]
        linear_exog_vars = linear_vars[2:end];
        linear_exog_terms = sum(term.(Symbol.(linear_exog_vars)));
    end
    L = maximum(size(linear_vars));

    # nonlinear_vars = replace.(temp_nonlin, " " => "");
    nonlinear_vars = nonlinear;

    iv_names = names(data[!, r"demand_instruments"]);

    num_fes = 0;

    # Raise error if model is clearly under-identified
    if length(iv_names) < length(nonlinear_vars) + 1
        error("Not enough instruments! FRAC requires one IV for price and for every nonlinear parameter.")
    end

    # Assuming one or two dimensions of FEs
    fe_terms = nothing;
    if fe_names!=[""]
        num_fes = maximum(size(fe_names));
        for i=1:num_fes
            fe_terms += fe(Symbol(fe_names[i]));
        end
    end

    # Construct regressors for variance parameters
    endog_vars = String[];
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

    # Construct regressors for covariance parameters between random coefficients
    # (each pair of nonlinear variables)
    if !isempty(cov)
        for (v1, v2) in cov
           # 1) compute market‐share‐weighted means e_v1, e_v2
           gdf = groupby(data, :market_ids)
        
            gdf = groupby(data, :market_ids)
            cdf1 = compute_combined(gdf, :shares, Symbol(v1))
            cdf2 = compute_combined(gdf, :shares, Symbol(v2))

           data = innerjoin(data, cdf1, on=:market_ids)
           data = innerjoin(data, cdf2, on=:market_ids)

           # 2) build K_{mn} = X_m X_n – e_m X_n – e_n X_m
           reg_name = string("K_cov_", v1, "_", v2)
           data[!, reg_name] =  data[!, v1] .* data[!, v2] .-
                                data[!, Symbol("e_", v1)] .* data[!, v2] .-
                                data[!, Symbol("e_", v2)] .* data[!, v1]
           push!(endog_vars, reg_name)

           # 3) clean up temporary columns
           select!(data, Not([Symbol("e_", v1), Symbol("e_", v2)]))
        end
    end

    push!(endog_vars, "prices")

    problem.data = data;

    return linear_exog_vars, linear_exog_terms, nonlinear_vars, fe_terms, endog_vars, iv_names
end