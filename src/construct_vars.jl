function make_vars(problem, linear::Vector{String}, nonlinear::Vector{String}, sigma_cov, 
    fe_names::Vector{String}, data::DataFrame)
    
    if sigma_cov !=[]
        @assert issymmetric(sigma_cov)
    end

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

    # Parameters for covariance parameters: NOT YET IMPLEMENTED
    if sigma_cov !=[]
        for i=1:size(nonlinear_vars,1)
            for j=1:size(nonlinear_vars,1)
                if sigma_cov[i,j] !=0 

                end
            end
        end
    end

    push!(endog_vars, "prices")

    problem.data = data;

    return linear_exog_vars, linear_exog_terms, nonlinear_vars, fe_terms, endog_vars, iv_names
end