function estimate!(problem::FRACProblem)
    # Normalize empty `nonlinear` and `fe_names`
    problem.nonlinear = filter(x -> x != "", problem.nonlinear)
    problem.fe_names   = filter(x -> x != "", problem.fe_names)
    
    constrained = problem.constrained;
    data = deepcopy(problem.data);
    
    se_type = problem.se_type
    iv_names = problem.iv_names;
    fe_terms = problem.fe_terms;
    endog_vars = problem.endog_vars;
    linear_exog_terms = problem.linear_exog_terms;
    
    if se_type == "simple"
        se = Vcov.simple();
    elseif se_type == "robust"
        se = Vcov.robust();
    elseif se_type == "cluster"
        se = Vcov.cluster(Symbol(cluster_var));
    else 
        se = Vcov.simple()
    end

    method = problem.method;
    by_var = problem.by_var;

    if "xi" ∉ names(data)
        data[!,"xi"] = zeros(Float64, size(data,1));
    end

    if constrained == false
        # Run IV regression(s)
        results = []
        full_formula = [];
        if by_var ==""
            if !isnothing(linear_exog_terms)
                full_formula = term(:y) ~ (sum(term.(Symbol.(endog_vars))) ~ sum(term.(Symbol.(iv_names)))) + linear_exog_terms + fe_terms
            else
                full_formula = term(:y) ~ (sum(term.(Symbol.(endog_vars))) ~ sum(term.(Symbol.(iv_names)))) + fe_terms
            end
            results = reg(data, full_formula, 
                se, 
                save = :all,
                drop_singletons = problem.drop_singletons, 
                method=method);
            for f ∈ problem.fe_names
                problem.data[!, "estimatedFE_$(f)"] .= 0.0;
            end

            # Save residuals
            problem.data[!,"xi"] .= residuals(results);
            
            # Save estimated fixed effects in data 
            estimated_FEs = fe(results);
            for f ∈ problem.fe_names
                problem.data[:,"estimatedFE_$(f)"] .= estimated_FEs[!,findfirst(problem.fe_names .== f)];
            end

            # Store estimated parameters in a dictionary
            # if problem.se_type ∈ ["robust", "cluster"] then also save regression standard errors
            estimated_param_dict = Dict();
            estimated_se_dict = Dict();
            for i ∈ coefnames(results)
                ind = findfirst(coefnames(results) .== i)
                push!(estimated_se_dict, Symbol(i) => sqrt(vcov(results)[ind, ind]))
            end
            for i ∈ problem.linear
                param_name = Symbol("β_$(i)");
                index = findfirst(coefnames(results) .== i)
                push!(estimated_param_dict, param_name => coef(results)[index])
            end
            for i ∈ problem.nonlinear
                param_name = Symbol("σ2_$(i)");
                index = findfirst(coefnames(results) .== "K_$(i)")
                push!(estimated_param_dict, param_name => coef(results)[index])
            end
            # Covariance parameters for correlated random coefficients
            for (v1, v2) ∈ problem.cov
                if !isempty(v1) && !isempty(v2)
                    # try both orderings
                    cov_names = ["K_cov_$(v1)_$(v2)", "K_cov_$(v2)_$(v1)"]
                    idx = findfirst(name -> name in cov_names, coefnames(results))
                    if idx !== nothing
                        param_sym = Symbol("σcov_$(v1)_$(v2)")
                        push!(estimated_param_dict, param_sym => coef(results)[idx])
                        push!(estimated_se_dict,    param_sym => sqrt(vcov(results)[idx, idx]))
                    end
                end
            end

            problem.estimated_parameters = estimated_param_dict
            problem.se                   = estimated_se_dict

        else
            # Estimate FRAC with heterogeneity via interactions
            # build base endog/iv sums
            endog = sum(term.(Symbol.(endog_vars)))
            iv    = sum(term.(Symbol.(iv_names)))
            # create interaction of each with by_var
            hetero = (endog & term(by_var)) ~ (iv   & term(by_var))
            full_formula = term(:y) ~ (endog ~ iv) + fe_terms + hetero
            if !isnothing(linear_exog_terms)
                full_formula += linear_exog_terms & term(by_var)
            end

            results = reg(data, full_formula,
                se, save = :all,
                drop_singletons = problem.drop_singletons,
                method = method)

            # Save residuals and FEs
            problem.data[!,"xi"] .= residuals(results)
            estimated_FEs = fe(results)
            for f ∈ problem.fe_names
                problem.data[:,"estimatedFE_$(f)"] .=
                    estimated_FEs[!, findfirst(==(f), problem.fe_names)]
            end

            # Extract all coeffs and SEs (including interactions)
            estimated_param_dict = Dict()
            estimated_se_dict    = Dict()
            for cn in coefnames(results)
                idx     = findfirst(==(cn), coefnames(results))
                cval    = coef(results)[idx]
                seval   = sqrt(vcov(results)[idx, idx])

                if occursin(":", cn)
                    # heterogeneity term, e.g. "x:GroupA"
                    parts = split(cn, ":")
                    base  = parts[1]
                    lvl   = parts[2]
                    sym   = Symbol("β_$(base)_by_$(by_var)_$(lvl)")
                elseif startswith(cn, "K_")
                    # variance term
                    sym = Symbol("σ2_$(replace(cn, "K_" => ""))")
                else
                    # plain slope
                    sym = Symbol("β_$(cn)")
                end

                push!(estimated_param_dict, sym => cval)
                push!(estimated_se_dict,    sym => seval)
            end

            problem.estimated_parameters = estimated_param_dict
            problem.se                   = estimated_se_dict
        end
    else
        # Estimate via constrained GMM
        results, xi_hat = FRAC_gmm(data = data, 
            linear_vars = problem.linear,
            nonlinear_vars = problem.nonlinear, 
            fe_names = problem.fe_names,
            iv_names = union(iv_names, setdiff(problem.linear,["prices"])), 
            by_var = by_var, 
            num_fes = length(problem.fe_names), 
            drop_singletons = true,
            gmm_steps = 2);
        
        # problem.data[!,"xi"] .= xi_hat;
        if "xi" ∈ names(problem.data)
            @warn "Overwriting existing xi column in data"
            select!(problem.data, Not(:xi))
        end
        leftjoin!(problem.data, xi_hat, on = [:market_ids, :product_ids])

        estimated_param_dict = Dict();
        for i ∈ eachindex(results[2])
            for j ∈ eachindex(results[2][i])
                index = (i-1)*length(results[2][i]) + j;
                var_name = results[2][i][j];
                if i==1
                    param_name = Symbol("β_$(var_name)");
                    push!(estimated_param_dict, param_name => results[1][index])
                else
                    param_name = Symbol("σ2_$(var_name)");
                    push!(estimated_param_dict, param_name => results[1][index])
                end
            end
        end
        problem.estimated_parameters = estimated_param_dict;

        # Get estimated fixed effects 
        get_FEs!(problem);
        # for f ∈ problem.fe_names
        #     problem.data[!,"estimatedFE_$(f)"] .= estimatedFEs[!,findfirst(problem.fe_names .== f)];
        # end
    end

    problem.raw_results_internal = results;
    
end