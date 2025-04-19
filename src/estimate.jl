function estimate!(problem::FRACProblem)
    
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
            problem.estimated_parameters = estimated_param_dict;
            problem.se = estimated_se_dict;
        else
            by_var_values = unique(data[!,by_var]);
            p = Progress(length(by_var_values), 1);
            estimated_param_dict = Dict();
            estimated_se_dict = Dict();
            for f ∈ problem.fe_names
                problem.data[!, "estimatedFE_$(f)"] .= 0.0;
            end
            for b = by_var_values
                # Estimate FRAC regression for each by_var value
                results_b = reg(data[data[!,by_var] .== b,:], 
                    term(:y) ~ (sum(term.(Symbol.(endog_vars))) ~ sum(term.(Symbol.(iv_names)))) + linear_exog_terms + fe_terms, 
                    se, save = :all,
                    drop_singletons = problem.drop_singletons,
                    method = method);
                # Save standard errors in dictionary
                by_val_se_dict = Dict()
                for i ∈ coefnames(results_b)
                    ind = findfirst(coefnames(results_b) .== i)
                    push!(by_val_se_dict, Symbol(i) => sqrt(vcov(results_b)[ind, ind]))
                end
                # Save residuals in data
                problem.data[problem.data[!,by_var] .== b,"xi"] .= residuals(results_b);
                # Save estimated fixed effects in data 
                estimated_FEs = fe(results_b);
                for f ∈ problem.fe_names
                    problem.data[problem.data[!,by_var] .== b,"estimatedFE_$(f)"] .= estimated_FEs[!,findfirst(problem.fe_names .== f)];
                end
                
                next!(p)
                push!(results, results_b)
                # Store estimated parameters in a nested dictionary: each dictionary entry is a by_var value, and each value is a dictionary of estimated parameters
                by_val_dict = Dict();
                for i ∈ problem.linear
                    param_name = Symbol("β_$(i)");
                    index = findfirst(coefnames(results_b) .== i)
                    push!(by_val_dict, param_name => coef(results_b)[index])
                end
                for i∈ problem.nonlinear 
                    param_name = Symbol("σ2_$(i)");
                    index = findfirst(coefnames(results_b) .== "K_$(i)")
                    push!(by_val_dict, param_name => coef(results_b)[index])
                end
                push!(estimated_param_dict, b => by_val_dict)
                push!(estimated_se_dict, b => by_val_se_dict)
            end
            problem.estimated_parameters = estimated_param_dict;
            problem.se = estimated_se_dict;
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