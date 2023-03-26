function estimate!(problem::FRACProblem)
    constrained = problem.constrained;
    data = problem.data;
    
    iv_names = problem.iv_names;
    fe_terms = problem.fe_terms;
    endog_vars = problem.endog_vars;
    linear_exog_terms = problem.linear_exog_terms;
    
    se = problem.se;
    method = problem.method;
    by_var = problem.by_var;


    if constrained == false
        # Run IV regression(s)
        results = []
        if by_var ==""
            results = reg(data, term(:y) ~ (sum(term.(Symbol.(endog_vars))) ~ sum(term.(Symbol.(iv_names)))) + linear_exog_terms + fe_terms, se, method=method);
        else
            by_var_values = unique(data[!,by_var]);
            p = Progress(length(by_var_values), 1);
            for b = by_var_values
                # oldstd = stdout;
                # redirect_stdout(open("/dev/null", "w"))
                # @suppress begin
                    results_b = reg(data[data[!,by_var] .== b,:], term(:y) ~ (sum(term.(Symbol.(endog_vars))) ~ sum(term.(Symbol.(iv_names)))) + linear_exog_terms + fe_terms, se, method=method);
                # end
                # redirect_stdout(oldstd) # recover origi
                next!(p)
                push!(results, results_b)
            end
        end
    else
        # Estimate via constrained GMM
        results = FRAC_gmm(data = data, 
            linear_vars = problem.linear,
            nonlinear_vars = problem.nonlinear, 
            fe_names = problem.fe_names,
            iv_names = iv_names, 
            by_var = by_var, 
            num_fes = length(problem.fe_names), 
            drop_singletons = true,
            gmm_steps = 2);
    end

    problem.results = results;
end