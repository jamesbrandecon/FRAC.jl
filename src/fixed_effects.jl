"""
pre_absorb(df::DataFrame, linear_vars, nonlinear_vars, iv_names, fe_names, drop_singletons::Bool)

This internal function is used to absorb fixed effects from instruments. 
"""
function pre_absorb(df::DataFrame, linear_vars, nonlinear_vars, iv_names, fe_names, drop_singletons::Bool)
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
    df[!,"y"] = residuals(reg_res);
    for i= 1:length(linear_vars)
        v = linear_vars[i];
        df[!,"v"] = df[!,linear_vars[i]];
        reg_res= reg(df, term(:v) ~ fe_terms, save = :residuals, double_precision = false, drop_singletons = drop_singletons);
        df[!,v] = residuals(reg_res);
    end
    for i= 1:length(nonlinear_vars)
        v = nonlinear_vars[i];
        df[!,"v"] = df[!,nonlinear_vars[i]];
        reg_res= reg(df, term(:v)  ~ fe_terms, save = :residuals, double_precision = false, drop_singletons = drop_singletons);
        df[!,v] = residuals(reg_res);
    end
    for i= 1:length(iv_names)
        v = iv_names[i];
        df[!,"v"] = df[!,iv_names[i]];
        reg_res= reg(df, term(:v) ~ fe_terms, save = :residuals);
        df[!,v] = residuals(reg_res);
    end
    return df
end

""" 
    function get_FEs(problem::FRACProblem)

This is an internal function which is used to estimate fixed effects at constrained estimates. We estimate fixed effects by residualizing the 
outcome variable at the constrained estimates and then regressing the residuals on fixed effects.
"""
function get_FEs!(problem::FRACProblem)
    
    fe_names = problem.fe_names;
    num_fes = maximum(size(fe_names));
    
    # Make FE terms
    fe_terms = nothing;
    if fe_names!=[""]
        num_fes = maximum(size(fe_names));
        for i=1:num_fes
            fe_terms += fe(Symbol(fe_names[i]));
        end
    end

    # Subtract estimated contribution of observables from y so we can estimate implied FEs at constrained estimates 
    data = problem.data;
    data[!,:y_minus_observables] .= data[!,:y];
    for l ∈ problem.linear
        data[!,:y_minus_observables] = data[!,:y_minus_observables] - (data[!,l] ) .* problem.estimated_parameters[Symbol("β_$(l)")];
    end
    for nl ∈ problem.nonlinear
        data[!,:y_minus_observables] = data[!,:y_minus_observables] - (data[!,"K_$(nl)"] )  .* sqrt(max(0, problem.estimated_parameters[Symbol("σ2_$(nl)")]));
    end

    # Regress remaining y on FEs
    results = reg(data, term(:y_minus_observables) ~ fe_terms, 
        save = :all, 
        double_precision = false, 
        drop_singletons = problem.drop_singletons);
    estimated_FEs = fe(results);
    for f ∈ problem.fe_names
        problem.data[:,"estimatedFE_$(f)"] .= estimated_FEs[!,findfirst(problem.fe_names .== f)];
    end   
    data[!,"xi"] .= residuals(results);
end