"""
pre_absorb(df::DataFrame, linear_vars, nonlinear_vars, iv_names, fe_names, drop_singletons::Bool)

This internal function is used to absoeb fixed effects from instruments. 
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