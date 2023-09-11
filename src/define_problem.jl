"""
    define_problem(; linear =["prices"], nonlinear = [""], 
        data, fixed_effects = [""],
        by_var = "", se_type = "robust", 
        cluster_var = "", constrained::Bool = false,
        drop_singletons = true, gmm_steps = 2, method = :cpu)

This function is used to construct a FRAC problem object. The key inputs are the data, the linear and nonlinear covariates, and the fixed effects, 
all of which are specified as vectors of strings. The user can also specify the type of standard errors to be used, the cluster variable, 
whether the model is constrained, and the number of GMM steps to be used. Presently, `method` is not used. 
"""
function define_problem(;linear::Vector{String}=["prices"], nonlinear::Vector{String} = [""], 
    data::DataFrame, fixed_effects::Vector{String} = [""],
    by_var::String="", se_type = "robust", 
    cluster_var = "", constrained::Bool = false,
    drop_singletons::Bool = true, gmm_steps = 2, method = :cpu)

    sigma_cov = [];
    
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

    linear_exog_vars = []; linear_exog_terms=[]; nonlinear_vars=[]; 
    fe_terms=[]; endog_vars=[]; iv_names=[]; 

    # Covariance terms
    cov = [];

    problem = FRACProblem(data,
            linear, 
            nonlinear,
            cov, 
            by_var,
            fixed_effects,
            iv_names, 
            endog_vars, 
            linear_exog_terms,
            fe_terms,
            se_type, 
            se,
            cluster_var, 
            constrained,
            drop_singletons, 
            gmm_steps, 
            method, 
            [],
            [],
            [],
            [],
            []);
    

    linear_exog_vars, linear_exog_terms, nonlinear_vars, 
    fe_terms, endog_vars, iv_names = make_vars(problem, linear, nonlinear, sigma_cov, fixed_effects, data);

    problem.linear_exog_terms = linear_exog_terms;
    problem.endog_vars = endog_vars;
    problem.fe_terms = fe_terms;
    problem.iv_names = iv_names;

    # Construct LHS variable
    gdf = groupby(data, :market_ids);
    cdf = combine(gdf, :shares => sum);
    data = innerjoin(data, cdf, on=:market_ids);
    problem.data[!, "y"] = log.(data[!,"shares"] ./ (1 .- data[!,"shares_sum"]));

    return problem
end

mutable struct FRACProblem 
    data
    linear 
    nonlinear
    cov 
    by_var
    fe_names
    iv_names 
    endog_vars 
    linear_exog_terms
    fe_terms
    se_type
    se 
    cluster_var 
    constrained
    drop_singletons 
    gmm_steps 
    method
    raw_results_internal
    estimated_parameters
    all_elasticities 
    bootstrapped_parameters_all
    bootstrap_debiased_parameters
end


function Base.show(io::IO, problem::FRACProblem)
    T = size(unique(problem.data[!,:market_ids]));
    J = size(unique(problem.data[!,:product_ids]));

    println(io, "FRAC Problem:")
    println(io, "- Number of total choices: $(J)")
    println(io, "- Number of markets: $(T)")

end

