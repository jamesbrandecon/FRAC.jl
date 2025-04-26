"""
    define_problem(; linear =["prices"], nonlinear = [""],
        cov = "all" or Vector{Tuple{String,String}},  # accept "all" to auto-pair nonlinear
        data, fixed_effects = [""],
        se_type = "bootstrap",
        cluster_var = "", constrained::Bool = false,
        drop_singletons = true, gmm_steps = 2, method = :cpu)

This function is used to construct a FRAC problem object. The key inputs are the data, the linear and nonlinear covariates, and the fixed effects, 
all of which are specified as vectors of strings. The user can also specify the type of standard errors to be used, the cluster variable, 
whether the model is constrained, and the number of GMM steps to be used. Presently, `method` is not used. 
"""
function define_problem(; linear::Vector{String} = ["prices"],
                        nonlinear::Vector{String} = String[],
                        cov::Union{String, Vector{Tuple{String,String}}} = Tuple{String,String}[],
                        data::DataFrame,
                        fixed_effects::Vector{String} = String[],
                        se_type::String = "bootstrap", 
                        cluster_var::String = "",
                        constrained::Bool = false,
                        drop_singletons::Bool = true,
                        gmm_steps::Int = 2,
                        method::Symbol = :cpu)

    # Optional grouping variable (unused here)
    by_var = "";
    # Covariance terms for random coefficients (pairs of variable names)
    # cov should be a vector of (var1,var2) pairs indicating which covariances to estimate

    if cov == "all"
        if isempty(nonlinear) || length(nonlinear) < 2
            # Need at least two nonlinear variables to form a covariance pair
            @warn "cov=\"all\" requires at least two nonlinear covariates to generate covariance terms. No covariance terms generated."
            cov = Tuple{String,String}[] # Ensure cov is the correct type even if empty
        else
            # Generate unique pairs (i, j) where j > i (excludes variances and duplicates like (B,A) if (A,B) exists)
            # e.g., for ["A", "B", "C"], generates [("A", "B"), ("A", "C"), ("B", "C")]
            cov = [(nonlinear[i], nonlinear[j]) for i in 1:length(nonlinear) for j in (i+1):length(nonlinear)]
        end
    end

    if linear== ""
        error("At least one covariate (price) must enter utility linearly")
    end
    if se_type!= "bootstrap"
        @warn "While other standard errors are supported for unconstrained problems, I advise using bootstrapped standard errors. `robust` and `cluster` standard errors seem to be biased downward."
    end
    if !(se_type in ["simple", "robust", "cluster", "bootstrap"])
        error("se_type must be either simple, robust, cluster, or bootstrap")
    end
    if se_type == "cluster" && cluster_var == ""
        error("cluster_var must be specified if se_type == 'cluster'")
    end

    linear_exog_vars = []; linear_exog_terms=[]; nonlinear_vars=[]; 
    fe_terms=[]; endog_vars=[]; iv_names=[]; 

    # Covariance terms
    # cov = [("", "")];

    data = sort(data, [:market_ids, :product_ids]);

    # Initialize FRACProblem; 'cov' stored for covariance regressors
    problem = FRACProblem(
        data,
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
        [],  # se placeholder
        cluster_var,
        constrained,
        drop_singletons,
        gmm_steps,
        method,
        [], [], [], [], []
    );
    

    # Populate regression variables, including variance and covariance constructs
    linear_exog_vars, linear_exog_terms, nonlinear_vars,
    fe_terms, endog_vars, iv_names = make_vars(problem, linear, nonlinear, cov, fixed_effects, data);

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
    T = length(unique(problem.data[!,:market_ids]));
    N = nrow(problem.data);
    # Calculate average number of products per market
    avg_J = N / T;

    println(io, "FRAC Problem Summary:")
    println(io, "---------------------")
    println(io, "- Total Observations (N): $(N)")
    println(io, "- Number of Markets (T): $(T)")
    println(io, "- Average Products per Market (N/T): $(round(avg_J, digits=2))")
    println(io, "")
    println(io, "Model Specification:")
    println(io, "- Linear Covariates: ", isempty(problem.linear) ? "None" : join(problem.linear, ", "))
    println(io, "- Nonlinear Covariates (Random Coefficients): ", isempty(problem.nonlinear) ? "None" : join(problem.nonlinear, ", "))
    if !isempty(problem.cov)
        cov_str = join(["($(p[1]), $(p[2]))" for p in problem.cov], ", ")
        println(io, "- Covariance Terms: ", cov_str)
    else
        println(io, "- Covariance Terms: None")
    end
    println(io, "- Fixed Effects: ", isempty(problem.fe_names) ? "None" : join(problem.fe_names, ", "))
    println(io, "- Endogenous Variables: ", isempty(problem.endog_vars) ? "None" : join(problem.endog_vars, ", "))
    # println(io, "- IV Names: ", isempty(problem.iv_names) ? "Not specified/generated yet" : join(problem.iv_names, ", ")) # IVs might be generated later
    println(io, "")
    println(io, "Estimation Settings:")
    println(io, "- Standard Error Type: $(problem.se_type)")
    if problem.se_type == "cluster"
        println(io, "- Cluster Variable: $(problem.cluster_var)")
    end
    println(io, "- Constrained Estimation: $(problem.constrained)")
    println(io, "- Drop Singletons: $(problem.drop_singletons)")
    println(io, "- GMM Steps: $(problem.gmm_steps)")
    # println(io, "- Method: $(problem.method)") # Currently unused

    # Optionally, indicate if results are available
    if !isempty(problem.estimated_parameters)
        println(io, "")
        println(io, "Status: Estimation results available.")
    else
        println(io, "")
        println(io, "Status: Problem defined, not yet estimated.")
    end
end

