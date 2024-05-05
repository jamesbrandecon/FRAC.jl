# function estimateFRAC(;linear::Vector{String}=["prices"], nonlinear::Vector{String} = [""], 
#     data::DataFrame, fes::Vector{String} = [""],
#     by_var::String="", se_type = "robust", 
#     cluster_var = "", constrained::Bool = false,
#     drop_singletons::Bool = true, gmm_steps = 1, method = :cpu)

#     if linear== ""
#         error("At least one covariate named `prices`` must enter utility linearly")
#     end

#     # Handle standard errors
#     if !(se_type in ["simple", "robust", "cluster"])
#         error("se_type must be either simple, robust, or cluster")
#     end
#     if se_type == "cluster" && cluster_var == ""
#         error("cluster_var must be specified if se_type == 'cluster'")
#     end
#     if se_type == "simple"
#         se = Vcov.simple();
#     elseif se_type == "robust"
#         se = Vcov.robust();
#     elseif se_type == "cluster"
#         se = Vcov.cluster(Symbol(cluster_var));
#     end

#     # Create terms for linear and nonlinear variables, IVs, and FEs 
#     linear_exog_vars, linear_exog_terms, nonlinear_vars, fe_terms, endog_vars, iv_names = make_vars(linear, nonlinear, fes, data);

#     # Check if box constraints desired, and either
#     # (1) Run IV regressions with FEs
#     # (2) Estimate via constrained GMM
#     if constrained == false
#         # Run IV regression(s)
#         results = []
#         if by_var ==""
#             results = reg(data, term(:y) ~ (sum(term.(Symbol.(endog_vars))) ~ sum(term.(Symbol.(iv_names)))) + linear_exog_terms + fe_terms, se, method=method)
#         else
#             by_var_values = unique(data[!,by_var]);
#             p = Progress(length(by_var_values), 1);
#             for b = by_var_values
#                 results_b = reg(data[data[!,by_var] .== b,:], term(:y) ~ (sum(term.(Symbol.(endog_vars))) ~ sum(term.(Symbol.(iv_names)))) + linear_exog_terms + fe_terms, se, method=method)
#                 next!(p)
#                 push!(results, results_b)
#             end
#         end
#     else
#         # Estimate via constrained GMM
#         results = FRAC_gmm(data = data, linear_vars = linear_vars,
#             nonlinear_vars = nonlinear_vars, fe_names = fe_names,
#             iv_names = iv_names, by_var = by_var, num_fes = num_fes, drop_singletons = drop_singletons,
#             gmm_steps = gmm_steps)
#     end

#     return results
# end


# function estimateFRAC(;linear::String="", nonlinear::String, data::DataFrame, fes::String = "",
#     by_var::String="", se_type = "robust", cluster_var = "", constrained::Bool = false,
#     drop_singletons::Bool = true, gmm_steps = 1)

#     # Using :gpu in FixedEffectModels would make everything faster but
#     method = :cpu;

#     # To - do
#     #   Correlation between random coefficients -- annoying because I have to keep e for both vars
#     if linear== ""
#         error("At least one covariate (price) must enter utility linearly")
#     end
#     if !(se_type in ["simple", "robust", "cluster"])
#         error("se_type must be either simple, robust, or cluster")
#     end
#     if se_type == "cluster" && cluster_var == ""
#         error("cluster_var must be specified if se_type == 'cluster'")
#     end
#     if se_type == "simple"
#         se = Vcov.simple();
#     elseif se_type == "robust"
#         se = Vcov.robust();
#     elseif se_type == "cluster"
#         se = Vcov.cluster(Symbol(cluster_var));
#     end

#     # Extract variable names from strings
#     temp_lin = split(linear, "+");
#     linear_vars = replace.(temp_lin, " " => "");
#     if linear == "prices"
#         linear_exog_terms = nothing;
#     else
#         linear_exog_vars = linear_vars[2:end];
#         linear_exog_terms = sum(term.(Symbol.(linear_exog_vars)));
#     end
#     L = maximum(size(linear_vars));

#     temp_nonlin = split(nonlinear, "+");
#     nonlinear_vars = replace.(temp_nonlin, " " => "");

#     iv_names = names(data[!, r"demand_instruments"]);

#     fe_names = "";
#     num_fes = 0;

#     # Raise error if model is under-identified
#     if length(iv_names) < length(nonlinear_vars) + 1
#         error("Not enough instruments! FRAC requires one IV for price and for every nonlinear parameter.")
#     end

#     # Assuming one or two dimensions of FEs
#     fe_terms = nothing;
#     if fes!=""
#         temp_fes = split(fes, "+");
#         fe_names = replace.(temp_fes," " => "");
#         num_fes = maximum(size(fe_names));
#         if num_fes == 2
#             fe_1 = fe_names[1];
#             fe_2 = fe_names[2];
#             fe_terms = fe(Symbol(fe_1)) + fe(Symbol(fe_2));
#         else
#             fe_1 = fe_names[1];
#             fe_terms = fe(Symbol(fe_1));
#         end
#     end

#     # Construct LHS variable
#     gdf = groupby(data, :market_ids);
#     cdf = combine(gdf, :shares => sum);
#     data = innerjoin(data, cdf, on=:market_ids);
#     data[!, "y"] = log.(data[!,"shares"] ./ (1 .- data[!,"shares_sum"]));

#     # Construct regressors for variance parameters
#     endog_vars = [];
#     for i=1:size(nonlinear_vars,1)
#         data[!,"e"] = data[!,nonlinear_vars[i]] .* data[!,"shares"];
#         gdf = groupby(data,:market_ids);
#         cdf = combine(gdf, :e => sum);
#         data = innerjoin(data, cdf, on=:market_ids);

#         # Generate constructed regressor for variances
#         data[!,string("K_", nonlinear_vars[i])] = (data[!,nonlinear_vars[i]] ./2 - data[!,"e_sum"]) .* data[!, nonlinear_vars[i]];
#         push!(endog_vars, string("K_", nonlinear_vars[i]))
#         data = select!(data, Not(:e_sum))
#     end
#     push!(endog_vars, "prices")

#     # Check if box constraints desired, and either
#     # (1) Run IV regressions with FEs
#     # (2) Estimate via constrained GMM
#     if constrained == false
#         # Run IV regression(s)
#         results = []
#         if by_var ==""
#             results = reg(data, term(:y) ~ (sum(term.(Symbol.(endog_vars))) ~ sum(term.(Symbol.(iv_names)))) + linear_exog_terms + fe_terms, se, method=method)
#         else
#             by_var_values = unique(data[!,by_var]);
#             p = Progress(length(by_var_values), 1);
#             for b = by_var_values
#                 results_b = reg(data[data[!,by_var] .== b,:], term(:y) ~ (sum(term.(Symbol.(endog_vars))) ~ sum(term.(Symbol.(iv_names)))) + linear_exog_terms + fe_terms, se, method=method)
#                 next!(p)
#                 push!(results, results_b)
#             end
#         end
#     else
#         results = FRAC_gmm(data = data, linear_vars = linear_vars,
#             nonlinear_vars = nonlinear_vars, fe_names = fe_names,
#             iv_names = iv_names, by_var = by_var, num_fes = num_fes, drop_singletons = drop_singletons,
#             gmm_steps = gmm_steps)
#     end

#     #results = reg(df, @formula(y~ (prices + K_1 ~ demand_instruments0 + demand_instruments0^2)))
#     return results
# end