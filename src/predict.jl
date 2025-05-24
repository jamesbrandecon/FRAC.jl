"""
    predict_shares!(data::DataFrame, problem::FRACProblem)

    Predicts the shares of each alternative in the dataset using the estimated parameters, replacing

    # Arguments
    - `data::DataFrame`: DataFrame containing market-level data with columns for each alternative's characteristics, as well as market and product identifiers.
    - `problem::FRACProblem`: The FRACProblem object containing the model specifications and estimated parameters.
    - `run_checks::Bool`: If true, run checks to ensure the model has been estimated and the data has the right columns.

    # Returns
    - `shares`: Dataframe column modified in-place with the predicted shares for each alternative.
"""
function predict_shares!(
    data::DataFrame, 
    problem::FRACProblem, 
    run_checks::Bool=true)

    if run_checks 
        # Check if the model has been estimated
        @assert problem.estimated_parameters != [] "You must estimate the model before predicting shares."

        # Check if the data has the right columns
        required_columns = union(problem.linear, problem.nonlinear, problem.fe_names, ["market_ids", "product_ids"])
        missing_columns = setdiff(required_columns, names(data))
        @assert isempty(missing_columns) "The data is missing the following required columns: $(missing_columns)"
    end

    # Construct Mean utilities 
    deltas = make_deltas(data, problem)

    # 2) draw random coefficients (match your estimation settings)
    I = 50  # or pick problem.n_draws if you store it
    raw = make_draws(data, I, length(problem.nonlinear);
                     method = :halton,    # or :normal
                     common_draws = true,
                     seed = 1234)

    # 3) correlate draws if you have covariances
    if !isempty(problem.cov)
        raw = correlate_draws(raw,
                              data,
                              problem.nonlinear,
                              problem.cov,
                              problem.estimated_parameters)
    end

    # 4) compute predicted shares
    data[!,"shares"] = dropdims(
        shares_from_deltas(
        deltas,
        data;
        monte_carlo_draws = I,
        raw_draws         = raw,
        nonlinear_vars    = problem.nonlinear,
        results           = problem.estimated_parameters
    ), dims=2);
end

function make_deltas(data::DataFrame, problem::FRACProblem)
    # Calculate the mean utilities (deltas) for each alternative
    deltas = zeros(size(data, 1))
    
    for linvar in problem.linear
        deltas .+= data[!, linvar] .* problem.estimated_parameters[Symbol("β_$(linvar)")]
    end
    
    if "xi" in names(data)
        deltas .+= data[!, "xi"];
    else
        # If xi is not in the data, we assume it is zero
        data[!, "xi"] = 0.0;
    end
    
    return deltas
end
