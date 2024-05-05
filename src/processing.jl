function extractEstimates(df;frac_results = [], var = "prices", param = "mean", by_var = "")
    if typeof(frac_results[1]) == FixedEffectModel
        B = maximum(size(unique(df[:,"$by_var"])));
        param_vec = zeros(B);
        se_vec = zeros(B);
        by_vec = zeros(B);
        for i = 1:B
            b = unique(df[:,"$by_var"])[i];
            if param == "mean"
                ind = findall(x-> x == string(var), coefnames(frac_results[i]));
            else
                ind = findall(x-> x == string("K_$var"), coefnames(frac_results[i]));
            end
            param_vec[i] = coef(frac_results[i])[ind][1]
            se_vec[i] = vcov(frac_results[i])[ind][1]
            by_vec[i] = unique(df[:,"$by_var"])[i];
        end
    elseif typeof(frac_results[end][1]) <: Array{String}
        # When running constrained GMM, will return string array of
        # variable names as the last element
        B = maximum(size(unique(df[:,"$by_var"])));
        param_vec = zeros(B);
        se_vec = [];
        by_vec = zeros(B);
        num_lin = length(frac_results[end][1]);
        for i = 1:B
            if param == "mean"
                ind = findall(x-> x== string(var), frac_results[end][1]); # search for linear vars
                param_vec[i] = frac_results[i][ind][1];
            else
                num_lin = length(frac_results[end][1]);
                ind = findall(x-> x== string(var), frac_results[end][2])[1] + num_lin;
                param_vec[i] = frac_results[i][ind];
            end
            by_vec[i] = unique(df[:,"$by_var"])[i];
        end
    else
        ArgumentError("Provided FRAC results are not an array of FixedEffectModels nor a string array of variable names")
    end
    return param_vec, se_vec, by_vec
end
    
function plotFRACResults(df;frac_results = [], var = "prices", param = "mean", plot = "hist", by_var = "")
    # Extract parameter estimates from frac_results
    param_vec, se_vec, by_vals = extractEstimates(df, frac_results = frac_results,
            param = param, by_var = by_var, var = var);

    if plot =="hist"
        return histogram(param_vec, leg=false, xaxis = "Estimates", yaxis = "Frequency", alpha = 0.4)
    elseif plot=="scatter"
        if se_vec ==[]
            return scatter(by_vals, param_vec, leg=false, xaxis = "Models", yaxis = "Estimates")
        else
            return scatter(by_vals, param_vec, leg=false, xaxis = "Models", yaxis = "Estimates", yerror = 1.96 .*se_vec)
        end
    end
end