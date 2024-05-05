function FRAC_gmm(;data::DataFrame, linear_vars::Vector{String} = [""], 
    nonlinear_vars::Vector{String} = [""], fe_names::Vector{String} = [""], 
    iv_names::Vector{String} = [""], by_var = "", num_fes = 0, drop_singletons = true,
    gmm_steps = 1, RETURN_XI = true)

    num_linear = maximum(size(linear_vars));
    num_nonlinear = maximum(size(nonlinear_vars));
    price_ind = 0;
    results = [];

    if RETURN_XI 
        # Initialize dataframe with market_ids, to store xi
        xi_storage = DataFrame(xi = zeros(size(data,1)), 
            product_ids = data.product_ids, 
            market_ids = data.market_ids);
    end

    if by_var == ""
        global price_ind, results
        if num_fes > 0
            data = pre_absorb(data, linear_vars, nonlinear_vars, iv_names, fe_names, drop_singletons);
        end

        # When reg() drops singletons, some residuals are missing
        # so I need to drop any rows w/ missing values of anything residualized
        for residual_var = linear_vars
            data = data[.! isequal.(Array(data[!,residual_var]), missing),:];
        end
        for residual_var = nonlinear_vars
            data = data[.! isequal.(Array(data[!,string("K_",residual_var)]), missing),:];
        end
        for residual_var = iv_names
            data = data[.! isequal.(Array(data[!,residual_var]), missing),:];
        end

        # lower is -INF for all linear
            # is zero for all K_
        # upper is INF for all except prices
        lower = convert(AbstractArray,zeros(num_linear + num_nonlinear));
        upper = convert(AbstractArray,Inf .* ones(num_linear + num_nonlinear));
        for i = 1:num_linear
            lower[i] = -Inf;
            if linear_vars[i] == "prices"
                upper[i] = 0; # upper bound α at 0
                price_ind = i;
            end
        end
        for i = 1:num_nonlinear
            lower[i + num_linear] = 0; # lower bound all σ^2 at 0
        end

        initial_param = 0.1 .* ones(num_linear + num_nonlinear);
        initial_param[price_ind] = -1 .* initial_param[price_ind];

        results = optimize(x-> gmm_obj(x; data = data, linear_vars = linear_vars,
            nonlinear_vars = nonlinear_vars, iv_names = iv_names),
            lower, upper, initial_param, Fminbox(); autodiff = :finite);

        results = Optim.minimizer(results);

        if gmm_steps == 2
            initial_param = results;
            moments = gmm_obj(initial_param; data = data, linear_vars = linear_vars,
                nonlinear_vars = nonlinear_vars, iv_names = iv_names, step = 1.5);
            W = inv(moments'moments / size(moments,1));
            # Second GMM step
            results = optimize(x-> gmm_obj(x; data = data, linear_vars = linear_vars,
                nonlinear_vars = nonlinear_vars, iv_names = iv_names, W = W, step = 2),
                lower, upper, initial_param, Fminbox(); autodiff = :finite);
            results = Optim.minimizer(results);
        end

        # Evaluate gmm_obj at final results, to get xi
        if RETURN_XI
            xi = gmm_obj(results; data = data, linear_vars = linear_vars,
                nonlinear_vars = nonlinear_vars, iv_names = iv_names, RETURN_XI = RETURN_XI);
            xi_storage[!,"xi"] = xi;
        end

        varnames = linear_vars,nonlinear_vars
        results = [results, varnames]
    else
        by_var_values = unique(data[!,by_var]);
        p = Progress(length(by_var_values),1);
        for b = by_var_values
            global price_ind, results
            data_b = data[data[!,by_var] .== b,:];
            # Absorb fixed effects from covariates
            if num_fes > 0
                data_b = pre_absorb(data_b, linear_vars, nonlinear_vars, iv_names, fe_names, drop_singletons);
            end

            # When reg() drops singletons, some residuals are missing
            # so I need to drop any rows w/ missing values of anything residualized
            for residual_var = linear_vars
                data_b = data_b[.! isequal.(Array(data_b[!,residual_var]), missing),:];
            end
            for residual_var = nonlinear_vars
                data_b = data_b[.! isequal.(Array(data_b[!,string("K_",residual_var)]), missing),:];
            end
            for residual_var = iv_names
                data_b = data_b[.! isequal.(Array(data_b[!,residual_var]), missing),:];
            end

            # lower is -INF for all linear
                # is zero for all K_
            # upper is INF for all except prices
            lower = convert(AbstractArray,zeros(num_linear + num_nonlinear));
            upper = convert(AbstractArray,Inf .* ones(num_linear + num_nonlinear));
            for i = 1:num_linear
                lower[i] = -Inf;
                if linear_vars[i] == "prices"
                    upper[i] = 0; # upper bound α at 0
                    price_ind = i;
                end
            end
            for i = 1:num_nonlinear
                lower[i + num_linear] = 0; # lower bound all σ^2 at 0
            end

            initial_param = 0.1 .* ones(num_linear + num_nonlinear);
            initial_param[price_ind] = -0.1 .* initial_param[price_ind];

            results_b = optimize(x-> gmm_obj(x; data = data_b, linear_vars = linear_vars,
                nonlinear_vars = nonlinear_vars, iv_names = iv_names),
                lower, upper, initial_param, Fminbox(), autodiff = :forward);
            results_b = Optim.minimizer(results_b);

              # Evaluate gmm_obj at final results, to get xi
            if RETURN_XI
                xi = gmm_obj(results; data = data_b, linear_vars = linear_vars,
                    nonlinear_vars = nonlinear_vars, iv_names = iv_names, RETURN_XI = RETURN_XI);
                xi_storage[data[!,by_var] .== b,"xi"] = xi;
            end

            push!(results,results_b)
            next!(p) # update progress meter
        end
        varnames = linear_vars,nonlinear_vars
        push!(results, varnames)
    end
    return results, xi_storage
end

function gmm_obj(params; data::DataFrame, linear_vars = "", nonlinear_vars = "",
    iv_names = "", W = inv(Array(data[!,iv_names])'Array(data[!,iv_names])), step = 1,
    RETURN_XI = false)
    # xi =  y - beta * linear_vars - Sigma * K
    # moments = xi * iv_names; (n * k)
    # gmm_obj = moments' * moments;

    xi = data[!,"y"]; # initialize xi
    # Z = Array(data[!,iv_names]);
    num_linear = maximum(size(linear_vars));
    for lin ∈ linear_vars
        i = findall(linear_vars .== lin)[1];
        xi = xi - params[i] .* data[!,lin];
    end

    for nonlin ∈ nonlinear_vars
        i = findall(nonlinear_vars .== nonlin)[1];
        xi = xi - params[i+num_linear] .* data[!,string("K_",nonlin)];
    end

    moments = mean(xi .* Array(data[!,iv_names]), dims=1);

    if (step in [1,2])
        obj = moments * W * moments';
    else
        obj = xi .* Array(data[!,iv_names]);
    end
    if RETURN_XI
        return xi;
    else
        return obj[1];
    end
end
