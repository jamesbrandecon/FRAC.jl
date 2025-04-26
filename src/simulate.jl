function sim_logit_vary_J(J1, J2, T, B, beta, sd, v; with_market_FEs = false)
    if with_market_FEs
        s,p,z,x,xi,marketFE = simulate_logit(J1,T,beta, sd, v, with_market_FEs = with_market_FEs);
        s2,p2,z2,x2,xi2,marketFE2 = simulate_logit(J2,T,beta, sd, v, with_market_FEs = with_market_FEs);
        
        # Reshape data into desired DataFrame, add necessary IVs
        df = reshape_pyblp(toDataFrame(s,p,z,x,xi,marketFE));
        df2 = reshape_pyblp(toDataFrame(s2,p2,z2,x2,xi2,marketFE2));
    else
        s,p,z,x, xi = simulate_logit(J1,T,beta, sd, v);
        s2,p2,z2,x2, xi2 = simulate_logit(J2,T,beta, sd, v);

        # Reshape data into desired DataFrame, add necessary IVs
        df = reshape_pyblp(toDataFrame(s,p,z,x,xi));
        df2 = reshape_pyblp(toDataFrame(s2,p2,z2,x2,xi2));
    end

    df2[!,"product_ids"] = df2.product_ids .+ 2;
    # Shift second block's market IDs to continue after first T markets
    df2[!,"market_ids"] = df2.market_ids .+ T;
    df = [df;df2]
    df[!,"by_example"] = mod.(1:size(df,1),B); # Generate variable indicating separate geographies
    df[!,"demand_instruments1"] = df.demand_instruments0.^2;
    df[!,"demand_instruments2"] = df.x .^2;

    # Add simple differentiation-style IVs: difference from market-level sum
    # gdf = groupby(df, :market_ids);
    # cdf = combine(gdf, names(df) .=> sum);
    # cdf = select(cdf, Not(:market_ids_sum));
    # sums = innerjoin(select(df, :market_ids), cdf, on = :market_ids);
    # df[!,"demand_instruments3"] = (df.demand_instruments0 - sums.demand_instruments0_sum).^2;
    # df[!,"demand_instruments4"] = (df.x - sums.x_sum).^2;

    # Add a dummy fixed-effect column (symbol key ensures column name is a Symbol)
    df[!, :dummy_FE] = rand(Bool, nrow(df))

    return df
end

function simulate_logit(J,T, beta, sd, v; with_market_FEs = false)
    # --------------------------------------------------%
    # Simulate mixed logit with outside option
    # --------------------------------------------------%
    # J = number of products
    # T = number of markets
    # beta = coefficients on price and x
    # sd = standard deviations of preferences on price and x
    # v = standard deviation of market-product demand shock
    
    # Shortcut for pure logit when no random coefficients and no noise
    if all(sd .== 0) && v == 0
        # draw exogenous variables
        zt = rand(T, J)
        xit = zeros(T, J)
        pt = 2 .* rand(T, J)
        xt = 2 .* rand(T, J)
        # compute deterministic logit shares
        s = zeros(T, J)
        for t in 1:T
            U = exp.(beta[1] .* pt[t, :] .+ beta[2] .* xt[t, :] .+ xit[t, :])
            denom = 1 + sum(U)
            s[t, :] = U ./ denom
        end
        if with_market_FEs
            market_FEs = zeros(T)
            return s, pt, zt, xt, xit, market_FEs
        else
            return s, pt, zt, xt, xit
        end
    end
    # If sd is a covariance matrix, extract upper-triangular factor
    if isa(sd, AbstractMatrix)
        sd = cholesky(sd).U
    end
    
        zt = 0.9 .* rand(T,J) .+ 0.05;
        xit = randn(T,J).*v;
        pt = 2 .*(zt .+ rand(T,J)*0.1) .+ xit;
        xt = 2 .* rand(T,J);
    
        # Loop over markets
        s = zeros(T,J);
        market_FEs = zeros(T);
        
        for t = 1:1:T
            if with_market_FEs == true
                market_FE = t/T;
                market_FEs[t] = market_FE;
            else
                market_FE = 0;
            end
            R = 1000
            K = length(beta)
            # Draw individual coefficients: always produce R×K
            if isa(sd, AbstractMatrix)
                beta_i = randn(R, K) * sd
            else
                sd_mat = repeat(reshape(sd, 1, K), R, 1)
                beta_i = randn(R, K) .* sd_mat
            end
            # add population mean
            beta_i .+= reshape(beta, 1, K)
            denominator = ones(R)
            for j = 1:J
                denominator .+= exp.(beta_i[:,1] .* pt[t,j] .+ beta_i[:,2] .* xt[t,j] .+ market_FE .+ xit[t,j])
            end
            for j = 1:1:J
                s[t,j] = mean(exp.(beta_i[:,1].* pt[t,j] + beta_i[:,2].*xt[t,j] .+ market_FE .+ xit[t,j])./denominator);
            end
        end
        if with_market_FEs == true
            return s, pt, zt, xt, xit, market_FEs
        else
            return s, pt, zt, xt, xit
        end
end

function toDataFrame(s::Matrix, p::Matrix, z::Matrix, x::Matrix = zeros(size(s)), xi::Matrix = zeros(size(s)), marketFE = zeros(size(s,1)))
    df = DataFrame();
    J = size(s,2);
    for j = 0:J-1
        df[!, "shares$j"] =  s[:,j+1];
        df[!, "prices$j"] =  p[:,j+1];
        df[!, "demand_instruments$j"] =  z[:,j+1];
        df[!, "x$j"] =  x[:,j+1];
        df[!, "xi$j"] =  xi[:,j+1];
    end
    
    if marketFE != zeros(size(s,1))
        df[!,"market_FEs"] .= marketFE;
    end
    return df;
end

function reshape_pyblp(df::DataFrame; random_constant = false)
    df.market_ids = 1:size(df,1);
    shares = Matrix(df[!,r"shares"]);

    market_ids = df[!, "market_ids"];
    market_ids = repeat(market_ids, size(shares,2));
    
    try 
        market_FEs = df[!, "market_FEs"];
        market_FEs = repeat(market_FEs, size(shares,2));
    catch 
    end

    product_ids = repeat((1:size(shares,2))', size(df,1),1);
    product_ids = dropdims(reshape(product_ids, size(df,1)*size(shares,2),1), dims=2);

    shares = dropdims(reshape(shares, size(df,1)*size(shares,2),1), dims=2);

    prices = Matrix(df[!,r"prices"]);
    prices = dropdims(reshape(prices, size(df,1)*size(prices,2),1), dims=2);

    xis = Matrix(df[!,r"xi"]);
    xis = dropdims(reshape(xis, size(df,1)*size(xis,2),1), dims=2);

    xs = Matrix(df[!,r"x\d"]);
    xs = dropdims(reshape(xs, size(df,1)*size(xs,2),1), dims=2);

    demand_instruments0 = Matrix(df[!,r"demand_instruments"]);

    if random_constant ==true
        demand_instruments1 = dropdims(repeat(sum(demand_instruments0, dims=2), size(demand_instruments0,2) ,1), dims = 2);
    end

    demand_instruments0 = dropdims(reshape(demand_instruments0, size(df,1)*size(demand_instruments0,2),1), dims=2);
    
    new_df = DataFrame();
    new_df[!,"shares"] = shares;
    new_df[!,"prices"] = prices;
    new_df[!,"product_ids"] = product_ids;
    new_df[!,"x"] = xs;
    new_df[!,"xi"] = xis;

    new_df[!,"demand_instruments0"] = demand_instruments0;
    if random_constant ==true
        new_df[!,"demand_instruments1"] = demand_instruments1;
    end
    new_df[!,"market_ids"] = market_ids;
    try 
        new_df[!,"market_FEs"] = market_FEs;
    catch 
    end
    return new_df
end

# Core function: Compute simulated-market logit shares given pre-drawn coefficients.
# This function takes coefficient draws as input and is designed to be differentiated by ForwardDiff.
function _sim_market_share_impl(p::AbstractVector, x::AbstractVector, xi::AbstractVector,
                                beta_draws::AbstractMatrix; # K x R matrix of draws (K parameters, R draws)
                                market_FE::Real=0)
    K, R = size(beta_draws) # K = number of parameters, R = number of draws
    J = length(p)          # Number of products
    
    # Check if dimensions match
    if K < 2
        # This implementation assumes at least beta[1] for price and beta[2] for x.
        # Adjust logic if fewer random coefficients are expected or handled differently.
        error("beta_draws must have at least 2 rows (for price and x coefficients). Found K=$K.")
    end
    if length(x) != J || length(xi) != J
        error("Input vectors p, x, xi must have the same length J.")
    end

    # Pre-allocate matrix for exponentiated utilities for each product and draw
    # exp_utility_draws[j, r] = exp(utility of product j for draw r)
    exp_utility_draws = zeros(eltype(p), J, R)

    # Calculate exponentiated utilities for all products for each draw
    for r in 1:R
        βi = view(beta_draws, :, r) # Get the r-th draw (K x 1 vector)
        # Assumes βi[1] is coefficient for price (p), βi[2] is for x
        exp_utility_draws[:, r] .= exp.(βi[1] .* p .+ βi[2] .* x .+ xi .+ market_FE)
    end

    # Calculate denominator for each draw: 1 (outside option) + sum of exp(utilities)
    denominator_draws = 1.0 .+ sum(exp_utility_draws, dims=1) # Results in a 1 x R vector

    # Calculate shares for each product by averaging over draws
    # s_j = mean over r [ exp(u_jr) / (1 + sum_k exp(u_kr)) ]
    # Element-wise division broadcasts denominator_draws correctly
    avg_shares = mean(exp_utility_draws ./ denominator_draws, dims=2) # Average across columns (draws)

    return vec(avg_shares) # Return as a J-element vector
end


# Wrapper function: Generates random coefficient draws and calls the implementation function.
# Useful if only simulated shares are needed for a single market condition.
function sim_market_share(p::AbstractVector, x::AbstractVector, xi::AbstractVector,
                          beta::AbstractVector, Σ::AbstractMatrix;
                          market_FE::Real=0, R::Int=500)
    # Ensure Σ is valid for MvNormal (e.g., positive semi-definite)
    # Distributions.jl typically handles PSD checks. Add jitter if needed:
    # Σ_adj = Σ + Diagonal(fill(eps(Float64), length(beta))) * 1e-9
    mvn = MvNormal(beta, Σ)
    beta_draws = rand(mvn, R) # Draw R times, result is K x R matrix
    return _sim_market_share_impl(p, x, xi, beta_draws; market_FE=market_FE)
end


# Compute price elasticities ∂s_i/∂p_j * p_j/s_i using ForwardDiff.
# Generates random coefficient draws ONCE and reuses them for base shares and Jacobian calculation
# to ensure consistency as required by differentiation.
function sim_price_elasticities(p::AbstractVector, x::AbstractVector, xi::AbstractVector,
                                beta::AbstractVector, Σ::AbstractMatrix;
                                market_FE::Real=0, R::Int=500)
    # Generate random draws ONCE
    # Σ_adj = Σ + Diagonal(fill(eps(Float64), length(beta))) * 1e-9 # Add jitter if needed
    mvn = MvNormal(beta, Σ)
    beta_draws = rand(mvn, R) # Draw R times, result is K x R matrix

    # Define the function to differentiate: calculates shares for a given price vector 'price_vec'
    # using the fixed, pre-generated beta_draws.
    share_func = price_vec -> _sim_market_share_impl(price_vec, x, xi, beta_draws; market_FE=market_FE)

    # Calculate base shares using the generated draws and the original prices 'p'
    s = share_func(p)

    # Calculate Jacobian (matrix of partial derivatives ∂s_i/∂p_j) using ForwardDiff
    # ForwardDiff computes jacobian(f, x) where f maps R^n -> R^m, result is m x n
    # Here, f maps R^J -> R^J, so jac is J x J. jac[i, j] = ∂s_i / ∂p_j
    jac = ForwardDiff.jacobian(share_func, p)

    # Ensure non-zero shares to avoid division by zero in elasticity calculation
    # Replace zero or very small shares with a small positive number
    s_safe = max.(s, eps(eltype(s)))

    # Calculate elasticities: E[i, j] = (∂s_i / ∂p_j) * (p_j / s_i)
    # Use broadcasting: jac is JxJ, p' is 1xJ, s_safe is Jx1
    # (p' ./ s_safe) creates a JxJ matrix where element [i,j] is p[j]/s_safe[i]
    elasticities = jac .* (p' ./ s_safe)

    return elasticities
end

# True elasticities over a DataFrame
function sim_true_price_elasticities(df::DataFrame, beta::AbstractVector, Σ::AbstractMatrix;
                                     price_col::Symbol=:prices,
                                     x_col::Symbol=:x,
                                     xi_col::Symbol=:xi)
    out = DataFrame(market_ids=Int[], product_i=Int[], product_j=Int[], elasticity=Float64[])
    for subdf in groupby(df, :market_ids)
        p  = subdf[!, price_col]
        x  = subdf[!, x_col]
        xi = subdf[!, xi_col]
        fe = hasproperty(subdf, :market_FEs) ? first(subdf.market_FEs) : 0
        E  = sim_price_elasticities(p, x, xi, beta, Σ; market_FE=fe)
        mid, J = first(subdf.market_ids), length(p)
        for i in 1:J, j in 1:J
            push!(out, (market_ids=mid, product_i=i, product_j=j, elasticity=E[i,j]))
        end
    end
    return out
end