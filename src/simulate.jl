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
        df2 = reshape_pyblp(toDataFrame(s2,p2,z2,x2, xi2));
    end

    df2[!,"product_ids"] = df2.product_ids .+ 2;
    df2[!,"market_ids"] = df2.market_ids .+ T .+1;
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

    df[!,"dummy_FE"] .= rand();
    df[!,"dummy_FE"] = (df.dummy_FE .> 0.5);

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
    
        if minimum(size(sd)) != 1
            sd = cholesky(sd).U;
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
            I = 1000;
            if minimum(size(sd)) != 1
                beta_i = randn(I,2) * sd;
            else
                beta_i = randn(I,2) .* sd;
            end
            beta_i = beta_i .+ beta;
            denominator = ones(I,1);
            for j = 1:1:J
                denominator = denominator .+ exp.(beta_i[:,1].*pt[t,j] + beta_i[:,2].*xt[t,j] .+ market_FE .+ xit[t,j]);
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