using Test
using Random
using DataFrames
using FRAC

@testset "simulate and estimate correlated logit" begin
    # fix seed for reproducibility
    Random.seed!(42)
    # two blocks of products, markets, examples
    J1, J2, T, B = 10, 10, 500, 1
    β = [-2.0, 1.5]
    # true covariance matrix for random coefficients
    Σ = [0.5 0.3;
         0.3 0.8]
    
    # simulate data with no market shocks
    df = FRAC.sim_logit_vary_J(J1, J2, T, B, β, Σ, 0.0)
    df[!,"demand_instruments3"] .= df.demand_instruments0 .* df.demand_instruments1 .* df.demand_instruments2

    # set up and run estimation: random coeffs on prices & x
    problem = define_problem(
        data = df,
        linear = String["prices", "x"],
        nonlinear = ["prices", "x"],
        cov = [("prices","x")],
        fixed_effects = ["product_ids"],
        se_type = "bootstrap",
        constrained = false
    )

    estimate!(problem)

    # extract estimated parameters
    est = problem.estimated_parameters
    β_x = get(est, :β_x, NaN)
    β_prices = get(est, :β_prices, NaN)
    σ2_p = get(est, :σ2_prices, NaN)
    σ2_x = get(est, :σ2_x, NaN)
    σcov_px = get(est, :σcov_prices_x, NaN)

    @test isapprox(β_x, β[2]; atol=0.1)
    @test isapprox(β_prices, β[1]; atol=0.1)
    @test isapprox(σ2_p, Σ[1,1]; atol=0.2)
    @test isapprox(σ2_x, Σ[2,2]; atol=0.2)
    @test isapprox(σcov_px, Σ[1,2]; atol=0.2)
end

@testset "three variable covariance structures" begin
    # fix seed for reproducibility
    Random.seed!(42)
    # two blocks of products, markets, examples
    J1, J2, T, B = 10, 10, 500, 1
    β = [-2.0, 1.5]
    # true covariance matrix for random coefficients
    Σ = [0.5 0.3;
         0.3 0.8]
    
    # simulate data with no market shocks
    df = FRAC.sim_logit_vary_J(J1, J2, T, B, β, Σ, 0.0)

    # add a new continuous variable and one more instrument
    df[!,"x2"] = randn(size(df,1))
    df[!,"demand_instruments3"] .= df.demand_instruments0 .* df.demand_instruments1 .* df.demand_instruments2
    df[!,"demand_instruments4"] .= df.x2 .^2
    df[!,"demand_instruments5"] .= df.x2 .* df.demand_instruments0 .* df.demand_instruments1
    df[!,"demand_instruments6"] .= df.x2 .* df.demand_instruments2

    # (a) all three correlated
    problem3 = define_problem(
        data = df,
        linear = String["prices","x","x2"],
        nonlinear = ["prices","x","x2"],
        cov = [("prices","x"),("prices","x2"),("x","x2")],
        fixed_effects = ["product_ids"],
        se_type = "bootstrap",
        constrained = false
    )
    estimate!(problem3)
    est3 = problem3.estimated_parameters
    @test haskey(est3, :σcov_prices_x2)
    @test haskey(est3, :σcov_x_x2)

    # (b) only prices and x correlated
    problem2 = define_problem(
        data = df,
        linear = String["prices","x","x2"],
        nonlinear = ["prices","x","x2"],
        cov = [("prices","x")],
        fixed_effects = ["product_ids"],
        se_type = "simple",
        constrained = false
    )
    estimate!(problem2)
    est2 = problem2.estimated_parameters

    # extract and test the two‐variable parameters
    β_x      = get(est2, :β_x, NaN)
    β_prices = get(est2, :β_prices, NaN)
    σ2_p     = get(est2, :σ2_prices, NaN)
    σ2_x     = get(est2, :σ2_x, NaN)
    σcov_px  = get(est2, :σcov_prices_x, NaN)

    @test haskey(est2, :β_prices)   && isapprox(β_prices, β[1]; atol=0.1)
    @test haskey(est2, :β_x)        && isapprox(β_x,      β[2]; atol=0.1)
    @test haskey(est2, :σ2_prices)  && isapprox(σ2_p,     Σ[1,1]; atol=0.2)
    @test haskey(est2, :σ2_x)       && isapprox(σ2_x,     Σ[2,2]; atol=0.2)
    @test haskey(est2, :σcov_prices_x) && isapprox(σcov_px, Σ[1,2]; atol=0.2)

    # the covariances with x2 should not exist
    @test !haskey(est2, :σcov_prices_x2)
    @test !haskey(est2, :σcov_x_x2)
end

# new test for cov="all"
@testset "cov=\"all\" option works" begin
    Random.seed!(123)
    J1, J2, T, B = 8, 8, 10, 1
    β = [-1.0, 2.0]
    Σ = [0.4 0.1;
         0.1 0.6]

    # simulate and augment data
    df = FRAC.sim_logit_vary_J(J1, J2, T, B, β, Σ, 0.0)
    df[!,"x2"] = randn(size(df,1))
    df[!,"demand_instruments3"] .= df.demand_instruments0 .* df.demand_instruments1 .* df.demand_instruments2
    df[!,"demand_instruments4"] .= df.x2 .^ 2
    df[!,"demand_instruments5"] .= df.x2 .* df.demand_instruments0 .* df.demand_instruments1
    df[!,"demand_instruments6"] .= df.x2 .* df.demand_instruments2

    problem_all = define_problem(
        data = df,
        linear = String["prices","x","x2"],
        nonlinear = ["prices","x","x2"],
        cov = "all",
        fixed_effects = ["product_ids"],
        se_type = "bootstrap",
        constrained = false
    )
    estimate!(problem_all)
    est_all = problem_all.estimated_parameters

    @test haskey(est_all, :σcov_prices_x)
    @test haskey(est_all, :σcov_prices_x2)
    @test haskey(est_all, :σcov_x_x2)
end
