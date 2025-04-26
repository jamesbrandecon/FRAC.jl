using Test
using Random
using DataFrames
using FRAC

@testset "price_elasticities! runs with non-empty cov" begin
    Random.seed!(123)
    # simulate data with two random coefficients (price & x) and a covariance
    J1, J2, T, B = 5, 5, 100, 1
    β = [-1.0, 2.0]
    Σ = [0.4 0.2;
         0.2 0.6]
    df = FRAC.sim_logit_vary_J(J1, J2, T, B, β, Σ, 0.0)
    # add one extra instrument so define_problem doesn't error
    df[!,"demand_instruments3"] .= df.demand_instruments0 .* df.demand_instruments1 .* df.demand_instruments2

    # define with a non-empty cov pair
    problem = define_problem(
        data = df,
        linear = ["prices", "x"],
        nonlinear = ["prices", "x"],
        cov = [("prices","x")],
        fixed_effects = ["product_ids"],
        se_type = "bootstrap",
        constrained = false
    )

    # estimate parameters and then compute elasticities
    estimate!(problem)
    @test isempty(problem.all_elasticities)

    # now run price_elasticities! explicitly
    price_elasticities!(problem)
    @test isempty(problem.all_elasticities) == false
    # check structure: one entry per market
    @test length(problem.all_elasticities) == length(unique(df.market_ids))
end
