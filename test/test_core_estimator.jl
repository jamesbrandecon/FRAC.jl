using Test
using Random
using DataFrames
using FRAC

@testset "core estimator recovery" begin
    # One-market, two-product deterministic data
    Random.seed!(100)
    J = 2; T = 100
    beta_true = [-2.0, 1.5]
    sd = [0.0, 0.0]; v = 0.0
    # Simulate data without random coefficients or noise
    s, p, z, x, xi = simulate_logit(J, T, beta_true, sd, v; with_market_FEs=false)
    df_raw = toDataFrame(s, p, z, x, xi)
    df = reshape_pyblp(df_raw)

    # Estimate FRAC with exogenous x, instrumenting only price
    prob = define_problem(
        data = df,
        linear = ["prices", "x"],
        nonlinear = String[],
        fixed_effects = String["market_ids"],
        se_type = "bootstrap",
        constrained = false
    )
    estimate!(prob)
    est = prob.estimated_parameters
    @test isapprox(est[:β_prices], beta_true[1]; atol=0.1)
    @test isapprox(est[:β_x],      beta_true[2]; atol=0.1)
end