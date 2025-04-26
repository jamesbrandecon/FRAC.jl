using Test
using Random
using DataFrames
using FRAC
using Statistics, LinearAlgebra

@testset "make_draws" begin
    # Create a dummy DataFrame with 3 markets, 4 observations per market
    df = DataFrame(market_ids = repeat(1:3, inner=4))
    I = 4; K = 2

    # Deterministic behavior with fixed seed
    Random.seed!(123)
    d1 = FRAC.make_draws(df, I, K; seed=123, method=:normal, antithetic=false, common_draws=false)
    Random.seed!(123)
    d2 = FRAC.make_draws(df, I, K; seed=123, method=:normal, antithetic=false, common_draws=false)
    @test length(d1) == K
    @test d1[1] == d2[1]
    @test d1[2] == d2[2]

    # Common draws: identical across markets
    dd = FRAC.make_draws(df, I, K; seed=1, method=:normal, antithetic=false, common_draws=true)
    markets = sort(unique(df.market_ids))
    first_market_mask = df.market_ids .== markets[1]
    first_block = dd[1][first_market_mask, :]
    for m in markets
        mask = df.market_ids .== m
        @test dd[1][mask, :] == first_block
    end

    # Antithetic sampling pairs (only for even I)
    da = FRAC.make_draws(df, I, 1; seed=42, method=:normal, antithetic=true, common_draws=false)
    mat = da[1]
    half = I ÷ 2
    @test all(mat[:, 1:half] .+ mat[:, half+1:end] .≈ 0.0)

    # Unknown method should error
    @test_throws ErrorException FRAC.make_draws(df, I, K; method=:foo)
end

@testset "correlate_draws" begin
    # Create a dummy DataFrame with 2 markets, 3 observations per market
    df = DataFrame(market_ids = repeat(1:2, inner=3))
    I = 500; K = 2
    # Independent draws
    Random.seed!(1234)
    raw = FRAC.make_draws(df, I, K; seed=1234, method=:normal, antithetic=false, common_draws=false)
    # Specify nonlinear variables and covariance structure
    nonlinear = ["x", "y"]
    cov_pairs = [("x", "y")]
    # Define parameters: unit variances and covariance 0.5
    params = Dict(:σ2_x => 1.0, :σ2_y => 1.0, :σcov_x_y => 0.5)
    # Transform draws
    cor = FRAC.correlate_draws(raw, df, nonlinear, cov_pairs, params)
    # Pick a representative observation (first row)
    idx = 1
    d1 = cor[1][idx, :]
    d2 = cor[2][idx, :]
    # Sample variances should be close to 1, covariance close to 0.5
    @test abs(var(d1) - 1.0) < 0.1
    @test abs(var(d2) - 1.0) < 0.1
    @test abs(cov(d1, d2) - 0.5) < 0.1
end