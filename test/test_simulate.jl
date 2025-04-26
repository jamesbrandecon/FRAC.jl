using Test
using Random
using DataFrames
using FRAC

@testset "simulate_logit" begin
    Random.seed!(2025)
    J = 2; T = 3
    beta = [-1.0, 0.5]
    sd = [0.0, 0.0]; v = 0.0
    # Without market fixed effects
    s, p, z, x, xi = simulate_logit(J, T, beta, sd, v; with_market_FEs=false)
    @test size(s) == (T, J)
    @test size(p) == (T, J)
    @test size(z) == (T, J)
    @test size(x) == (T, J)
    @test size(xi) == (T, J)
    # Shares in [0,1) and sum less than 1
    for t in 1:T
        @test all(0 .<= s[t, :]) && all(s[t, :] .< 1)
        @test sum(s[t, :]) < 1
    end
    # Reproducibility
    Random.seed!(2025)
    s2, p2, z2, x2, xi2 = simulate_logit(J, T, beta, sd, v; with_market_FEs=false)
    @test s == s2 && p == p2 && z == z2 && x == x2 && xi == xi2
end

@testset "sim_logit_vary_J" begin
    Random.seed!(2025)
    J1 = 2; J2 = 3; T = 4; B = 1
    beta = [0.0, 0.0]; sd = [0.0, 0.0]; v = 0.0
    df = FRAC.sim_logit_vary_J(J1, J2, T, B, beta, sd, v; with_market_FEs=false)
    @test isa(df, DataFrame)
    @test nrow(df) == T*J1 + T*J2
    pid = unique(df.product_ids)
    @test sort(pid) == pid
    @test maximum(pid) == J1 + J2
    @test minimum(pid) == 1
    @test all(df.market_ids .>= 1) && all(df.market_ids .<= 2*T)
    # dummy_FE column exists and is Bool
    @test "dummy_FE" in names(df)
    @test eltype(df.dummy_FE) == Bool
end