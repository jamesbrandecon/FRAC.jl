using Test
using Random
using DataFrames
using FRAC

@testset "Base.show for FRACProblem" begin
    Random.seed!(1)
    # simulate a small dataset
    J1, J2, T, B = 5, 5, 50, 1
    β = [0.5, -0.5]
    Σ = [0.2 0.0;
         0.0 0.3]
    df = FRAC.sim_logit_vary_J(J1, J2, T, B, β, Σ, 0.0)

    # define problem (no instruments needed for this test)
    problem = define_problem(
        data = df,
        linear = String["prices", "x"],
        nonlinear = ["prices", "x"],
        cov = [("prices","x")],
        fixed_effects = String[],
        se_type = "bootstrap",
        constrained = false
    )

    # capture show output
    io = IOBuffer()
    @test show(io, problem) === nothing
    out = String(take!(io))

    @test occursin("FRAC Problem Summary:", out)
    @test occursin("Model Specification:", out)
    @test occursin("Estimation Settings:", out)
    @test occursin("Status: Problem defined, not yet estimated.", out)
end
