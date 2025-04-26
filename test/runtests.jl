using Test
using Random
using DataFrames
using FRAC

# Run individual test sets from this directory
include(joinpath(@__DIR__, "test_make_draws.jl"))
include(joinpath(@__DIR__, "test_simulate.jl"))
include(joinpath(@__DIR__, "test_core_estimator.jl"))
include(joinpath(@__DIR__, "test_simulate_estimate_correlated.jl"))
include(joinpath(@__DIR__, "test_show_problem.jl"))
include(joinpath(@__DIR__, "test_price_elasticities.jl"))