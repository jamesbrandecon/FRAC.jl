module FRACDemand

using DataFrames, FixedEffectModels, LinearAlgebra
using Statistics, Plots, Optim, Missings
using ProgressMeter, Suppressor, Random, Distributions, Primes
using ForwardDiff, QuasiMonteCarlo

# Add helpful extension of addition function
    # This allows us to drop groups of terms from regression when irrelevant
    # by setting them to `nothing`

    # I've tried to define the addition functions as specifically as necessary
    # but these may cause issues in ways I cannot predict.
import Base.+

function +(a::Nothing, b::FixedEffectModels.Term)
    return b;
end

function +(a::Nothing,b::FixedEffectModels.AbstractTerm)
    return b;
end
function +(a::FixedEffectModels.AbstractTerm,b::Nothing)
    return a;
end

function +(a::Tuple,b::Nothing)
    return a;
end
function +(a::Nothing,b::Tuple)
    return b;
end
include("define_problem.jl")
include("estimate.jl")
include("construct_vars.jl")
include("fixed_effects.jl")
include("constrained.jl")
include("bootstrap.jl")

include("processing.jl")
include("simulate.jl")
include("elasticities.jl")
include("predict.jl")

export estimateFRAC, plotFRACResults, extractEstimates, simulate_logit, toDataFrame, reshape_pyblp,
    define_problem, FRACProblem, estimate!, bootstrap!, get_FEs!, price_elasticities!,
    inner_price_elasticities, all_elasticities, get_elasticities, own_elasticities,
    correlate_draws, HaltonDraws!, HaltonSeq!, pre_absorb, predict_shares!
end
