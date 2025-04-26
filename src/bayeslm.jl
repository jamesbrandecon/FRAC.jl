# module HierarchicalBayesReg

# using Turing
# using Distributions
# using LinearAlgebra


@model function hier_reg_model(y, X, group, n_groups, sign_vec)
    n, p = size(X)
    # Priors for hyperparameters
    σ ~ InverseGamma(2, 3) # Error standard deviation
    τ ~ InverseGamma(2, 3) # Group-level coefficient scale (used differently depending on constraint)
    μβ ~ MvNormal(zeros(p), 10.0 * I) # Population mean coefficients

    # Group-level coefficients β (p x n_groups matrix)
    β = Matrix{Real}(undef, p, n_groups) # Hint for shape/type

    # Temporary variables for negative constraints if needed
    # Declared within the model to be properly handled by Turing
    β_abs = Matrix{Real}(undef, p, n_groups)

    # Assign priors to each element β[k, j] based on constraints
    # This version uses HalfCauchy for constrained coefficients.
    # Note: This changes the model interpretation slightly compared to truncated Normal.
    # For constrained coefficients (sign_vec[k] != 0), the prior used here (HalfCauchy)
    # depends only on the scale τ and ignores the population mean μβ[k].
    # The unconstrained coefficients still follow Normal(μβ[k], τ).
    for j in 1:n_groups # Iterate over groups
        for k in 1:p     # Iterate over predictors
            if sign_vec[k] == 1      # Positive constraint
                # Use HalfCauchy distribution with scale τ.
                # This prior is naturally positive.
                β[k, j] ~ LogNormal(μβ[k], τ) # Exponential of Normal
            elseif sign_vec[k] == -1 # Negative constraint
                # Use the negative of a HalfCauchy distribution with scale τ.
                # Sample the absolute value from HalfCauchy, then negate.
                β_abs[k, j] ~ LogNormal(μβ[k], τ) # Exponential of Normal
                β[k, j] = -β_abs[k, j] # Deterministic assignment
            else                     # No constraint
                # Use Normal distribution centered around population mean μβ[k] with scale τ.
                β[k, j] ~ Normal(μβ[k], τ)
            end
        end
    end

    # Likelihood
    # Loop through observations
    for i in 1:n
        # Select the coefficient vector for the group of observation i
        β_group = β[:, group[i]]
        # Calculate the linear predictor
        μ_i = dot(X[i, :], β_group)
        # Define the likelihood for observation i
        y[i] ~ Normal(μ_i, σ)
    end

    # Alternative: Vectorized likelihood (may or may not be faster)
    # This requires calculating the mean for each observation based on its group
    # means = [dot(X[i, :], β[:, group[i]]) for i in 1:n]
    # y ~ MvNormal(means, σ^2 * I) # Note: MvNormal uses variance, not std dev
end

"""
    bayeslm(y, X, group; n_groups = nothing, sign_constraints = [], sampler = NUTS(), nsamples = 1000, nadapt = 500)

Fit a hierarchical Bayesian linear regression model:
- y: response vector of length n.
- X: predictor matrix of size n × p (or a vector of length n for a single predictor).
- group: vector of length n with group labels (any type or ints 1:n_groups).
- sign_constraints: optional vector of (var_index::Int, sign::String) tuples where sign is "positive" or "negative", enforcing coefficient sign constraints.

Returns a MCMCChains.Chains object containing posterior samples.
"""
function bayeslm(y, X, group; n_groups = nothing, sign_constraints = [], sampler = NUTS(), nsamples = 1000, nadapt = 500)
    y = collect(y)
    X = collect(X)
    if ndims(X) == 1
        X = reshape(X, :, 1)
    end
    # number of predictors
    p = size(X, 2)
    # parse sign constraints: vector of length p with 1=positive, -1=negative, 0=unconstrained
    sign_vec = zeros(Int, p)
    for sc in sign_constraints
        var, s = sc
        # ensure valid sign
        if !(s in ("positive", "negative") || s in (:positive, :negative))
            error("sign must be \"positive\" or \"negative\"")
        end
        # variable index
        if !(var isa Int) || var < 1 || var > p
            error("variable index for sign constraint must be an integer between 1 and $p")
        end
        sign_vec[var] = s in ("positive", :positive) ? 1 : -1
    end
    group = collect(group)
    if n_groups === nothing
        uniq = unique(group)
        mapping = Dict(val => idx for (idx, val) in enumerate(uniq))
        group_int = [mapping[g] for g in group]
        n_groups = length(uniq)
    else
        group_int = group
    end
    model = hier_reg_model(y, X, group_int, n_groups, sign_vec)
    samples = sample(model, sampler, nsamples)
    return samples
end