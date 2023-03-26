# # Get estimated \xi using contraction mapping (maybe adapt price elasticity func to do the same?)

# # Resample xi 

# # for each sample, calculate shares using estimated parameters and sampled \xi 

# # re-estimate parameters via FRAC 

# # adjusted params = 2 * \hat β - 1/B ∑_b β^b 



# function contraction(delta::Vector{<:Real}, s::Vector{Float64}, nubeta::Matrix{<:Real}, 
#     df::DataFrame, param, inner_tol::Float64)
#     delta = reshape(delta, size(s))
#     S = size(nubeta,2);

#     delta_i = repeat(reshape(delta, size(s)), 1, S) + nubeta;

#     # Calculate predicted market shares
#     s_hat = reshape(s_hat_logitRC1(delta_i, eta_i1, df.market_ids), size(delta));
#     s_reshape = reshape(s, size(delta));

#     # Update delta
#     out = delta .+  log.(s_reshape ./ s_hat)
#     return out
# end

# cdf = combine(groupby(df, :market_ids), :shares => sum => :sumshare)
# df = leftjoin(df, cdf, on = :market_ids);
# logs0 = log.(1 .- df.sumshare);

# logit_inv= log.(df.shares) - logs0;
# for k = 1:1:K2
#     nubeta = nubeta + X_i[:,k,:] .* (draws[:,k,:] .* sigma[k])
# end

# delta_1 = squaremJL(x -> contraction(x, df[!,"shares"], nubeta, df, param,
#  inner_tol), logit_inv; tol = outer_tol);
# xi = delta_1 - Matrix(theta .*df[!,linear]);