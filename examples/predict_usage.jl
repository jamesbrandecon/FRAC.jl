using FRACDemand, DataFrames, Random

# 1) Simulate a small dataset
Random.seed!(123)
df = FRACDemand.sim_logit_vary_J(
    3, 3, 200, 1,
    [-2.0, 1.5],
    [0.1 0.0; 0.0 0.1],
    0.5
)
sort!(df, [:market_ids, :product_ids])

# 2) Define and estimate the problem
problem = define_problem(
    data          = df,
    linear        = ["prices","x"],
    nonlinear     = ["prices","x"],
    fixed_effects = ["market_ids", "product_ids"],
    se_type       = "bootstrap",
    constrained   = false
)
estimate!(problem)

# 3) Predict baseline shares
predict_shares!(df, problem)

# 4) Create counterfactual: +10% prices
df_cf = deepcopy(df)
df_cf.prices .*= 1.10

# 5) Predict counterfactual shares
predict_shares!(df_cf, problem)

# 6) Display a few rows
println("Baseline shares:")
first(select(df, [:market_ids, :product_ids, :shares]), 6) |> println
println("Counterfactual (+10% prices) shares:")
first(select(df_cf, [:market_ids, :product_ids, :shares]), 6) |> println
