### A Pluto.jl notebook ###
# v0.16.1

using Markdown
using InteractiveUtils

# ╔═╡ 8cef9d74-457a-11ec-1874-0b8fe7c58775
# Activate a clean environment, install necessary packages
begin
    import Pkg
    Pkg.activate(mktempdir())

    Pkg.add([
        Pkg.PackageSpec(name="Plots"),
        Pkg.PackageSpec(name="PlutoUI"),
		Pkg.PackageSpec(name="DataFrames"),
		Pkg.PackageSpec(name="BenchmarkTools"),
		Pkg.PackageSpec(url="https://github.com/jamesbrandecon/FRAC.jl.git"),
		Pkg.PackageSpec(url = "https://github.com/mthelm85/PlutoDataTable.jl.git")	
    ])

    using Plots
    using PlutoUI
	using PlutoDataTable
	using FRAC 
	using LinearAlgebra
	using Statistics
	using BenchmarkTools
	using Plots
	using DataFrames
	
	plotly();
end

# ╔═╡ 565d9692-240c-4f61-9e43-d298c1c42e8c
md"**Pluto Notebook to Demonstrate FRAC.jl Example**"

# ╔═╡ 0c2d13de-9f76-48c6-a3c6-1c01e379103b
	# In this example, we simulate demand for `T` markets, and assume the researcher
	# wants to allow for preferences to differ in `B` separate time periods/geographic
	# regions.

	# This file performs the following:
	# -- Simulate demand data for many markets with varying numbers of products
	# -- Calculate IVs
	# -- Estimate FRAC models and calculate various price elasticities
	# -- Extract and plot results of all models

# ╔═╡ fc9c71dc-30ee-45ee-86c6-57dd6bc55fe7
begin
	B = 100; # number of separate models to estimate
	T = 20000; # number of markets/2

	J1 = 2; # number of products in half of markets
	J2 = 4;

	# Simulate demand with two characteristics, both with random coefficients
	# Run benchmarks for multiple FRAC models
	s,p,z,x = simulate_logit(J1,T,[-0.4 0.1], [0.5 0.5], 0.3);
	s2,p2,z2,x2 = simulate_logit(J2,T,[-0.4 0.1], [0.5 0.5], 0.3);

	# Reshape data into desired DataFrame, add necessary IVs
	df = toDataFrame(s,p,z,x);
	df = reshape_pyblp(df);
	df2 = reshape_pyblp(toDataFrame(s2,p2,z2,x2));
	df2[!,"product_ids"] = df2.product_ids .+ 2;
	df2[!,"market_ids"] = df2.market_ids .+ T .+1;
	df = [df;df2];
	df[!,"by_example"] = mod.(1:size(df,1),B); # Generate variable indicating separate geographies
	df[!,"demand_instruments1"] = df.demand_instruments0.^2;
	df[!,"demand_instruments2"] = df.x .^2;

	# Add simple differentiation-style IVs: difference from market-level sum
	gdf = groupby(df, :market_ids);
	cdf = combine(gdf, names(df) .=> sum);
	cdf = select(cdf, Not(:market_ids_sum));
	sums = innerjoin(select(df, :market_ids), cdf, on = :market_ids);
	df[!,"demand_instruments3"] = (df.demand_instruments0 - sums.demand_instruments0_sum).^2;
	df[!,"demand_instruments4"] = (df.x - sums.x_sum).^2;

	df[!,"dummy_FE"] .= rand();
	df[!,"dummy_FE"] = (df.dummy_FE .> 0.5);
end


# ╔═╡ dc92136c-1fea-4825-aa60-2a058ceda836
# Estimate FRAC
# All models include two dimensions of random coefficients and two fixed effects
results = estimateFRAC(data = df, linear= "prices + x", nonlinear = "prices + x",    by_var = "by_example", fes = "product_ids + dummy_FE",
    se_type = "robust", constrained = false)

# ╔═╡ 47c68e86-eead-4301-bdf5-c309bd4101cb
# Calculate own-price elasticities of demand for all goods in all markets
own_elast = price_elasticities(frac_results = results, data = df,
    linear = "prices + x", nonlinear = "prices + x", which = "own",
    by_var="by_example");

# ╔═╡ a25656ed-3630-4a0d-bb6b-a24424fdcdbb
# Calculate elasticities of demand for each good with respect to the price of good 2
cross_elast = price_elasticities(frac_results = results, data = df,
    linear = "prices + x", nonlinear = "prices + x", which = "cross",
    by_var="by_example", product = 2);

# ╔═╡ ac3bccaa-a174-43d6-bbe3-326efe87f4a4
# Extraneous helper functions can be used to plot FRAC results with standard
# errors (when unconstrained). Here we show, for all estimated models,
#   (1) Mean price coefficient
#   (2) Variance of x coefficient
# (Horizontal lines indicate true value)
begin
plot1 = plotFRACResults(df, frac_results = results, var = "prices", plot = "scatter", by_var = "by_example")
plot!(plot1, ylims = (-1, 0), leg=false)
plot!([-0.4], linetype = :hline, linewidth = 2)
end

# ╔═╡ 530de509-df75-4ee8-818d-9a55c71b5f96
begin
plot2 = plotFRACResults(df, frac_results = results, var = "prices", param = "sd", plot = "scatter", by_var = "by_example")
plot!(plot2, ylims = (0, 0.5))
plot!([0.25], linetype = :hline, linewidth = 2)
end

# ╔═╡ 17ab959c-079e-4cbc-88c0-fead76fa4576
# Can also extract estimates, SEs, and groups (values of by_var) in the same order
alphas, alpha_se, byvals1 = extractEstimates(df, frac_results = results, param = "mean", by_var = "by_example")

# ╔═╡ 29867a9b-2eb9-4aa2-94a4-977a6cfc41d5
# By changing the keyword "param" to "sigmas", we extract estimates of the 
xsigmas, xsigma_se, byvals2 = extractEstimates(df, frac_results = results, var = "x", param = "sigmas", by_var = "by_example")

# ╔═╡ efb61664-a76f-4a27-b1ec-27fcd458b593
# When relevant, we can then easily pass estimates and group names into a DataFrame
begin
	table = DataFrame();
	table[!,"groups"] = byvals2;
	table[!,"alpha_estimates"] = alphas;
	table[!,"alpha_se"] = alpha_se;
	data_table(table; items_per_page = 10)
end

# ╔═╡ Cell order:
# ╟─565d9692-240c-4f61-9e43-d298c1c42e8c
# ╠═8cef9d74-457a-11ec-1874-0b8fe7c58775
# ╠═0c2d13de-9f76-48c6-a3c6-1c01e379103b
# ╠═fc9c71dc-30ee-45ee-86c6-57dd6bc55fe7
# ╠═dc92136c-1fea-4825-aa60-2a058ceda836
# ╠═47c68e86-eead-4301-bdf5-c309bd4101cb
# ╠═a25656ed-3630-4a0d-bb6b-a24424fdcdbb
# ╠═ac3bccaa-a174-43d6-bbe3-326efe87f4a4
# ╠═530de509-df75-4ee8-818d-9a55c71b5f96
# ╠═17ab959c-079e-4cbc-88c0-fead76fa4576
# ╠═efb61664-a76f-4a27-b1ec-27fcd458b593
# ╠═29867a9b-2eb9-4aa2-94a4-977a6cfc41d5
