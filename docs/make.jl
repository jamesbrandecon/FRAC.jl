using Documenter
using FRAC

makedocs(
    sitename = "FRAC",
    format = Documenter.HTML(),
    modules = [FRAC]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
