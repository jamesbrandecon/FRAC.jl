using Documenter
using FRAC

makedocs(
    sitename = "FRAC",
    format = Documenter.HTML(),
    modules = [FRAC], 
    warnonly = true,
    pages = [
        "Home" => "index.md",
        "API" => "api.md"]
)
