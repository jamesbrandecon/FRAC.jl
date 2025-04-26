using Documenter
using FRAC

makedocs(
    sitename = "FRAC",
    format = Documenter.HTML(),
    modules = [FRAC], 
    strict = false,
    pages = [
        "Home" => "index.md",
        "API" => "api.md"]
)
