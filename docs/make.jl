using Documenter
using FRACDemand

makedocs(
    sitename = "FRACDemand.jl",
    format = Documenter.HTML(),
    modules = [FRACDemand], 
    warnonly = true,
    pages = [
        "Home" => "index.md",
        "API" => "api.md"]
)

deploydocs(
    repo = "github.com/jamesbrandecon/FRACDemand.jl.git",
    push_preview = true,
    devbranch = "main"
)
