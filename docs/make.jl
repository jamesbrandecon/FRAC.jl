using Documenter
using FRAC

makedocs(
    sitename = "FRAC.jl",
    format = Documenter.HTML(),
    modules = [FRAC], 
    warnonly = true,
    pages = [
        "Home" => "index.md",
        "API" => "api.md"]
)

deploydocs(
    repo = "github.com/jamesbrandecon/FRAC.jl.git",
    push_preview = true,
    devbranch = "main"
)
