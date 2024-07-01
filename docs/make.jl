using JML_XAI_Project
using Documenter

DocMeta.setdocmeta!(JML_XAI_Project, :DocTestSetup, :(using JML_XAI_Project); recursive=true)

makedocs(;
    modules=[JML_XAI_Project],
    authors="Elias Strauss lathancer@gmx.de",
    sitename="JML_XAI_Project.jl",
    format=Documenter.HTML(;
        canonical="https://e-strauss.github.io/JML_XAI_Project.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Getting Started" => [
            "Lime" => "lime.md",
            "SHAP"     => "shap.md"
        ],
        "Important Functions" => "api.md"
    ],
)

deploydocs(;
    repo="github.com/e-strauss/JML_XAI_Project.jl",
    devbranch="main",
)
