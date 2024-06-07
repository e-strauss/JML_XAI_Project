using JML_XAI_Project
using Test

@testset "JML_XAI_Project.jl" begin
    include("getting-started-test.jl")
    include("lime_xai_test.jl")
end
