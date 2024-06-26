using JML_XAI_Project
using Test


@testset "JML_XAI_Project.jl" begin
    #sanity check on the getting started tutorial
    #include("getting-started-test.jl")
    
    include("lime_feature_selection_test.jl")
    include("lime_xai_test.jl")
    include("Lime-images-tests.jl")
end
