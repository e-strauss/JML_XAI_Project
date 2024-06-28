using JML_XAI_Project
using Test


@testset "JML_XAI_Project.jl" begin
    #component tests:
    include("Lime-images-tests.jl")
    include("lime_feature_selection_test.jl")
    include("weight_feature_test.jl")
    include("train_ridge_regressor_test.jl")


    #end-to-end tests:
    include("lime_xai_test.jl")
end
