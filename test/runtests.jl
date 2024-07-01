using JML_XAI_Project
using Test


@testset "JML_XAI_Project.jl" begin
    #component tests:
    include("lime_image_test.jl")
    include("lime_feature_selection_test.jl")
    include("weight_feature_test.jl")
    include("train_ridge_regressor_test.jl")


    #end-to-end tests:
    include("lime_xai_test.jl")
    include("test_train_ridge_regressor.jl")
    include("Lime-images-tests.jl")
    include("test_agnostic_kernel.jl")
end
