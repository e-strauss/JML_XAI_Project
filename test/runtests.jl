using JML_XAI_Project
using Test
using ExplainableAI
using Flux
using BSON
using Metalhead: ResNet
using CSV
using DataFrames
using Images
using VisionHeatmaps
using ImageSegmentation: felzenszwalb

@testset "JML_XAI_Project.jl" begin
    #component tests:
    include("lime_image_test.jl")
    include("lime_feature_selection_test.jl")
    include("weight_feature_test.jl")
    include("train_ridge_regressor_test.jl")


    #end-to-end tests:
    include("lime_xai_test.jl")
end
