using Test
using Images
using ImageSegmentation: felzenszwalb
using CSV
using DataFrames

include("../src/Lime-image.jl")

@testset "data_labels function simple test" begin

    img = load("../data/4x4_pixel.jpg")

    segments = [1 2 2 3
                1 1 2 3
                4 4 3 3
                4 4 5 6]
    # 1: orange
    # 2: pink
    # 3: yellow
    # 4: blue
    # 5: red
    # 6: green 

    img_white = load("../data/4x4_pixel_all_white.jpg")

    img = RGB{Float32}.(colorview(RGB, img))

    img_white = RGB{Float32}.(colorview(RGB, img_white))

    dumb_classifier(input) = ["duck";;]
    
    data, labels = data_labels(img, img_white, segments, dumb_classifier, 2)

    @test typeof(data) === Matrix{Int64}
    @test typeof(labels) === Matrix{String}
end

@testset "pairwise_distance-function" begin
    A = zeros(4, 2)
    B = [1:4 zeros(4)]
    C = reshape([0:49;;], 5,10)'
    x1 = [1;; 0]
    x2 = ones(1,2)

    #computed by scikitlearn
    df = CSV.read("../data/cosine_distance.csv", DataFrame, types=Float32)
    cos_dist = Matrix(df)

    @test pairwise_distance(A, x1, "euclidian") == ones(4)
    @test pairwise_distance(A, x2, "euclidian") == (ones(4).*2).^0.5
    @test cosine_similiarity(B, x1) == [ones(4);;]
    @test pairwise_distance(B, x1) == zeros(4)
    @test pairwise_distance(C, ones(1,5)) ≈ reshape(cos_dist, :)
end