using Test
using Images
using ImageSegmentation: felzenszwalb

include("../src/Lime-images.jl")

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

    dumb_classifier(input) = "duck"

    data, labels = data_labels(img, img_white, segments, dumb_classifier, 2)

    @test typeof(data) === Matrix{Int64}
    @test typeof(labels) === Vector{Any}
end

@testset "pairwise_distance-function" begin
    A = zeros(4, 2)
    x1 = [1;; 0]
    x2 = ones(1,2)
    @test pairwise_distance(A, x1) == ones(4)
    @test pairwise_distance(A, x2) == (ones(4).*2).^0.5
end