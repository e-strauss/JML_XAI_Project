using Test
using Images, ImageView
using ImageSegmentation: felzenszwalb

include("../src/Lime-images.jl")

segmentation_fn = default_segmentation_function("felzenszwalb")

desired_function(img) = labels_map(felzenszwalb(img, 10, 100))


@testset "import-correct-segmentation-function" begin
    @test segmentation_fn === segmentation_fn
end