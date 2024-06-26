using Test
#using Images, ImageView
using ImageSegmentation: felzenszwalb

include("../src/Lime-images.jl")

segmentation_fn = default_segmentation_function("felzenszwalb")

desired_function(img) = labels_map(felzenszwalb(img, 10, 100))


@testset "import-correct-segmentation-function" begin
    #enable the test again once it's working
    #@test segmentation_fn == desired_function
end