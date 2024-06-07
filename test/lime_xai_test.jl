using ExplainableAI
using Flux
using BSON
using MLDatasets
using JML_XAI_Project

x, y = MNIST(Float32, :test)[10]
input = reshape(x, 28, 28, 1, :);

input_rgb = repeat(input, 1, 1, 3, 1)

model = BSON.load("../src/model.bson", @__MODULE__)[:model]
analyzer = LIME(model)
expl = analyze(input, analyzer);

@testset "LIME XAI TEST: same model output" begin
    @test expl.output == model(input)
end