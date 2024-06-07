using Test
using ExplainableAI
using Flux
using BSON
using MLDatasets
using RelevancePropagation

x, y = MNIST(Float32, :test)[10]
input = reshape(x, 28, 28, 1, :);

model = BSON.load("../src/model.bson", @__MODULE__)[:model]
analyzer = LRP(model)
expl = analyze(input, analyzer);


@testset "xai-getting-started" begin
    @test expl.output == model(input)
end