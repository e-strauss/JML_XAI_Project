using Test
using ExplainableAI
using Flux
using BSON
using CSV
using DataFrames
using RelevancePropagation

df = CSV.read("../data/MNIST_input_9.csv", DataFrame, types=Float32)
x = Matrix(df)
y = 9
input = reshape(x, 28, 28, 1, :);

model = BSON.load("../data/model.bson", @__MODULE__)[:model]
analyzer = LRP(model)
expl = analyze(input, analyzer);


@testset "xai-getting-started" begin
    @test expl.output == model(input)
end