module JML_XAI_Project

include("LimeTest.jl")
using ExplainableAI
using Flux
using BSON
using MLDatasets
using ImageCore, ImageIO, ImageShow
using RelevancePropagation
using VisionHeatmaps

index = 10
x, y = MNIST(Float32, :test)[10]

img = convert2image(MNIST, x)
display(img)

input = reshape(x, 28, 28, 1, :);

model = BSON.load("./model.bson", @__MODULE__)[:model]
analyzer = LRP(model)
expl = analyze(input, analyzer);

heat = heatmap(expl.val)
display(heat)

println("done")

# Write your package code here.
timestwo(x) =  x


export timestwo, lime

end
