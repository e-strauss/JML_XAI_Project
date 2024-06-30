using ExplainableAI
using Flux
using Metalhead: ResNet
#using JML_XAI_Project
include("../src/Lime.jl")
using CSV
using DataFrames
using Images
using VisionHeatmaps

#usage: include("src/xai-getting-started.jl")



img = load("data/n01443537_goldfish.JPEG")
img = permutedims(channelview(img),(2,3,1))
img = reshape(img, size(img)..., 1)
input = Float32.(img[1:32,1:32,:,1:1])
@info size(input)

model = ResNet(18; pretrain = true);
model = model.layers;analyzer = LIME(model)
expl = analyze(input, analyzer);

heat = heatmap(expl.val)
display(heat)

println("done")