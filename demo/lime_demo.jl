using ExplainableAI
using Metalhead: ResNet
using JML_XAI_Project
using Images
using VisionHeatmaps


#img = load("data/n01689811_alligator_lizard.JPEG")
img = load("data/n01742172_boa_constrictor.JPEG")
# img = load("data/dogs.png")
# img = RGB.(img)

display(img)
img = permutedims(channelview(img),(3,2,1))
img = reshape(img, size(img)..., 1)
input = Float32.(img)
@info size(input)

model = ResNet(18; pretrain = true);
model = model.layers;analyzer = LIME(model)
expl = analyze(input, analyzer);
print("Label: ", argmax(expl.output[:,1]) - 1)
heat = heatmap(expl.val)
display(heat)