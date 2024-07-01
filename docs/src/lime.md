# LIME

For the example, we load an image from the [imagenet-sample-images](https://github.com/EliSchwartz/imagenet-sample-images/tree/master) repository.

```@example implementations
using ExplainableAI
using Flux
using Metalhead: ResNet
include("../src/Lime.jl")
#using JML_XAI_Project
using CSV
using DataFrames
using Images

img = load("n01742172_boa_constrictor.JPEG")
```


The image is then pre-processed.

```@example implementations
img = permutedims(channelview(img),(3,2,1))
img = reshape(img, size(img)..., 1)
input = Float32.(img)
```
The next step is to initialize a pre-trained ResNet model and apply LIME to it.

```@example implementations
model = ResNet(18; pretrain = true);
model = model.layers;
analyzer = LIME(model);
expl = analyze(input, analyzer);
```
The generated explanation can now be displayed as a heat map

```@example implementations
using VisionHeatmaps
heatmap(expl.val)
```
To find out the generated corresponding label of the image, we output the number of the label, which can be looked up in the [linked text file](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a).

```@example implementations
print("Label: ", argmax(expl.output[:,1]) - 1)
```

