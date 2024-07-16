# JML_XAI_Project

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://e-strauss.github.io/JML_XAI_Project/dev/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://e-strauss.github.io/JML_XAI_Project/dev/)
[![Build Status](https://github.com/e-strauss/JML_XAI_Project/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/e-strauss/JML_XAI_Project/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/e-strauss/JML_XAI_Project/branch/main/graph/badge.svg)](https://codecov.io/gh/e-strauss/JML_XAI_Project)

## About JML_XAI_Project
The JML_XAI_Project package implements the explainable AI methods SHAP and LIME for image inputs. The project was developed as part of the "Julia for Machine Learning" course at TU Berlin.

[Here](https://e-strauss.github.io/JML_XAI_Project/dev/) you can find the documentation.


## Installation

You can install `LIME` by adding it directly from our GitHub repository. Here are the steps:

1. Open Julia's REPL (the Julia command-line interface).

2. Press `]` to enter Pkg mode.

3. Run the following command to add the necessary LARS algorithm dependency:

```julia
pkg> add https://github.com/e-strauss/LARS.jl
```

4. Run the following command to add Lime:

```julia
pkg> add https://github.com/e-strauss/JML_XAI_Project
```


## Usage
```julia
using ExplainableAI
using Flux
using Metalhead: ResNet
using JML_XAI_Project
using Images
using VisionHeatmaps

#usage: include("src/xai-getting-started.jl")

#Plots heatmap
    #If overlay => heatmap on image (image in black and white)
    #If blurring => heatmap blurred
    #gaussSTD = standard deviation of gauss kernel, gaussSTD higher => more blurring
function generate_heatmap(map; img=nothing, overlay=false, blurring=false, gaussSTD=2)
    map = heatmap(map.val)

    if blurring == true
        gaussKern2 = ImageFiltering.KernelFactors.gaussian((gaussSTD,gaussSTD))
        map = ImageFiltering.imfilter(map, gaussKern2)
    end

    if overlay == true
        map = (0.5.*Gray.(img) + 0.5.*map)
    end

    return map
end



img = load("data/n01742172_boa_constrictor.JPEG")
display(img)
imgVec = permutedims(channelview(img),(3,2,1))
imgVec = reshape(imgVec, size(imgVec)..., 1)
input = Float32.(imgVec)
@info size(input)

model = ResNet(18; pretrain = true);

#Explanation Using LIME
model = model.layers;analyzer = LIME(model)

#Explanation Using SHAP with Model-Agnostic Kernel (lasso has to be set to false)
#model = model.layers;analyzer = LIME(model, agnostic_kernel, false)

expl = analyze(input, analyzer);
print("Label: ", argmax(expl.output[:,1]) - 1)
heat = generate_heatmap(expl, img=img, overlay=true, blurring=true)
display(heat)

#Save Heatmap to File
#save("C:/Users/USERNAME/Desktop/heatmap boa.png", heat)
```
