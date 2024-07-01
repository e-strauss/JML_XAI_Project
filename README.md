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
using Metalhead: ResNet
using JML_XAI_PROJECT
using Images

img = load("data/n01742172_boa_constrictor.JPEG")

display(img)
img = permutedims(channelview(img),(3,2,1))
img = reshape(img, size(img)..., 1)
input = Float32.(img)

model = ResNet(18; pretrain = true);
model = model.layers;analyzer = LIME(model)
expl = analyze(input, analyzer);
print("Label: ", argmax(expl.output[:,1]) - 1)
heat = heatmap(expl.val)
display(heat)
```
