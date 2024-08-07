```@meta
CurrentModule = JML_XAI_Project
```

# JML_XAI_Project - LIME and SHAP for Julia
This package implements the explainable AI methods [LIME](https://arxiv.org/abs/1602.04938) and [SHAP](https://arxiv.org/pdf/1705.07874) using [XAIBase.jl](https://julia-xai.github.io/XAIDocs/XAIBase/stable/). LIME and SHAP are **model-agnostic** explainable AI methods, so they can be used to explain different kinds of models.
The JMLXAIproject package provides explanations for image inputs and visualizes them as heatmaps.

| **Input**                                  | **Output - Lime** |**Output - Shap** |
|:--------------------------------------------- |:------------------------------:|:------------------------------:|
| ![](images/dog.jpeg)                          | ![](images/heatMap.jpg)               | ![](images/heatMapShap.jpg)               |



### Getting started
To familiarise yourself with the package, there are sample implementations of LIME & SHAP in the documentation. On the other hand, there is the subfolder "demo" in the JML_XAI_Project package folder, from which "lime_demo.jl" can be executed, which runs LIME or SHAP.

!!! warning
    To use the package please add the following code to your environment

```julia
using Pkg
Pkg.add(url="https://github.com/e-strauss/JML_XAI_Project.jl")
```

Documentation for [JML_XAI_Project](https://github.com/e-strauss/JML_XAI_Project.jl).





