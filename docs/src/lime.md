# LIME Example

For the example, we load an image from the [imagenet-sample-images](https://github.com/EliSchwartz/imagenet-sample-images/tree/master) repository. An explanation will be generated for the image later on. 

```@example implementations
using ExplainableAI
using Metalhead: ResNet
using JML_XAI_Project
using Images
using VisionHeatmaps

img = load("images/dog.jpeg")
```


## Image pre-processing
The image is processed in order to use it as input for a model.

```@example implementations
imgVec = permutedims(channelview(img),(3,2,1)); 
imgVec = reshape(imgVec, size(imgVec)..., 1);
input = Float32.(imgVec);
nothing # hide
```

## Generation of the explanation 
The next step is to initialize a pre-trained ResNet model and apply LIME to it.

!!! info
    Any classifier or regressor can be used at this point.

```@example implementations
model = ResNet(18; pretrain = true);
model = model.layers;
analyzer = LIME(model);
expl = analyze(input, analyzer);
```

## Visualize explanaition
The generated explanation can now be displayed as a heat map.

```@example implementations
using VisionHeatmaps
heatmap(expl.val)
```

!!! info
    The following code generates a clearer representation of the explanation.

```@example implementations
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
```

```@example implementations
generate_heatmap(expl, img=img, overlay=true, blurring=true)
```

## Label 

To find out the generated corresponding label of the image, we output the number of the label, which can be looked up in the [linked text file](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a).

```@example implementations
print("Label: ", argmax(expl.output[:,1]) - 1)
```