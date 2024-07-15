using ExplainableAI
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



#img = load("data/n01689811_alligator_lizard.JPEG")
img = load("data/n01742172_boa_constrictor.JPEG")
# img = load("data/dogs.png")
# img = RGB.(img)

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