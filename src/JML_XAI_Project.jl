module JML_XAI_Project
using Statistics
using XAIBase
using ImageSegmentation: felzenszwalb
using Images: labels_map,colorview,channelview, RGB, Gray, red, green, blue, N0f8


import LinearAlgebra.I
import LinearAlgebra.Diagonal
import LARS.lars

include("Lime-image.jl")
include("Lime-base.jl")
include("Lime.jl")
include("SHAP-Kernel.jl")


export LIME,
weighted_data,
feature_selection,
train_ridge_regressor,
explain_instance_with_data,
explain_instance,
create_fudged_image,
default_segmentation_function,
data_labels,
euclidian_distance,
cosine_similiarity,
pairwise_distance,
exponential_kernel,
agnostic_kernel

end
