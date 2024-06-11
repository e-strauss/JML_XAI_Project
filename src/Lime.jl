using XAIBase
include("Lime-base.jl")

struct LIME{M} <: AbstractXAIMethod 
    model::M    
end


#implementation of LIME XAI method based on https://arxiv.org/pdf/1602.04938
function (method::LIME)(input, output_selector::AbstractOutputSelector)
    output = method.model(input)
    output_selection = output_selector(output)

    samples = reshape(input, 1, size(input)...)
    labels = transpose(output)
    kernel_fn = identity
    max_features = length(input)
    distances = [0]
    #TODO:  replace "dummy" values with meaniningful values
    #       - create actual samples around the point input, compute labels and distances 
    #       - specify a kernel function that transforms an array of distances into an array of proximity values
    #       - specify a maximum num feature of features for the simpliefied model
    val = explain_instance_with_data(samples, labels, distances, kernel_fn, output_selection[1], max_features)
    extras = nothing
    return Explanation(val, output, output_selection, :MyMethod, :attribution, extras)
end

export LIME