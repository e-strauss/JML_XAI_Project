using XAIBase
include("Lime-base.jl")

"""
    struct LIME{M} <: AbstractXAIMethod

The `LIME` (Local Interpretable Model-agnostic Explanations) struct is used to create an instance of the LIME method for explainable AI (XAI). This method provides local explanations for the predictions of any machine learning model by approximating the model's behavior in the vicinity of a specific input.

# Fields
- `model::M`: The machine learning model to be explained. The model should be callable with an input to produce an output.
"""
struct LIME{M} <: AbstractXAIMethod 
    model::M    
end

# TODO: docstring
#implementation of LIME XAI method based on https://arxiv.org/pdf/1602.04938
function (method::LIME)(input, output_selector::AbstractOutputSelector)
    output = method.model(input)
    output_selection = output_selector(output)

    input = vec(input)

    samples = reshape(input, 1, size(input)...)
    labels = transpose(output)
    kernel_fn = (x) -> 1 .- x
    max_features = length(input) / 6
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