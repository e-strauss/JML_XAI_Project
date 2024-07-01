using XAIBase
include("Lime-base.jl")
include("Lime-image.jl")

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

    #select first input
    input = input[:,:,:,1]

    val = explain_instance(input, method.model, output_selection[1])
    extras = nothing
    return Explanation(val, output, output_selection[1], :MyMethod, :attribution, extras)
end

export LIME