using XAIBase
include("Lime-base.jl")
include("Lime-image.jl")
include("SHAP-Kernel.jl")

struct LIME{M} <: AbstractXAIMethod 
    model::M    
end


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