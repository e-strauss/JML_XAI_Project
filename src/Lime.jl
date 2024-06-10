using XAIBase
include("Lime-base.jl")

struct LIME{M} <: AbstractXAIMethod 
    model::M    
end

function (method::LIME)(input, output_selector::AbstractOutputSelector)
    output = method.model(input)
    output_selection = output_selector(output)

    val = rand(size(input)...) #TODO: use Lime-base for explanation
    #val = explain_instance_with_data()
    extras = nothing
    return Explanation(val, output, output_selection, :MyMethod, :attribution, extras)
end

export LIME