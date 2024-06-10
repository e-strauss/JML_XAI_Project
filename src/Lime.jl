using XAIBase

# Write your package code here.
struct LIME{M} <: AbstractXAIMethod 
    model::M    
end

function (method::LIME)(input, output_selector::AbstractOutputSelector)
    output = method.model(input)
    output_selection = output_selector(output)

    val = rand(size(input)...) #TODO: use Lime-base for explanation
    extras = nothing
    return Explanation(val, output, output_selection, :MyMethod, :attribution, extras)
end

export LIME