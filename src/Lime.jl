using XAIBase
include("Lime-base.jl")
include("Lime-image.jl")
include("SHAP-Kernel.jl")

struct LIME{M, K, l} <: AbstractXAIMethod 
    model::M
    kernel::K
    lasso::l

end

LIME(M) = LIME(M, exponential_kernel, true)

#implementation of LIME XAI method based on https://arxiv.org/pdf/1602.04938
function (method::LIME)(input, output_selector::AbstractOutputSelector)
    output = method.model(input)
    output_selection = output_selector(output)

    #select first input
    input = input[:,:,:,1]

    val = explain_instance(input, method.model, output_selection[1]; kernel=method.kernel, lasso=method.lasso)
    extras = nothing
    return Explanation(val, output, output_selection[1], :MyMethod, :attribution, extras)
end

export LIME