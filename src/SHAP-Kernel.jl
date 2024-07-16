"""
    agnostic_kernel(simpleFeatures::Matrix{<:Real})

Calculates the weights used for building the ridge regression model (simplified model),
    when applying SHAP with a model-agnostic kernel.
    Implentation based on "A Unified Approach to Interpreting Model Predictions" (https://arxiv.org/abs/1705.07874)

# Parameters
- `simpleFeatures::Matrix{<:Real}`: The simplified features used for building the simlified model.

# Returns
- `Vector{Float64}`: A vector of weights with one weight for each simplified sample.

"""
function agnostic_kernel(simpleFeatures:: Matrix{<:Real})
    weights = zeros(1)

    for i in 1:size(simpleFeatures)[1]
        
        z = simpleFeatures[i,:]
        M = length(z)
        zAbsolute = count(!iszero, z)

        weight = (M - 1)/(binomial(big(M), zAbsolute)*zAbsolute*(M - zAbsolute))
        push!(weights, weight)
    end
    
    return weights
end